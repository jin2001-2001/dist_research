#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_cpu_hybrid_pp.py

This is the main experiment script for training Qwen-0.6B using a custom 
Hybrid Pipeline Parallelism framework built on top of PiPPy.

- Stage 0 (rank 0): embedding + layers 0–13
- Stage 1 (ranks 1 & 2): layers 14–27 + norm + lm_head (with DDP)

Key Features:
- CPU-only execution with 3 ranks
- Manual microbatch scheduling via custom action list
- Used for testing correctness and loss behavior of the hybrid PP runtime
"""


import argparse, os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torch.distributed as dist

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import time, math

from stage_with_mutiple_ranks import PipelineStage_with_mutiple_ranks
from schedule_runtime import PipelineScheduleRuntimeWithDirection
from pipelining_source_code.schedules import _Action, _ComputationType

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from simple_1F1B_Action import generate_1f1b_pipeline_actions
from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

# ==== 加载 Thinker（Omni-3B，只要文本主干 + 多模态编码器） ====
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from transformers import AutoProcessor

import torch, torch.nn as nn, torch.nn.functional as F

def build_causal(mask_len, device):
    return torch.triu(torch.full((mask_len, mask_len), float("-inf"), device=device), diagonal=1)\
                .unsqueeze(0).unsqueeze(0)  # [1,1,T,T]

def pack_modalities(text_embeds, audio_seq=None, vision_seq=None):
    """把三路序列拼成 [B, T_total, D]；若某一路是 [T, D] 则兜底补 batch 维"""
    def _ensure_3d(x):
        if x is None:
            return None
        return x if x.dim() == 3 else x.unsqueeze(0)

    seqs = [_ensure_3d(x) for x in [text_embeds, audio_seq, vision_seq] if x is not None]
    return torch.cat(seqs, dim=1) if len(seqs) > 1 else seqs[0]


class Stage0(nn.Module):
    """0a: Text embed  | 0b: Audio encoder → 2048 | 0c: Vision encoder → 2048"""
    def __init__(self, text_model, audio_enc=None, vision_enc=None, rotary_emb=None):
        super().__init__()
        self.embed_tokens = text_model.embed_tokens        # [B,T_txt] -> [B,T_txt,2048]
        self.audio_enc    = audio_enc                      # 原生编码器（内部会投到 2048）
        self.vision_enc   = vision_enc
        self.rotary_emb   = rotary_emb

    def forward(self, input_ids, attention_mask=None, vision_inputs=None, audio_inputs=None, **kwargs):
        """
        期望返回三模态拼好的 [B, T_total, D] 送入后续 LLM
        """
        B = input_ids.size(0)

        # ===== 文本侧（示例） =====
        text_emb = self.embed_tokens(input_ids)  # [B, T_text, D]
        # 其它文本预处理...

        # ===== 视觉侧 =====
        vision_seq = None
        if vision_inputs is not None:
            x = vision_inputs["pixel_values"]  # [B,C,T,H,W]
            grid = vision_inputs["grid_thw"]   # [B,3] with (T, H_patch, W_patch)

            # 编码器内部会把 [B,C,T,H,W] 展平为 token 序列 [B*Tv, D] 或 [Tv, D]
            # 注意：你的 self.vision_enc 接口若不同，请按实际调整
            vision_seq = self.vision_enc(x, grid)  # 常见返回: [Tv_total, D] 或 [B, Tv, D]

            # 统一为 [B, Tv, D]
            if isinstance(grid, torch.Tensor):
                smu = 4  # spatial_merge_unit = 4
                Tv_per = (grid[:, 0] * grid[:, 1] * grid[:, 2]) * smu  # [B]
                Tv0 = int(Tv_per[0].item())
                assert torch.all(Tv_per == Tv0), \
                    f"Visual token count per sample not equal in batch: {Tv_per.tolist()}"

                if vision_seq.dim() == 2:  # [Tv_total, D] 或 [Tv, D]
                    D = vision_seq.size(-1)
                    total = vision_seq.size(0)
                    if total == Tv0 * B:
                        vision_seq = vision_seq.view(B, Tv0, D)
                    elif total == Tv0 and B == 1:
                        vision_seq = vision_seq.view(1, Tv0, D)
                    else:
                        raise ValueError(f"Cannot reshape vision_seq {vision_seq.shape} with B={B}, Tv={Tv0}")
                elif vision_seq.dim() == 3:
                    pass  # 已是 [B, Tv, D]
                else:
                    raise ValueError(f"Unexpected vision_seq shape: {vision_seq.shape}")
            else:
                # 没有 grid（不推荐），兜底把 2D 序列加 batch 维
                if vision_seq.dim() == 2:
                    vision_seq = vision_seq.unsqueeze(0)

        # ===== 音频侧（如需；确保为 [B, T_audio, D]） =====
        audio_seq = None
        if audio_inputs is not None:
            audio_seq = self.audio_enc(audio_inputs)  # [B, T_audio, D]
            if audio_seq.dim() == 2:
                audio_seq = audio_seq.unsqueeze(0)

        # ===== 三模态拼接 =====
        hidden = pack_modalities(text_emb, audio_seq, vision_seq)  # -> [B, T_total, D]
        return hidden

class Stage1(nn.Module):
    def __init__(self, text_model, L1, rotary_emb=None):
        super().__init__()
        self.layers = nn.ModuleList(text_model.layers[:L1])
        self.rotary_emb = rotary_emb
    def forward(self, hidden, attn_mask, position_ids, pos_emb=None):
        # 若上游未给 RoPE，可在此处现算：pos_emb = self.rotary_emb(hidden, position_ids)
        for blk in self.layers:
            hidden = blk(hidden_states=hidden,
                         attention_mask=attn_mask,
                         position_ids=position_ids,
                         position_embeddings=pos_emb,
                         output_attentions=False,
                         use_cache=False)[0]
        return hidden.contiguous(), attn_mask, position_ids, pos_emb

class Stage2(nn.Module):
    def __init__(self, text_model, L1, L2, rotary_emb=None):
        super().__init__()
        self.layers = nn.ModuleList(text_model.layers[L1:L2])
        self.rotary_emb = rotary_emb
    def forward(self, hidden, attn_mask, position_ids, pos_emb=None):
        for blk in self.layers:
            hidden = blk(hidden_states=hidden,
                         attention_mask=attn_mask,
                         position_ids=position_ids,
                         position_embeddings=pos_emb,
                         output_attentions=False,
                         use_cache=False)[0]
        return hidden.contiguous(), attn_mask, position_ids, pos_emb

class Stage3(nn.Module):
    def __init__(self, full_thinker, text_model, L2):
        super().__init__()
        self.layers = nn.ModuleList(text_model.layers[L2:])
        self.norm   = text_model.norm
        self.lm_head = full_thinker.lm_head   # Thinker 自带 LM 头（不含 Talker）
    def forward(self, hidden, attn_mask, position_ids, pos_emb=None):
        for blk in self.layers:
            hidden = blk(hidden_states=hidden,
                         attention_mask=attn_mask,
                         position_ids=position_ids,
                         position_embeddings=pos_emb,
                         output_attentions=False,
                         use_cache=False)[0]
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits


parser = argparse.ArgumentParser()
parser.add_argument("--train_steps", type=int, default=None,
                    help="The total number of steps for training. If omitted, run the entire DataLoader.")
parser.add_argument("--batch_size", type=int,
                    default=int(os.getenv("BATCH_SIZE", 16)),
                    help="The batch size of each rank (the environment variable BATCH_SIZE can be overridden)")
parser.add_argument("--microbatch_num", type=int,
                    default=int(os.getenv("MICROBATCH_NUM", 4)),
                    help="Micro-batch number (the environment variable MICROBATCH_NUM can be overridden)")
parser.add_argument("--sudo_pass", default=os.getenv("SUDO_PASS"),
                    help='Write the password of root')
parser.add_argument("--upstream", default=os.getenv("upstream"),
                    help='Write the upstream in mbps')
args = parser.parse_args()
def main():

    dist.init_process_group("gloo", init_method="env://")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    device = torch.device("cpu")     

    MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    
    # 便捷访问
    text_model   = thinker.model           # 纯解码器主干（有 embed_tokens / layers / norm）
    audio_enc    = getattr(thinker, "audio_tower", None)
    vision_enc   = getattr(thinker, "visual", None)
    rotary_emb   = getattr(text_model, "rotary_emb", None)
    vocab_size   = tok.vocab_size

    # 自动切分点
    L  = len(text_model.layers)
    L1 = L // 3
    L2 = (2 * L) // 3
    
    if rank == 0:
        stage_mod = Stage0(text_model, audio_enc, vision_enc, rotary_emb)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=0,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=None, this_group=[0], next_group=[1])
        
    elif rank == 1:
        stage_mod = Stage1(text_model, L1, rotary_emb)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=1,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=[0], this_group=[1], next_group=[2])
    elif rank == 2:
        stage_mod = Stage2(text_model, L1, L2, rotary_emb)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=2,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=[1], this_group=[2], next_group=[3])
    elif rank == 3:
        stage_mod = Stage3(thinker, text_model, L2)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=3,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=[2], this_group=[3], next_group=None)
    
    del thinker                        
    import gc; gc.collect()

    raw = load_dataset("jxie/flickr8k", split="train")
    
    def pick_caption(example):
        text = example.get("caption_0", None)
        if text is None:
            caps = example.get("captions")  # 某些镜像用这个字段
            text = caps[0] if isinstance(caps, list) and caps else ""
        return {"text": text}

    # 仅保留 image 和我们新增的 text 列
    keep_cols = {"image", "text"}
    raw = raw.map(pick_caption, remove_columns=[c for c in raw.column_names if c not in keep_cols])

    # 文本分词：一图一文，直接截断/填充到定长 block；labels=输入右移时的目标
    block = 128
    def tok_fn(batch):
        out = tok(batch["text"],
                return_attention_mask=False,
                truncation=True,
                max_length=block,
                padding="max_length")
        # 语言建模：labels 与 input_ids 相同（loss_fn 内部会做左移对齐）
        out["labels"] = out["input_ids"].copy()
        return out

    ds = raw.map(tok_fn, batched=True)

    from PIL import Image, ImageOps

    def _round_to_multiple(x: int, m: int) -> int:
        """将数字向上取整到m的倍数"""
        return max(m, int(math.ceil(x / m) * m))

    def _pad_to_valid_size(img: Image.Image) -> Image.Image:
        """
        将图像填充到符合 Qwen2.5-Omni 视觉编码器要求的尺寸。
        
        Qwen2.5-Omni 的要求：
        - patch_size = 14
        - spatial_merge_size = 2  
        - spatial_merge_unit = 4
        - 最终 (H/14) 和 (W/14) 必须是 4 的倍数
        """
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        
        # 计算需要的最小尺寸
        # 1. 必须是 14 的倍数（patch_size）
        # 2. H/14 和 W/14 必须是 4 的倍数（spatial_merge_unit）
        # 这意味着 H 和 W 必须是 56 (14*4) 的倍数
        
        # 直接填充到 56 的倍数
        new_w = _round_to_multiple(w, 56)
        new_h = _round_to_multiple(h, 56)
        
        # 进一步确保 grid 维度是 4 的倍数
        grid_h = new_h // 14
        grid_w = new_w // 14
        
        # 如果 grid 维度不是 4 的倍数，继续增加
        while grid_h % 4 != 0:
            new_h += 56
            grid_h = new_h // 14
        
        while grid_w % 4 != 0:
            new_w += 56
            grid_w = new_w // 14
        
        # 为了额外的安全性，确保最终尺寸至少是 112x112
        # （这确保有足够的 patches 进行 spatial merging）
        new_w = max(new_w, 112)
        new_h = max(new_h, 112)
        
        pad_right = new_w - w
        pad_bottom = new_h - h
        
        if pad_right == 0 and pad_bottom == 0:
            return img
        
        # 填充图像（右边和下边）
        return ImageOps.expand(img, border=(0, 0, pad_right, pad_bottom), fill=0)

    def collate_fn(batch):
        """
        batch: List[dict]，每个样本里至少包含文本与图像/视频的原始输入。
        这里假设你在外层已经用 AutoProcessor 做了基础处理，或你能在这里拿到像素张量。
        输出:
        input_ids, attention_mask, vision_inputs = {"pixel_values": [B,C,T,H,W], "grid_thw": [B,3]}
        其它字段按你现有逻辑返回
        """
        # ===== 文本侧（示例，你可沿用自己原有实现） =====
        input_ids_list = [item["input_ids"] for item in batch]
        attention_mask_list = [item["attention_mask"] for item in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        # ===== 视觉侧：统一 pixel_values 与 grid_thw 语义 =====
        # 假设每个 item["pixel_values"] 是 [C,H,W]（图像）或 [C,T,H,W]（视频/多帧）
        pixels = []
        for item in batch:
            pv = item["pixel_values"]
            if pv.ndim == 3:           # [C,H,W] -> [C,1,H,W]
                pv = pv.unsqueeze(1)
            elif pv.ndim == 4:         # [C,T,H,W]
                pass
            else:
                raise ValueError(f"Unexpected pixel_values shape: {pv.shape}")
            pixels.append(pv)

        # pad/stack 到 [B,C,T,H,W]（要求同 H,W；若不一致，请在前处理统一 resize/pad）
        C = pixels[0].size(0)
        T = pixels[0].size(1)
        H = pixels[0].size(2)
        W = pixels[0].size(3)
        for pv in pixels:
            assert pv.size(0) == C and pv.size(2) == H and pv.size(3) == W, \
                f"All samples must have the same C,H,W; got {pv.shape} vs ({C},{H},{W})"
        pixel_values = torch.stack(pixels, dim=0)  # -> [B,C,T,H,W]
        B = pixel_values.size(0)

        # 以 Qwen2.5-Omni 约定：patch_size=14，spatial_merge_unit=4（2x2 融合）
        grid_h = H // 14
        grid_w = W // 14
        # 2×2 merge 要求 H_patch、W_patch 都为偶数
        assert (grid_h % 2 == 0) and (grid_w % 2 == 0), \
            f"H,W after 14x14 patching must be even: grid_h={grid_h}, grid_w={grid_w} (H={H}, W={W})"

        # 统一构造标准语义的 grid_thw: (T, H_patch, W_patch)（每样本相同）
        grid_thw = torch.tensor([[T, grid_h, grid_w]] * B, dtype=torch.long)

        print(f"[collate] pixel_values [B={B},C={C},T={T},H={H},W={W}] -> grid_thw[0]={grid_thw[0].tolist()}")

        vision_inputs = {"pixel_values": pixel_values, "grid_thw": grid_thw}

        # ===== 其它你已有的字段按需返回 =====
        batch_out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "vision_inputs": vision_inputs,
            # 例如 labels/targets 等...
        }
        return batch_out


    batch_size = args.batch_size
    microbatch_num = args.microbatch_num

    # 仅 rank==0 负责取数（你原先的做法）
    if rank == 0:
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn
        )

    # ==== 语言模型损失（与原来相同：next-token LM）====
    def loss_fn(output, target):
        if output is None or target is None:
            return None
        vocab_size = output.size(-1)
        Ttxt = target.size(1)
        output = output[:, :Ttxt, :]          # <-- 新增
        output = output[:, :-1, :].reshape(-1, vocab_size)
        target = target[:, 1:].reshape(-1)
        return F.cross_entropy(output, target)


    # ==== 调度与动作表（保持你现有逻辑）====
    sched = PipelineScheduleRuntimeWithDirection([stage], n_microbatches=microbatch_num,
                                                loss_fn=loss_fn, root_pass=args.sudo_pass)
    actions = generate_1f1b_pipeline_actions(num_stages=4, num_microbatches=8, upstream = args.upstream)
    sched._load_actions(actions, format="compute_comms")

    # ==== 优化器（不变）====
    opt = optim.Adam(stage_mod.parameters(), lr=1e-4)
    prev_loss = None
    
    
    for epoch in range(1):
        if rank == 0:
            if args.train_steps is None:
                steps_tensor = torch.tensor(len(loader), device=device)
            else:
                steps_tensor = torch.tensor(args.train_steps, device=device)
            dist.broadcast(steps_tensor, src=0)
            data_iter = iter(loader)
            print(f"Total training steps: {steps_tensor.item()}")
        else:
            steps_tensor = torch.tensor(0, device=device)
            dist.broadcast(steps_tensor, src=0)

        total_steps = int(steps_tensor.item())

        if rank == 0:
            pbar = tqdm(
                total=int(total_steps),
                desc=f"Training Epoch {epoch+1}",
                unit="step",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            start_time = time.time()

        for step in range(total_steps):
            step_start_time = time.time()
            opt.zero_grad(set_to_none=True)

            if rank == 0:
                batch = next(data_iter)
                inp_ids = batch["input_ids"].to(device)             # [B, block]
                vis_pack = batch["vision_inputs"]
                vis_pack["pixel_values"] = vis_pack["pixel_values"].to(device)
                if torch.is_tensor(vis_pack["grid_thw"]):
                    vis_pack["grid_thw"] = vis_pack["grid_thw"].to(device)

                tgt = batch["labels"].to(device)                # [B, block]
                # 广播 label（仅文本部分需要参与 loss）
                dist.broadcast(tgt, src=0)

                # 传入流水：把 (input_ids, vision_inputs) 作为 Stage0 的输入
                sched.step(inp_ids, vision_inputs=vis_pack, target=tgt)

            else:
                # 其它 rank 只需要 label 的占位与广播
                tgt = torch.empty(batch_size, block, dtype=torch.long, device=device)
                dist.broadcast(tgt, src=0)
                sched.step(target=tgt)

            # 清理时间线（按你原逻辑）
            if (step + 1) % 50 == 0:
                try:
                    sched.timeline_rec.events.clear()
                except Exception:
                    pass

            opt.step()

            if rank == 0:
                step_time = time.time() - step_start_time
                tokens_processed = batch_size * block
                tokens_per_second = tokens_processed / step_time
                pbar.set_postfix({
                    'tokens/s': f'{tokens_per_second:.0f}',
                    'step_time': f'{step_time:.2f}s',
                    'lr': f'{opt.param_groups[0]["lr"]:.2e}'
                })
                pbar.update(1)

            cur_loss = getattr(sched, "last_step_loss", None)
            if cur_loss is not None and rank == 0:
                print(f"[rank0] step {step+1} loss {cur_loss:.4f}")
                prev_loss = cur_loss

            dist.barrier()

        if rank == 0:
            pbar.close()
            total_time = time.time() - start_time
            print(f"\nEpoch {epoch+1} completed in {total_time:.2f}s")
            print(f"Average speed: {total_steps / total_time:.2f} steps/s")

    # ========= 合并并保存（Thinker-only，四段聚合） =========
    # 所有 rank 参与三轮广播：src 分别为 1、2、3；rank0 收集，其他 rank 在自己回合发送
    recv1, recv2, recv3 = [None], [None], [None]

    buf = [stage_mod.state_dict()] if rank == 1 else [None]
    dist.broadcast_object_list(buf, src=1)
    if rank == 0: part1_state = buf[0]

    buf = [stage_mod.state_dict()] if rank == 2 else [None]
    dist.broadcast_object_list(buf, src=2)
    if rank == 0: part2_state = buf[0]

    buf = [stage_mod.state_dict()] if rank == 3 else [None]
    dist.broadcast_object_list(buf, src=3)
    if rank == 0: part3_state = buf[0]

    # rank0 本地的 Stage0 权重
    if rank == 0:
        print("\nMerging and saving model...")
        part0_state = stage_mod.state_dict()

        merged = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=True)
        merged_state = merged.state_dict()

        # --- Stage0: embed_tokens / rotary_emb / audio_encoder / vision_encoder ---
        for k, v in part0_state.items():
            if k.startswith("embed_tokens."):
                newk = "model.embed_tokens." + k[len("embed_tokens."):]
                merged_state[newk] = v
            elif k.startswith("rotary_emb."):
                newk = "model.rotary_emb." + k[len("rotary_emb."):]
                if newk in merged_state:  # 有的版本可能没有持久化该缓冲
                    merged_state[newk] = v
            elif k.startswith("audio_enc."):
                newk = "audio_tower." + k[len("audio_enc."):]
                if newk in merged_state:
                    merged_state[newk] = v
            elif k.startswith("vision_enc."):
                newk = "visual." + k[len("vision_enc."):]
                if newk in merged_state:
                    merged_state[newk] = v
            # Stage0 通常不包含 layers.*

        # 工具函数：把局部层 key 映射为全局层 key
        def _map_layer_key(local_key: str, global_offset: int) -> str:
            # local_key 形如: "layers.<i>.<rest>"
            parts = local_key.split(".")
            assert parts[0] == "layers", f"unexpected key {local_key}"
            li = int(parts[1]) + global_offset
            rest = ".".join(parts[2:])
            return f"model.layers.{li}.{rest}"

        # --- Stage1: 前 1/3 层（偏移 +0） ---
        for k, v in part1_state.items():
            if k.startswith("layers."):
                newk = _map_layer_key(k, 0)
                merged_state[newk] = v

        # --- Stage2: 中 1/3 层（偏移 +L1） ---
        for k, v in part2_state.items():
            if k.startswith("layers."):
                newk = _map_layer_key(k, L1)
                merged_state[newk] = v

        # --- Stage3: 后 1/3 层（偏移 +L2）+ norm + lm_head ---
        for k, v in part3_state.items():
            if k.startswith("layers."):
                newk = _map_layer_key(k, L2)
                merged_state[newk] = v
            elif k == "norm.weight":
                merged_state["model.norm.weight"] = v
            elif k == "norm.bias":
                if "model.norm.bias" in merged_state:
                    merged_state["model.norm.bias"] = v
            elif k == "lm_head.weight":
                merged_state["lm_head.weight"] = v
            elif k == "lm_head.bias":
                if "lm_head.bias" in merged_state:
                    merged_state["lm_head.bias"] = v

        # 加载合并权重并保存
        merged.load_state_dict(merged_state, strict=False)
        merged.save_pretrained("trained_qwen_pp")
        tok.save_pretrained("trained_qwen_pp")
        print("Saved to ./trained_qwen_pp")

    dist.destroy_process_group()



if __name__ == "__main__":
    main()