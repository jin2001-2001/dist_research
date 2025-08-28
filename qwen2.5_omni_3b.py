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
    """把三路序列拼成 [B, T_total, 2048]；你也可改成交错（按时间戳）。"""
    seqs = [x for x in [text_embeds, audio_seq, vision_seq] if x is not None]
    return torch.cat(seqs, dim=1) if len(seqs) > 1 else seqs[0]

class Stage0(nn.Module):
    """0a: Text embed  | 0b: Audio encoder → 2048 | 0c: Vision encoder → 2048"""
    def __init__(self, text_model, audio_enc=None, vision_enc=None, rotary_emb=None):
        super().__init__()
        self.embed_tokens = text_model.embed_tokens        # [B,T_txt] -> [B,T_txt,2048]
        self.audio_enc    = audio_enc                      # 原生编码器（内部会投到 2048）
        self.vision_enc   = vision_enc
        self.rotary_emb   = rotary_emb

    def forward(self, input_ids, audio_inputs=None, vision_inputs=None):
        if isinstance(input_ids, (tuple, list)):
            input_ids, vision_inputs = input_ids
        
        # 解包后兜底：若视觉是 [B,C,H,W]，补成 [B,C,1,H,W]
        if vision_inputs is not None and isinstance(vision_inputs, torch.Tensor) and vision_inputs.ndim == 4:
            vision_inputs = vision_inputs.unsqueeze(2)

        
        device   = input_ids.device
        bsz, Ttxt = input_ids.shape
        text_emb = self.embed_tokens(input_ids)

        # 可选：音频/视觉编码（如果此时就有对应输入）
        audio_seq = (self.audio_enc(audio_inputs)
             if (self.audio_enc is not None and audio_inputs is not None) else None)

        vision_seq = None
        if self.vision_enc is not None and vision_inputs is not None:
            if isinstance(vision_inputs, dict):
                vi = dict(vision_inputs)  # 浅拷贝，避免原地改
                # 兜底取图像张量
                # 在 vision_inputs 是 dict 的分支里，替换你现在的取值语句
                x = vi.get("pixel_values", None)
                if x is None:
                    x = vi.get("x", None)
                if x is None:
                    x = vi.get("images", None)
                if x is None:
                    x = vi.get("video", None)

                grid = vi.get("grid_thw", None)
                if grid is None:
                    grid = vi.get("image_grid_thw", None)
                if grid is None:
                    grid = vi.get("thw", None)

                assert x is not None,    "vision_inputs 缺少图像张量（pixel_values/x/images/video 任一）"
                assert grid is not None, "vision_inputs 缺少 grid_thw（或 image_grid_thw/thw）"

                if x.ndim == 4:
                    x = x.unsqueeze(2)  # -> [B,C,1,H,W]
                if isinstance(vision_inputs, dict):
                    grid = vision_inputs.get("grid_thw")
                    if torch.is_tensor(grid):
                        seq = grid[:, 0] * grid[:, 1] * grid[:, 2]
                        assert (seq % 4 == 0).all(), f"Visual seq must be divisible by 4, got grid={grid}"

                
                vision_seq = self.vision_enc(x, grid)  # 用位置实参调用


            elif isinstance(vision_inputs, (list, tuple)):
                assert len(vision_inputs) == 2, "vision_inputs 期望为 (x, grid_thw)"
                x, grid = vision_inputs
                if isinstance(x, torch.Tensor) and x.ndim == 4:
                    x = x.unsqueeze(2)
                if isinstance(vision_inputs, dict):
                    grid = vision_inputs.get("grid_thw")
                    if torch.is_tensor(grid):
                        seq = grid[:, 0] * grid[:, 1] * grid[:, 2]
                        assert (seq % 4 == 0).all(), f"Visual seq must be divisible by 4, got grid={grid}"

                vision_seq = self.vision_enc(x, grid)

            else:
                raise ValueError("vision_inputs 必须是 dict 或 (x, grid_thw)")


        # 它们的输出在官方 config 中会被映射到 2048 维，与你的文本隐向量同域（便于 concat）

        # 打包统一序列
        hidden = pack_modalities(text_emb, audio_seq, vision_seq)    # [B, T_total, 2048]
        B, T = hidden.shape[:2]

        # 位置与掩码（后续 stage 直接复用）
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1).contiguous()  # [B,T]
        attn_mask = build_causal(T, device).expand(B, 1, -1, -1).contiguous()                 # [B,1,T,T]

        # 可选：提前计算 RoPE tables（如果下游层需要传入）
        pos_emb = self.rotary_emb(hidden, position_ids) if self.rotary_emb is not None else None
        return hidden.contiguous(), attn_mask, position_ids, pos_emb

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

    # 视觉编码器的关键超参（不要硬编码 14 / 2 / 112）
    if vision_enc is not None:
        vit_cfg = getattr(vision_enc, "config", None)
        PATCH = int(getattr(vit_cfg, "patch_size", 14))
        SMS   = int(getattr(vit_cfg, "spatial_merge_size", 2))   # spatial_merge_unit = SMS*SMS (通常=4)
        WIN   = int(getattr(vit_cfg, "window_size", 112))        # ViT 的窗口（像素）
        # 为了让 (H/patch) 能被窗口网格整除，令 patch 网格的窗口步长：
        VIT_MERGE = WIN // (PATCH * SMS)  # 通常是 4（112 / (14*2)）
    else:
        PATCH, SMS, VIT_MERGE = 14, 2, 4

    
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

    def collate_fn(batch):
        # ---- 文本：用 tokenizer 统一 padding/truncation ----
        texts = [ex["text"] for ex in batch]
        tok_out = tok(
            texts,
            padding="max_length",      # 或者 True；你已有 block=128 可继续用 max_length=block
            truncation=True,
            max_length=128,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = tok_out["input_ids"]
        labels = input_ids.clone()

        # ---- 图像：逐张对齐到 (patch, window) 友好的网格，并在 batch 内对齐到同一尺寸 ----
        from PIL import Image, ImageOps
        import numpy as np
        to_tensor = lambda img: torch.from_numpy(np.array(img, dtype="float32").transpose(2,0,1)) / 255.0

        # 逐张：算出“对齐后”的目标 patch 网格（先对齐到 patch，再对齐到窗口网格）
        per_img = []
        for ex in batch:
            img: Image.Image = ex["image"].convert("RGB")
            H0, W0 = img.height, img.width

            # 先对齐到 patch：ceil 到能被 PATCH 整除
            Hp = ( (H0 + PATCH - 1) // PATCH )
            Wp = ( (W0 + PATCH - 1) // PATCH )

            # 再让 patch 网格能被 VIT_MERGE 整除（窗口网格大小，通常=4）
            if VIT_MERGE > 1:
                Hp = ((Hp + VIT_MERGE - 1) // VIT_MERGE) * VIT_MERGE
                Wp = ((Wp + VIT_MERGE - 1) // VIT_MERGE) * VIT_MERGE

            Ha, Wa = Hp * PATCH, Wp * PATCH  # 该图“自有对齐尺寸”

            img_resized = img.resize((Wa, Ha), Image.BICUBIC)  # 保证可被 PATCH 整除
            per_img.append((img_resized, Ha, Wa, Hp, Wp))

        # 这批次的统一尺寸：取各自对齐尺寸的 max（避免再二次缩放导致 token 不一致）
        Ht = max(Ha for _, Ha, _, _, _ in per_img)
        Wt = max(Wa for _, _, Wa, _, _ in per_img)

        # 逐张：pad 到 (Ht, Wt)，转 tensor，并做与处理器一致的归一化
        # 从 AutoProcessor 里拿 mean/std，避免手抄
        image_mean = getattr(getattr(proc, "image_processor", proc), "image_mean", [0.5, 0.5, 0.5])
        image_std  = getattr(getattr(proc, "image_processor", proc), "image_std",  [0.5, 0.5, 0.5])
        mean = torch.tensor(image_mean).view(3,1,1)
        std  = torch.tensor(image_std).view(3,1,1)

        pixel_list = []
        for img_resized, Ha, Wa, _, _ in per_img:
            if (Ha, Wa) != (Ht, Wt):
                # 左上对齐的 letterbox（不改变已对齐好的 patch 网格）
                canvas = Image.new("RGB", (Wt, Ht), (0,0,0))
                canvas.paste(img_resized, (0,0))
                img_final = canvas
            else:
                img_final = img_resized

            px = to_tensor(img_final)                # [3,Ht,Wt], 0..1
            px = (px - mean) / std                   # 标准化
            pixel_list.append(px)

        # 堆叠为 [B, C, T=1, Ht, Wt]
        pixel_values = torch.stack(pixel_list, dim=0).unsqueeze(2)

        # ---- 关键：用“实际的 H/W”计算 grid_thw，确保 seq_len == T*grid_h*grid_w ----
        B, C, T, H, W = pixel_values.shape
        assert T == 1, "当前示例是静态图像，T 应为 1；如为视频请按帧数设置。"

        # 断言 H/W 一定是 PATCH 的整数倍（否则上面流程有误）
        assert H % PATCH == 0 and W % PATCH == 0, f"H/W 需可被 PATCH={PATCH} 整除，got {(H,W)}"
        Hp_tar, Wp_tar = H // PATCH, W // PATCH

        # grid_h/grid_w = 2 * (H/patch), 2 * (W/patch)  —— 这是 Qwen2.5 的约定
        grid_h = SMS * Hp_tar
        grid_w = SMS * Wp_tar

        # 为“整批统一尺寸”，每个样本的 grid_thw 完全相同
        grid_row = torch.tensor([1, grid_h, grid_w], dtype=torch.long)
        grid_thw = grid_row.unsqueeze(0).repeat(B, 1)

        # 最后打包
        vision_inputs = {"pixel_values": pixel_values, "grid_thw": grid_thw}
        return {
            "input_ids": input_ids,
            "labels": labels,
            "vision_inputs": vision_inputs,
        }




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