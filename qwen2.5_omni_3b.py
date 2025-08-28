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
    import numpy as np

    def _round_to_multiple(x: int, m: int) -> int:
        return ((x + m - 1) // m) * m

    def _pad_to_multiple_of_56(img: Image.Image) -> Image.Image:
        # 不形变，只在右/下补 0；56 = 14(patch) * 4(merge unit)
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        new_w = _round_to_multiple(w, 56)
        new_h = _round_to_multiple(h, 56)
        if (new_w, new_h) == (w, h):
            return img
        return ImageOps.expand(img, border=(0, 0, new_w - w, new_h - h), fill=0)

    def _pil_to_tensor_chw(img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.uint8)
        if arr.ndim == 2:                      # 灰度图→3通道
            arr = np.stack([arr, arr, arr], axis=-1)
        t = torch.from_numpy(arr).to(torch.float32) / 255.0  # [H,W,C] in [0,1]
        return t.permute(2, 0, 1).contiguous()               # [C,H,W]

    def _normalize_3c(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if mean.ndim == 1: mean = mean.view(-1, 1, 1)
        if std.ndim == 1:  std  = std.view(-1, 1, 1)
        return (t - mean) / std


    def collate_fn(batch):
        """
        - 文本：用 processor 的 tokenizer 做 padding/truncation，labels=输入副本
        - 图像：自行处理，不走 processor 的图像管线
            1) EXIF 矫正 + pad 到 56 的倍数（56=14*4，满足视觉塔 merge=4 的约束）
            2) PIL -> [C,H,W] float32 in [0,1]，用 processor 的 mean/std 归一化（无则退回 0.5）
            3) 批内统一目标大小 H_target×W_target（各自对齐到 56 倍数后取最大）
            4) 堆叠为 [B,C,1,H,W]（T=1）
            5) 自行构建 grid_thw=[T,H/14,W/14]，保证 H'/W' 均为 4 的倍数
        返回键：
            - input_ids: [B, L]
            - labels:    [B, L]
            - vision_inputs: {"pixel_values": [B,C,1,H,W], "grid_thw": [B,3]}
        """
        import numpy as np
        from PIL import ImageOps

        # --------- 工具函数 ---------
        def _round_to_multiple(x: int, m: int) -> int:
            return ((x + m - 1) // m) * m

        def _pad_to_multiple_of_56(img):
            # 不形变，只在右/下补 0；56 = 14(patch) * 4(merge unit)
            img = ImageOps.exif_transpose(img)
            w, h = img.size
            new_w = _round_to_multiple(w, 56)
            new_h = _round_to_multiple(h, 56)
            if (new_w, new_h) == (w, h):
                return img
            return ImageOps.expand(img, border=(0, 0, new_w - w, new_h - h), fill=0)

        def _pil_to_tensor_chw(img):
            arr = np.array(img, dtype=np.uint8)          # [H,W,C] or [H,W]
            if arr.ndim == 2:                            # 灰度 -> 3 通道
                arr = np.stack([arr, arr, arr], axis=-1)
            t = torch.from_numpy(arr).to(torch.float32) / 255.0
            return t.permute(2, 0, 1).contiguous()       # [C,H,W]

        def _normalize_3c(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
            if mean.ndim == 1: mean = mean.view(-1, 1, 1)
            if std.ndim == 1:  std  = std.view(-1, 1, 1)
            return (t - mean) / std

        # --------- 文本处理（processor 只用于文本） ---------
        texts = [ex.get("text", "") for ex in batch]
        pack_txt = proc(
            text=texts,
            return_tensors="pt",
            text_kwargs={
                "padding": True,
                "truncation": True,
                "return_attention_mask": True,
            },
        )
        input_ids = pack_txt["input_ids"]          # [B, L]
        labels    = input_ids.clone()              # LM 目标（左移在 loss 内部做）

        # --------- 图像处理（完全自管） ---------
        images = [_pad_to_multiple_of_56(ex["image"]) for ex in batch]

        # 归一化参数：优先用 processor 的 mean/std；无则退回 0.5
        try:
            mean = torch.tensor(proc.image_processor.image_mean, dtype=torch.float32)
            std  = torch.tensor(proc.image_processor.image_std,  dtype=torch.float32)
        except Exception:
            mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            std  = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

        # 1) 逐图：PIL->Tensor、归一化，并记录对齐到 56 倍数的尺寸
        per_img = []  # (tensor_CHW, H, W, H_aligned, W_aligned)
        H_aligned_list, W_aligned_list = [], []
        for img in images:
            t = _pil_to_tensor_chw(img)         # [C,H,W]
            t = _normalize_3c(t, mean, std)
            _, H, W = t.shape
            H_a = _round_to_multiple(H, 56)
            W_a = _round_to_multiple(W, 56)
            per_img.append((t, H, W, H_a, W_a))
            H_aligned_list.append(H_a)
            W_aligned_list.append(W_a)

        # 2) 批内统一目标尺寸（仍为 56 的倍数）
        H_target = max(H_aligned_list)
        W_target = max(W_aligned_list)

        # 3) pad 到统一尺寸并堆叠 -> [B,C,1,H,W]
        tensors = []
        for t, H, W, H_a, W_a in per_img:
            # 先 pad 到各自对齐尺寸
            if H_a != H or W_a != W:
                t = F.pad(t, (0, W_a - W, 0, H_a - H), value=0.0)  # (W_left, W_right, H_top, H_bottom)
                H, W = H_a, W_a
            # 再 pad 到批内共同最大尺寸
            if H_target != H or W_target != W:
                t = F.pad(t, (0, W_target - W, 0, H_target - H), value=0.0)
            tensors.append(t)

        pixel_values = torch.stack(tensors, dim=0).unsqueeze(2)  # [B,C,1,H_target,W_target]

        # 4) 构建 grid_thw，确保 H'/W' 为 4 的倍数（merge_unit=4）
        patch = 14
        Hp = H_target // patch
        Wp = W_target // patch
        if (Hp % 4 != 0) or (Wp % 4 != 0):
            # 理论不会发生；若发生，说明上面 pad 非 56 倍数
            raise ValueError(f"H'/W' must be multiples of 4; got Hp={Hp}, Wp={Wp}, H_target={H_target}, W_target={W_target}")
        B = pixel_values.size(0)
        grid_thw = torch.tensor([[1, Hp, Wp]] * B, dtype=torch.long)  # T=1

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