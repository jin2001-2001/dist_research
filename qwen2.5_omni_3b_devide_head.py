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

    def _ensure_3d(x):
        if x is None:
            return None
        return x if x.dim() == 3 else x.unsqueeze(0)  # [T,D] -> [1,T,D]
    seqs = [_ensure_3d(x) for x in [text_embeds, audio_seq, vision_seq] if x is not None]
    return torch.cat(seqs, dim=1) if len(seqs) > 1 else seqs[0]



class AudioStage(nn.Module):
    def __init__(self, audio_enc):
        super().__init__()
        self.audio_enc = audio_enc

    def forward(self, audio_inputs):
        """
        返回: audio_embeds
        要求: collate 后的 audio_inputs（dict 或 tensor）其拼接顺序要与 input_ids 中 audio_token 的扫描顺序一致。
        """
        if audio_inputs is None:
            return None

        if isinstance(audio_inputs, dict):
            audio_values = (audio_inputs.get("input_values")
                            or audio_inputs.get("audio_values")
                            or audio_inputs.get("input_features"))
        else:
            audio_values = audio_inputs

        if audio_values is None:
            return None

        if hasattr(self.audio_enc, "get_dtype"):
            audio_values = audio_values.type(self.audio_enc.get_dtype())
        audio_values = audio_values.to(next(self.audio_enc.parameters()).device if hasattr(self.audio_enc, "parameters") else audio_values.device)

        try:
            audio_embeds = self.audio_enc(audio_values)
        except TypeError:
            # 适配部分 encoder 的 (x, attention_mask=None) 签名
            audio_embeds = self.audio_enc(audio_values, None)

        return audio_embeds.contiguous() if audio_embeds is not None else None


class VisionStage(nn.Module):
    def __init__(self, vision_enc):
        super().__init__()
        self.vision_enc = vision_enc

    @staticmethod
    def _cat_pixel_values(vision_inputs):
        if vision_inputs is None or not isinstance(vision_inputs, dict):
            return None, None
        if "pixel_values_list" in vision_inputs:
            pv_list = vision_inputs["pixel_values_list"]
            if pv_list is None or len(pv_list) == 0:
                return None, vision_inputs.get("grid_thw", None)
            pv = torch.cat(pv_list, dim=0)
            return pv, vision_inputs.get("grid_thw", None)
        return vision_inputs.get("pixel_values", None), vision_inputs.get("grid_thw", None)

    def forward(self, vision_inputs):
        """
        返回: image_embeds, grid_thw
        要求: collate 后的 pixel_values_list 拼接顺序与 input_ids 中 image_token 的扫描顺序一致。
        """
        if vision_inputs is None:
            return None, None

        pixel_values, grid_thw = self._cat_pixel_values(vision_inputs)
        if pixel_values is None:
            return None, grid_thw

        if hasattr(self.vision_enc, "get_dtype"):
            pixel_values = pixel_values.type(self.vision_enc.get_dtype())
        pixel_values = pixel_values.to(next(self.vision_enc.parameters()).device if hasattr(self.vision_enc, "parameters") else pixel_values.device)

        image_embeds = self.vision_enc(pixel_values, grid_thw=grid_thw)
        return (image_embeds.contiguous() if image_embeds is not None else None), grid_thw


class TextStage(nn.Module):
    def __init__(self, text_model):
        super().__init__()
        self.text_model = text_model
        self.embed_tokens = text_model.embed_tokens

    def forward(self, input_ids, attention_mask=None):
        """
        返回:
            hidden: [B, T, H]
            attn_4d: [B, 1, T, T]（含 pad 与因果遮罩）
            position_ids: 这里返回 None，交由 Stage1 统一计算（含多模态感知索引）
        """
        device = input_ids.device
        B, T = input_ids.shape

        # 文本嵌入
        hidden = self.embed_tokens(input_ids)

        # 4D attention mask（因果+pad）
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
        attn_4d = build_causal(T, device=device).expand(B, -1, -1, -1).clone()
        pad = (attention_mask == 0).view(B, 1, 1, T)
        attn_4d = attn_4d.masked_fill(pad, float("-inf")).contiguous()

        # 不在这里计算 position_ids；交给 Stage1
        position_ids = None
        return hidden.contiguous(), attn_4d.contiguous(), position_ids


# -----------------------------
# Stage1: 先做 pack（按特殊 token 替换 embedding），再计算/补全 position_ids，最后跑 L1 层 Transformer
# -----------------------------
class Stage1(nn.Module):
    def __init__(self, text_model, L1):
        super().__init__()
        self.layers = nn.ModuleList(text_model.layers[:L1])
        self.rotary_emb = text_model.rotary_emb
        self.get_rope_index = getattr(text_model, "get_rope_index", None)

        cfg = getattr(text_model, "config", None)
        self.image_token_id = getattr(cfg, "image_token_id", None)
        self.audio_token_id = getattr(cfg, "audio_token_id", None)

    @staticmethod
    def _replace_feats_by_token_id(input_ids, inputs_embeds, feats, special_token_id):
        if feats is None or special_token_id is None or input_ids is None:
            return inputs_embeds
        flat_mask = (input_ids == special_token_id).reshape(-1)
        n_tokens = int(flat_mask.sum().item())
        if n_tokens == 0:
            return inputs_embeds
        if feats.size(0) != n_tokens:
            raise RuntimeError(
                f"Feature count mismatch for token_id={special_token_id}: "
                f"tokens={n_tokens} vs feats={feats.size(0)}"
            )
        emb_flat = inputs_embeds.reshape(-1, inputs_embeds.size(-1))
        feats = feats.to(device=emb_flat.device, dtype=emb_flat.dtype)
        emb_flat[flat_mask] = feats
        return emb_flat.view_as(inputs_embeds)

    def forward(self, *args, **kwargs):
        """
        期望输入（来自各 stage 的聚合）：
            基础三元组：
                hidden, attn_mask_4d, position_ids (可能为 None)
            以及 pack/位置计算所需的 kwargs：
                input_ids: [B, T]（用于替换定位 & 重新计算 position_ids）
                attention_mask_2d: [B, T]（若需用 get_rope_index 计算位置）
                image_embeds: [sum_img_tokens, H] 或 None
                audio_embeds: [sum_aud_tokens, H] 或 None
                grid_thw: 视觉网格元信息（VisionStage 产出），用于 get_rope_index
                image_token_id/audio_token_id: 可覆盖 config 默认
        """
        # 兼容三元组
        if len(args) >= 3:
            hidden, attn_mask, position_ids = args[:3]
        else:
            hidden = args[0] if len(args) > 0 else kwargs['hidden']
            attn_mask = args[1] if len(args) > 1 else kwargs['attn_mask']
            position_ids = args[2] if len(args) > 2 else kwargs.get('position_ids', None)

        # pack 所需
        input_ids = kwargs.get('input_ids', None)
        attention_mask_2d = kwargs.get('attention_mask_2d', None)
        grid_thw = kwargs.get('grid_thw', None)

        image_embeds = kwargs.get('image_embeds', None)
        audio_embeds = kwargs.get('audio_embeds', None)
        image_token_id = kwargs.get('image_token_id', self.image_token_id)
        audio_token_id = kwargs.get('audio_token_id', self.audio_token_id)

        # 1) pack：若提供了任一模态特征且有 input_ids，则做按位替换
        if input_ids is not None:
            if image_embeds is not None:
                hidden = self._replace_feats_by_token_id(input_ids, hidden, image_embeds, image_token_id)
            if audio_embeds is not None:
                hidden = self._replace_feats_by_token_id(input_ids, hidden, audio_embeds, audio_token_id)

        # 2) 计算/补全 position_ids：
        #    - 若 TextStage 已给出 position_ids，可直接沿用；
        #    - 否则（为 None），优先使用 get_rope_index(感知 grid_thw) 计算；
        #    - 若模型无 get_rope_index，则回退到等距 1D 位置编码。
        if position_ids is None:
            if self.get_rope_index is not None and input_ids is not None and attention_mask_2d is not None:
                position_ids, _ = self.get_rope_index(
                    input_ids,
                    image_grid_thw=grid_thw,
                    video_grid_thw=None,
                    attention_mask=attention_mask_2d
                )
            else:
                # 回退：三路堆叠（与原实现保持一致的形状）
                B, T, _ = hidden.shape
                base_pos = torch.arange(T, device=hidden.device).unsqueeze(0).repeat(B, 1)
                position_ids = torch.stack([base_pos, base_pos, base_pos], dim=0).contiguous()

        # 3) 原有 Transformer 前向保持不变
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).repeat(3, 1, 1)

        pos_emb = self.rotary_emb(hidden, position_ids)
        for blk in self.layers:
            hidden = blk(
                hidden_states=hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                position_embeddings=pos_emb,
                output_attentions=False,
                use_cache=False
            )[0]

        return hidden.contiguous(), attn_mask.contiguous(), position_ids.contiguous()


class Stage2(nn.Module):
    def __init__(self, text_model, L1, L2):
        super().__init__()
        self.layers = nn.ModuleList(text_model.layers[L1:L2])
        self.rotary_emb = text_model.rotary_emb
        
    def forward(self, *args, **kwargs):
        # Handle flexible arguments
        if len(args) >= 3:
            hidden, attn_mask, position_ids = args[:3]
        else:
            hidden = args[0] if len(args) > 0 else kwargs['hidden']
            attn_mask = args[1] if len(args) > 1 else kwargs['attn_mask']
            position_ids = args[2] if len(args) > 2 else kwargs['position_ids']
        
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).repeat(3, 1, 1)
        
        pos_emb = self.rotary_emb(hidden, position_ids)
        for blk in self.layers:
            hidden = blk(
                hidden_states=hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                position_embeddings=pos_emb,
                output_attentions=False,
                use_cache=False
            )[0]
        return hidden.contiguous(), attn_mask.contiguous(), position_ids.contiguous()


class Stage3(nn.Module):
    def __init__(self, full_thinker, text_model, L2):
        super().__init__()
        self.layers = nn.ModuleList(text_model.layers[L2:])
        self.norm = text_model.norm
        self.lm_head = full_thinker.lm_head
        self.rotary_emb = text_model.rotary_emb
        
    def forward(self, *args, **kwargs):
        # Handle flexible arguments
        if len(args) >= 3:
            hidden, attn_mask, position_ids = args[:3]
        else:
            hidden = args[0] if len(args) > 0 else kwargs['hidden']
            attn_mask = args[1] if len(args) > 1 else kwargs['attn_mask']
            position_ids = args[2] if len(args) > 2 else kwargs['position_ids']
        
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).repeat(3, 1, 1)
        
        pos_emb = self.rotary_emb(hidden, position_ids)
        for blk in self.layers:
            hidden = blk(
                hidden_states=hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                position_embeddings=pos_emb,
                output_attentions=False,
                use_cache=False
            )[0]
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits
