import argparse, os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torch.distributed as dist

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import time, math

from stage_with_mutiple_ranks import PipelineStage_with_mutiple_ranks, PipelineStage_Multimodality
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
        import time
        _t0 = time.perf_counter()
        try:
            
            rid = dist.get_rank() if dist.is_initialized() else -1
        except Exception:
            rid = -1

        if audio_inputs is None:
            device = next(self.audio_enc.parameters()).device if hasattr(self.audio_enc, "parameters") else torch.device("cpu")
            out = torch.zeros(1, 1, 768, device=device, dtype=torch.float32)
            try:
                print(f"[rank{rid}] AudioStage.forward: audio_inputs=None -> return zeros {tuple(out.shape)}")
            except Exception:
                pass
            return out

        if isinstance(audio_inputs, dict):
            # 避免对张量做布尔短路判断，逐键安全取值
            audio_values = None
            for k in ("input_features", "input_values", "audio_values"):
                v = audio_inputs.get(k, None)
                if v is not None:
                    audio_values = v
                    break
            feature_attention_mask = audio_inputs.get("feature_attention_mask", None)
        else:
            audio_values = audio_inputs
            feature_attention_mask = None

        if audio_values is None:
            device = next(self.audio_enc.parameters()).device if hasattr(self.audio_enc, "parameters") else torch.device("cpu")
            out = torch.zeros(1, 1, 768, device=device, dtype=torch.float32)
            try:
                print(f"[rank{rid}] AudioStage.forward: audio_values=None -> return zeros {tuple(out.shape)}")
            except Exception:
                pass
            return out

        # 确保audio_values是float类型
        if audio_values.dtype != torch.float32:
            audio_values = audio_values.float()

        if hasattr(self.audio_enc, "get_dtype"):
            audio_values = audio_values.type(self.audio_enc.get_dtype())
        audio_values = audio_values.to(next(self.audio_enc.parameters()).device if hasattr(self.audio_enc, "parameters") else audio_values.device)

        B, D1, D2 = audio_values.shape[:3]
        _pre_shape = tuple(audio_values.shape)

        # 维度自适配：推断 [B, n_mels, n_frames] 或 [B, n_frames, n_mels]
        typical_mels = {64, 80, 96, 128}
        n_mels = None
        n_frames = None
        input_feats = None

        # 目标：让“时间轴”在 dim=1，便于与 feature_lens 对齐
        if D1 in typical_mels:
            # 当前是 [B, mels, frames] -> 转成 [B, frames, mels]
            n_mels, n_frames = D1, D2
            input_feats = audio_values.transpose(1, 2)
        elif D2 in typical_mels:
            # 已是 [B, frames, mels]
            n_mels, n_frames = D2, D1
            input_feats = audio_values
        else:
            # 启发式：把较大的维度当作帧数，较小的当作梅尔通道
            if D1 >= D2:
                # [B, frames, mels]
                n_mels, n_frames = D2, D1
                input_feats = audio_values
            else:
                # [B, mels, frames] -> 转到 [B, frames, mels]
                n_mels, n_frames = D1, D2
                input_feats = audio_values.transpose(1, 2)

        _post_shape = tuple(input_feats.shape)

        # 扁平 batch：拼到 [1, sumT, 128] 并构造 feature_lens=[T1..TB]（单位：帧）
        per_list = []
        T_list = []
        for i in range(B):
            t_i = input_feats[i:i+1]              # [1, T_i, 128]
            per_list.append(t_i)
            T_list.append(int(t_i.size(1)))
        feats_cat = torch.cat(per_list, dim=1)     # [1, sumT, 128]
        feature_lens = torch.tensor(T_list, dtype=torch.long, device=feats_cat.device)  # [B]

        # 由模型工具函数推导卷积后的长度
        try:
            aftercnn_lens, _ = self.audio_enc._get_feat_extract_output_lengths(feature_lens)
        except Exception:
            aftercnn_lens = feature_lens.clone()

        # dtype 对齐到卷积权重（避免 conv 类型不匹配）
        try:
            want_dtype = self.audio_enc.conv1.weight.dtype  # type: ignore[attr-defined]
        except Exception:
            want_dtype = feats_cat.dtype
        feats_cat = feats_cat.to(want_dtype)

        try:
            print(
                f"[rank{rid}] AudioStage.forward: pre={_pre_shape} mid={_post_shape} post={tuple(feats_cat.shape)} "
                f"feature_lens={feature_lens.tolist()} aftercnn_lens={aftercnn_lens.tolist()} dtype={feats_cat.dtype}"
            )
        except Exception:
            pass

        _path = "main"
        try:
            res = self.audio_enc(
                input_features=feats_cat,      # [1, sumT, 128]，时间在 dim=1
                feature_lens=feature_lens,     # [B]
                aftercnn_lens=aftercnn_lens    # [B]
            )
        except Exception as e1:
            try:
                _path = "fallback_no_aftercnn_lens"
                res = self.audio_enc(
                    input_features=input_feats,
                    feature_lens=feature_lens
                )
            except Exception as e2:
                _path = "zeros_fallback"
                device = input_feats.device
                # 兜底：按启发式估算下采样后序列长度
                out_len = max(1, n_frames // 4)
                out = torch.zeros(B, out_len, 768, device=device, dtype=torch.float32)
                try:
                    _ms = (time.perf_counter() - _t0) * 1000.0
                    print(
                        f"[rank{rid}] AudioStage.forward: path={_path} -> zeros {tuple(out.shape)} took={_ms:.2f}ms pre={_pre_shape} post={_post_shape}; "
                        f"feature_lens={feature_lens.tolist()} aftercnn_lens={aftercnn_lens.tolist()} "
                        f"errors=({type(e1).__name__}: {e1}; {type(e2).__name__}: {e2})"
                    )
                except Exception:
                    pass
                return out

        audio_embeds = None
        if isinstance(res, torch.Tensor):
            audio_embeds = res
        elif hasattr(res, "last_hidden_state") and isinstance(res.last_hidden_state, torch.Tensor):
            audio_embeds = res.last_hidden_state
        elif isinstance(res, (list, tuple)):
            audio_embeds = next((x for x in res if isinstance(x, torch.Tensor)), None)
        elif isinstance(res, dict):
            audio_embeds = res.get("last_hidden_state", None)
            if not isinstance(audio_embeds, torch.Tensor):
                for v in res.values():
                    if isinstance(v, torch.Tensor):
                        audio_embeds = v
                        break

        if isinstance(audio_embeds, torch.Tensor):
            out = audio_embeds.contiguous()
            try:
                _ms = (time.perf_counter() - _t0) * 1000.0
                abs_mean = float(out.detach().abs().mean().item())
                enc_norm = None
                try:
                    # 简单统计编码器参数范数
                    ps = list(self.audio_enc.parameters())
                    if ps:
                        enc_norm = float(sum(p.detach().norm().item() for p in ps[:3]))  # 采样前3个层参数
                except Exception:
                    pass
                print(f"[rank{rid}] AudioStage.forward: path={_path} in={_pre_shape}->{_post_shape} out={tuple(out.shape)} took={_ms:.2f}ms abs_mean={abs_mean:.3e} enc_norm={enc_norm}")
            except Exception:
                pass
            return out
        else:
            device = audio_values.device
            batch_size, seq_len = audio_values.shape[:2]
            out = torch.zeros(batch_size, seq_len // 4, 768, device=device, dtype=torch.float32)
            try:
                _ms = (time.perf_counter() - _t0) * 1000.0
                print(f"[rank{rid}] AudioStage.forward: no tensor in result -> zeros {tuple(out.shape)} took={_ms:.2f}ms pre={_pre_shape} post={_post_shape}")
            except Exception:
                pass
            return out


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
    

        # 添加padding逻辑确保输出shape一致
        if image_embeds is not None:
            # 设定一个固定的最大长度（根据错误信息，期望长度是252）
            MAX_VISION_TOKENS = 252  # 根据pipeline期望的shape设置

            current_len = image_embeds.shape[0]  # 当前序列长度
            if current_len < MAX_VISION_TOKENS:
                # Padding到固定长度
                pad_len = MAX_VISION_TOKENS - current_len
                embed_dim = image_embeds.shape[1]
                padding = torch.zeros(pad_len, embed_dim, device=image_embeds.device, dtype=image_embeds.dtype)
                image_embeds = torch.cat([image_embeds, padding], dim=0)
            elif current_len > MAX_VISION_TOKENS:
                # 截断到固定长度
                image_embeds = image_embeds[:MAX_VISION_TOKENS]

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
        import time
        _t0 = time.perf_counter()
        device = input_ids.device
        B, T = input_ids.shape

        # 文本嵌入
        _t_emb0 = time.perf_counter()
        hidden = self.embed_tokens(input_ids)
        _t_emb1 = time.perf_counter()

        # 4D attention mask（因果+pad）
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
        _t_attn0 = time.perf_counter()
        attn_4d = build_causal(T, device=device).expand(B, -1, -1, -1).clone()
        pad = (attention_mask == 0).view(B, 1, 1, T)
        attn_4d = attn_4d.masked_fill(pad, float("-inf")).contiguous()
        _t_attn1 = time.perf_counter()

        # 在头部阶段直接给出基础 position_ids（三路堆叠），避免下游元信息校验因 None 失败
        _t_pos0 = time.perf_counter()
        base_pos = torch.arange(T, device=device).unsqueeze(0).repeat(B, 1)
        position_ids = torch.stack([base_pos, base_pos, base_pos], dim=0).contiguous()
        _t_pos1 = time.perf_counter()
        # 额外返回 input_ids 和 attention_mask，供 packing 阶段做多模态替换与位置计算
        out = (hidden.contiguous(), attn_4d.contiguous(), position_ids, input_ids.contiguous(), attention_mask.contiguous())
        try:
            
            rid = dist.get_rank() if dist.is_initialized() else -1
            _ms = (time.perf_counter() - _t0) * 1000.0
            # 简要统计，避免过重代价
            attn_sum = int(attention_mask.sum().item()) if isinstance(attention_mask, torch.Tensor) else -1
            uid_min = int(input_ids.min().item()) if isinstance(input_ids, torch.Tensor) else -1
            uid_max = int(input_ids.max().item()) if isinstance(input_ids, torch.Tensor) else -1
            emb_ms = (_t_emb1 - _t_emb0) * 1000.0
            attn_ms = (_t_attn1 - _t_attn0) * 1000.0
            pos_ms = (_t_pos1 - _t_pos0) * 1000.0
            ham = float(hidden.detach().abs().mean().item()) if isinstance(hidden, torch.Tensor) else float('nan')
            try:
                w = self.embed_tokens.weight
                enorm = float(w.detach().norm().item())
            except Exception:
                enorm = float('nan')
            print(f"[rank{rid}] TextStage.forward: B={B} T={T} attn_sum={attn_sum} input_ids[min,max]=({uid_min},{uid_max}) took={_ms:.2f}ms; parts: emb={emb_ms:.2f}ms, attn={attn_ms:.2f}ms, pos={pos_ms:.2f}ms; hidden_abs_mean={ham:.3e}, embed_norm={enorm:.3e}")
        except Exception:
            pass
        return out


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

        # 推断与适配工具（不引入新可训练参数，保持最小侵入）

    @staticmethod
    def _adapt_feats_to_hidden(feats: torch.Tensor, hidden_dim: int) -> torch.Tensor:
        """将任意形状 [..., D] 的特征适配为 [N, hidden_dim]：
        - 若 feats.dim() == 3，拉平成 [N, D]
        - 若 D < hidden_dim：在最后一维右侧 zero-pad 保持梯度传递（使用 cat）
        - 若 D > hidden_dim：直接截断到左侧 hidden_dim（保持梯度传递）
        不引入可训练参数，最大限度减少侵入。
        """
        if feats is None:
            return feats
        if feats.dim() == 3:
            feats = feats.reshape(-1, feats.size(-1))
        elif feats.dim() == 2:
            pass
        else:
            feats = feats.reshape(-1, feats.size(-1))

        D = feats.size(-1)
        if D == hidden_dim:
            return feats
        if D < hidden_dim:
            pad = torch.zeros(feats.size(0), hidden_dim - D, device=feats.device, dtype=feats.dtype)
            return torch.cat([feats, pad], dim=-1)
        else:  # D > hidden_dim
            return feats[:, :hidden_dim]

    @staticmethod
    def _infer_token_id_by_count(input_ids: torch.Tensor, expected_count: int, prefer_large_id: bool = True) -> int | None:
        """在 input_ids 中根据出现次数推断特殊 token 的 id。
        - 统计每个 id 的出现次数，寻找与 expected_count 恰好相等的 id；
        - 若有多个候选，返回数值更大的 id（通常新增特殊 token id 较大）；
        - 若找不到，返回 None。
        """
        if input_ids is None or expected_count <= 0:
            return None
        ids = input_ids.reshape(-1)
        uniq, cnt = torch.unique(ids, return_counts=True)
        mask = (cnt == expected_count)
        cand = uniq[mask]
        if cand.numel() == 0:
            return None
        if cand.numel() == 1:
            return int(cand.item())
        # 多候选：选取较大的 id（启发式）
        return int(torch.max(cand).item()) if prefer_large_id else int(torch.min(cand).item())

    @staticmethod
    def _infer_token_id_by_longest_run(input_ids: torch.Tensor) -> tuple[int | None, int]:
        """在 input_ids 的单样本序列中寻找最长连续相同 id 的片段，返回 (id, run_len)。
        B>1 时取第一个样本。若失败，返回 (None, 0)。
        """
        if input_ids is None or not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 2 or input_ids.numel() == 0:
            return None, 0
        ids = input_ids[0].detach()
        best_id = None
        best_len = 0
        cur_id = None
        cur_len = 0
        for v in ids:
            v = int(v.item())
            if cur_id is None or v != cur_id:
                if cur_len > best_len:
                    best_id, best_len = cur_id, cur_len
                cur_id, cur_len = v, 1
            else:
                cur_len += 1
        if cur_len > best_len:
            best_id, best_len = cur_id, cur_len
        return best_id, best_len

    @staticmethod
    def _align_feats_length(feats: torch.Tensor, target_len: int) -> torch.Tensor:
        """将 feats 的第一维（N）对齐到 target_len：
        - N == target_len: 直接返回
        - N > target_len: 截断到前 target_len
        - N < target_len: 重复最后一个向量或零填充到 target_len（使用重复以保持梯度可传）
        """
        if feats is None:
            return feats
        N = feats.size(0)
        if N == target_len:
            return feats
        if N > target_len:
            return feats[:target_len]
        # N < target_len
        if N == 0:
            # 退化：构造全零（保持 dtype/device）
            return torch.zeros(target_len, feats.size(1), device=feats.device, dtype=feats.dtype)
        # 重复最后一个向量填充到目标长度
        pad_count = target_len - N
        last_vec = feats[-1:].expand(pad_count, -1)
        return torch.cat([feats, last_vec], dim=0)

    @staticmethod
    def _replace_feats_by_token_id(input_ids, inputs_embeds, feats, special_token_id):
        if feats is None or special_token_id is None or input_ids is None:
            try:
                
                rid = dist.get_rank() if dist.is_initialized() else -1
                print(f"[rank{rid}] Stage1._replace_feats_by_token_id: skip replacement due to feats/input_ids/token_id None. token_id={special_token_id} feats={(None if feats is None else tuple(feats.shape))} input_ids_is_none={input_ids is None}")
            except Exception:
                pass
            return inputs_embeds
        flat_mask = (input_ids == special_token_id).reshape(-1)
        n_tokens = int(flat_mask.sum().item())
        if n_tokens == 0:
            try:
                
                rid = dist.get_rank() if dist.is_initialized() else -1
                print(f"[rank{rid}] Stage1._replace_feats_by_token_id: no tokens for token_id={special_token_id}; feats_shape={(tuple(feats.shape) if hasattr(feats,'shape') else type(feats).__name__)}")
            except Exception:
                pass
            return inputs_embeds
        if feats.size(0) != n_tokens:
            raise RuntimeError(
                f"Feature count mismatch for token_id={special_token_id}: "
                f"tokens={n_tokens} vs feats={feats.size(0)}"
            )
        # 使用 out-of-place scatter 构造替换后的新张量，避免对 view/leaf 的就地写入
        emb_flat = inputs_embeds.reshape(-1, inputs_embeds.size(-1))
        feats = feats.to(device=emb_flat.device, dtype=emb_flat.dtype)
        idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(1).to(dtype=torch.long)
        idx2 = idx.unsqueeze(1).expand(-1, emb_flat.size(1))
        out_flat = emb_flat.scatter(0, idx2, feats)
        try:
            
            rid = dist.get_rank() if dist.is_initialized() else -1
            print(f"[rank{rid}] Stage1._replace_feats_by_token_id: replaced token_id={special_token_id} count={n_tokens} feats_shape={tuple(feats.shape)} emb_dim={emb_flat.size(-1)}")
        except Exception:
            pass
        return out_flat.view_as(inputs_embeds)

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

        # Debug: summarize inputs to Stage1
        try:
            
            rid = dist.get_rank() if dist.is_initialized() else -1
            def _s(x):
                return tuple(x.shape) if isinstance(x, torch.Tensor) else None
            img_cnt = None
            aud_cnt = None
            if isinstance(input_ids, torch.Tensor):
                try:
                    img_cnt = int((input_ids == (image_token_id if image_token_id is not None else -999999)).sum().item())
                    aud_cnt = int((input_ids == (audio_token_id if audio_token_id is not None else -999999)).sum().item())
                except Exception:
                    pass
            print(
                f"[rank{rid}] Stage1.forward: input_ids={_s(input_ids)} attention_mask_2d={_s(attention_mask_2d)} grid_thw={_s(grid_thw)} "
                f"image_embeds={_s(image_embeds)} audio_embeds={_s(audio_embeds)} image_token_id={image_token_id} audio_token_id={audio_token_id} "
                f"counts(img,aud)=({img_cnt},{aud_cnt})"
            )
        except Exception:
            pass

        # 1) pack：若提供了任一模态特征且有 input_ids，则做按位替换
        if input_ids is not None:
            # 目标 hidden 维度
            H = hidden.size(-1)

            # 视觉替换：维度自适配 + id 推断（计数匹配失败时用最长连续片段回退，并对齐长度）
            if image_embeds is not None:
                img_feats = self._adapt_feats_to_hidden(image_embeds, H)
                # 若未提供 token_id，依据计数推断
                if image_token_id is None:
                    image_token_id = self._infer_token_id_by_count(input_ids, expected_count=img_feats.size(0))
                    try:
                        
                        rid = dist.get_rank() if dist.is_initialized() else -1
                        print(f"[rank{rid}] Stage1.forward: inferred image_token_id={image_token_id} from count={img_feats.size(0)}")
                    except Exception:
                        pass
                if image_token_id is None:
                    # 回退：最长连续片段
                    run_id, run_len = self._infer_token_id_by_longest_run(input_ids)
                    if run_id is not None and run_len > 0:
                        image_token_id = run_id
                        try:
                            
                            rid = dist.get_rank() if dist.is_initialized() else -1
                            print(f"[rank{rid}] Stage1.forward: fallback longest-run image_token_id={image_token_id}, run_len={run_len}")
                        except Exception:
                            pass
                        # 根据 run_len 对齐特征长度
                        img_feats = self._align_feats_length(img_feats, run_len)
                if image_token_id is not None:
                    try:
                        # 计算 mask 长度，确保与特征长度一致
                        flat_mask = (input_ids == image_token_id).reshape(-1)
                        n_tokens = int(flat_mask.sum().item())
                        if img_feats.size(0) != n_tokens:
                            img_feats = self._align_feats_length(img_feats, n_tokens)
                        hidden = self._replace_feats_by_token_id(input_ids, hidden, img_feats, image_token_id)
                    except Exception as e:
                        try:
                            
                            rid = dist.get_rank() if dist.is_initialized() else -1
                            print(f"[rank{rid}] Stage1.forward: image replace failed with {type(e).__name__}: {e}")
                        except Exception:
                            pass
                else:
                    try:
                        
                        rid = dist.get_rank() if dist.is_initialized() else -1
                        print(f"[rank{rid}] Stage1.forward: could not infer image_token_id; skip image replace")
                    except Exception:
                        pass
            else:
                try:
                    
                    rid = dist.get_rank() if dist.is_initialized() else -1
                    print(f"[rank{rid}] Stage1.forward: skip image replace; image_embeds is None")
                except Exception:
                    pass

            # 音频替换：拉平为 [N, D] + 维度自适配 + id 推断（计数失败回退最长连续片段，并对齐长度）
            if audio_embeds is not None:
                aud_feats = self._adapt_feats_to_hidden(audio_embeds, H)
                if audio_token_id is None:
                    audio_token_id = self._infer_token_id_by_count(input_ids, expected_count=aud_feats.size(0))
                    try:
                        
                        rid = dist.get_rank() if dist.is_initialized() else -1
                        print(f"[rank{rid}] Stage1.forward: inferred audio_token_id={audio_token_id} from count={aud_feats.size(0)}")
                    except Exception:
                        pass
                if audio_token_id is None:
                    run_id, run_len = self._infer_token_id_by_longest_run(input_ids)
                    if run_id is not None and run_len > 0:
                        audio_token_id = run_id
                        try:
                            
                            rid = dist.get_rank() if dist.is_initialized() else -1
                            print(f"[rank{rid}] Stage1.forward: fallback longest-run audio_token_id={audio_token_id}, run_len={run_len}")
                        except Exception:
                            pass
                        aud_feats = self._align_feats_length(aud_feats, run_len)
                if audio_token_id is not None:
                    try:
                        flat_mask = (input_ids == audio_token_id).reshape(-1)
                        n_tokens = int(flat_mask.sum().item())
                        if aud_feats.size(0) != n_tokens:
                            aud_feats = self._align_feats_length(aud_feats, n_tokens)
                        hidden = self._replace_feats_by_token_id(input_ids, hidden, aud_feats, audio_token_id)
                    except Exception as e:
                        try:
                            
                            rid = dist.get_rank() if dist.is_initialized() else -1
                            print(f"[rank{rid}] Stage1.forward: audio replace failed with {type(e).__name__}: {e}")
                        except Exception:
                            pass
                else:
                    try:
                        
                        rid = dist.get_rank() if dist.is_initialized() else -1
                        print(f"[rank{rid}] Stage1.forward: could not infer audio_token_id; skip audio replace")
                    except Exception:
                        pass
            else:
                try:
                    
                    rid = dist.get_rank() if dist.is_initialized() else -1
                    print(f"[rank{rid}] Stage1.forward: skip audio replace; audio_embeds is None")
                except Exception:
                    pass
        else:
            try:
                
                rid = dist.get_rank() if dist.is_initialized() else -1
                print(f"[rank{rid}] Stage1.forward: input_ids is None; cannot perform multimodal replacement")
            except Exception:
                pass

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

        try:
            
            rid = dist.get_rank() if dist.is_initialized() else -1
            print(f"[rank{rid}] Stage1.forward: final position_ids shape={(tuple(position_ids.shape) if isinstance(position_ids, torch.Tensor) else None)}")
        except Exception:
            pass

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

def create_pipeline_actions():

    # [stage],[rank],[id],[action type],[microbatch],[dest_rank],[upstream],[dependency], [split_parts], [chunk_deps], [multimodality] id不用管，不影响运行
    # Rank 0 (Stage 0)
    rank0_actions = [
        _Action(0, 0, 0, _ComputationType.FORWARD, (0), None, None, None, None, None, ["audio"]),
        _Action(0, 0, 1, _ComputationType.SEND_F, (0,), 3, None, None, None, None, ["audio"]),
        _Action(0, 0, 2, _ComputationType.FORWARD, (1), None, None, None, None, None, ["audio"]),
        _Action(0, 0, 3, _ComputationType.SEND_F, (1,), 3, None, None, None, None, ["audio"]),
        _Action(0, 0, 4, _ComputationType.FORWARD, (2), None, None, None, None, None, ["audio"]),
        _Action(0, 0, 5, _ComputationType.SEND_F, (2,), 3, None, None, None, None, ["audio"]),
        _Action(0, 0, 6, _ComputationType.FORWARD, (3), None, None, None, None, None, ["audio"]),
        _Action(0, 0, 7, _ComputationType.SEND_F, (3,), 3, None, None, None, None, ["audio"]),

        _Action(0, 0, 8, _ComputationType.RECV_B, (0,), 3, None, None, None, None, ["audio"]),
        _Action(0, 0, 9, _ComputationType.FULL_BACKWARD, (0), None, None, None, None, None, ["audio"]),
        _Action(0, 0, 10, _ComputationType.RECV_B, (1,), 3, None, None, None, None, ["audio"]),
        _Action(0, 0, 11, _ComputationType.FULL_BACKWARD, (1), None, None, None, None, None, ["audio"]),
        _Action(0, 0, 12, _ComputationType.RECV_B, (2,), 3, None, None, None, None, ["audio"]),
        _Action(0, 0, 13, _ComputationType.FULL_BACKWARD, (2), None, None, None, None, None, ["audio"]),
        _Action(0, 0, 14, _ComputationType.RECV_B, (3,), 3, None, None, None, None, ["audio"]),
        _Action(0, 0, 15, _ComputationType.FULL_BACKWARD, (3), None, None, None, None, None, ["audio"]),
    ]

    rank1_actions = [
        _Action(0, 1, 0, _ComputationType.FORWARD, (0), None, None, None, None, None, ["vision"]),
        _Action(0, 1, 1, _ComputationType.SEND_F, (0,), 3, None, None, None, None, ["vision"]),
        _Action(0, 1, 2, _ComputationType.FORWARD, (1), None, None, None, None, None, ["vision"]),
        _Action(0, 1, 3, _ComputationType.SEND_F, (1,), 3, None, None, None, None, ["vision"]),
        _Action(0, 1, 4, _ComputationType.FORWARD, (2), None, None, None, None, None, ["vision"]),
        _Action(0, 1, 5, _ComputationType.SEND_F, (2,), 3, None, None, None, None, ["vision"]),
        _Action(0, 1, 6, _ComputationType.FORWARD, (3), None, None, None, None, None, ["vision"]),
        _Action(0, 1, 7, _ComputationType.SEND_F, (3,), 3, None, None, None, None, ["vision"]),

        _Action(0, 1, 8, _ComputationType.RECV_B, (0,), 3, None, None, None, None, ["vision"]),
        _Action(0, 1, 9, _ComputationType.FULL_BACKWARD, (0), None, None, None, None, None, ["vision"]),
        _Action(0, 1, 10, _ComputationType.RECV_B, (1,), 3, None, None, None, None, ["vision"]),
        _Action(0, 1, 11, _ComputationType.FULL_BACKWARD, (1), None, None, None, None, None, ["vision"]),
        _Action(0, 1, 12, _ComputationType.RECV_B, (2,), 3, None, None, None, None, ["vision"]),
        _Action(0, 1, 13, _ComputationType.FULL_BACKWARD, (2), None, None, None, None, None, ["vision"]),
        _Action(0, 1, 14, _ComputationType.RECV_B, (3,), 3, None, None, None, None, ["vision"]),
        _Action(0, 1, 15, _ComputationType.FULL_BACKWARD, (3), None, None, None, None, None, ["vision"]),
    ]
    
    rank2_actions = [
        _Action(0, 2, 0, _ComputationType.FORWARD, (0), None, None, None, None, None, ["text"]),
        _Action(0, 2, 1, _ComputationType.SEND_F, (0,), 3, None, None, None, None, ["text"]),
        _Action(0, 2, 2, _ComputationType.FORWARD, (1), None, None, None, None, None, ["text"]),
        _Action(0, 2, 3, _ComputationType.SEND_F, (1,), 3, None, None, None, None, ["text"]),
        _Action(0, 2, 4, _ComputationType.FORWARD, (2), None, None, None, None, None, ["text"]),
        _Action(0, 2, 5, _ComputationType.SEND_F, (2,), 3, None, None, None, None, ["text"]),
        _Action(0, 2, 6, _ComputationType.FORWARD, (3), None, None, None, None, None, ["text"]),
        _Action(0, 2, 7, _ComputationType.SEND_F, (3,), 3, None, None, None, None, ["text"]),
        
        _Action(0, 2, 8, _ComputationType.RECV_B, (0,), 3, None, None, None, None, ["text"]),
        _Action(0, 2, 9, _ComputationType.FULL_BACKWARD, (0), None, None, None, None, None, ["text"]),
        _Action(0, 2, 10, _ComputationType.RECV_B, (1,), 3, None, None, None, None, ["text"]),
        _Action(0, 2, 11, _ComputationType.FULL_BACKWARD, (1), None, None, None, None, None, ["text"]),
        _Action(0, 2, 12, _ComputationType.RECV_B, (2,), 3, None, None, None, None, ["text"]),
        _Action(0, 2, 13, _ComputationType.FULL_BACKWARD, (2), None, None, None, None, None, ["text"]),
        _Action(0, 2, 14, _ComputationType.RECV_B, (3,), 3, None, None, None, None, ["text"]),
        _Action(0, 2, 15, _ComputationType.FULL_BACKWARD, (3), None, None, None, None, None, ["text"]),
    ]
    
    rank3_actions = [
        _Action(1, 3, 0, _ComputationType.RECV_F, (0,), 0, None, None, None, None, ["audio"]),
        _Action(1, 3, 1, _ComputationType.RECV_F, (0,), 1, None, None, None, None, ["vision"]),
        _Action(1, 3, 2, _ComputationType.RECV_F, (0,), 2, None, None, None, None, ["text"]),
        _Action(1, 3, 3, _ComputationType.FORWARD, (0), None, None, None, None, None, ["audio","vision","text"]),
        _Action(1, 3, 4, _ComputationType.SEND_F, (0,), 4, None, None, None, None, None),

        _Action(1, 3, 5, _ComputationType.RECV_F, (1,), 0, None, None, None, None, ["audio"]),
        _Action(1, 3, 6, _ComputationType.RECV_F, (1,), 1, None, None, None, None, ["vision"]),
        _Action(1, 3, 7, _ComputationType.RECV_F, (1,), 2, None, None, None, None, ["text"]),
        _Action(1, 3, 8, _ComputationType.FORWARD, (1), None, None, None, None, None, ["audio","vision","text"]),
        _Action(1, 3, 9, _ComputationType.SEND_F, (1,), 4, None, None, None, None, None),

        _Action(1, 3, 10, _ComputationType.RECV_F, (2,), 0, None, None, None, None, ["audio"]),
        _Action(1, 3, 11, _ComputationType.RECV_F, (2,), 1, None, None, None, None, ["vision"]),
        _Action(1, 3, 12, _ComputationType.RECV_F, (2,), 2, None, None, None, None, ["text"]),
        _Action(1, 3, 13, _ComputationType.FORWARD, (2), None, None, None, None, None, ["audio","vision","text"]),
        _Action(1, 3, 14, _ComputationType.SEND_F, (2,), 4, None, None, None, None, None),

        _Action(1, 3, 15, _ComputationType.RECV_F, (3,), 0, None, None, None, None, ["audio"]),
        _Action(1, 3, 16, _ComputationType.RECV_F, (3,), 1, None, None, None, None, ["vision"]),
        _Action(1, 3, 17, _ComputationType.RECV_F, (3,), 2, None, None, None, None, ["text"]),
        _Action(1, 3, 18, _ComputationType.FORWARD, (3), None, None, None, None, None, ["audio","vision","text"]),
        _Action(1, 3, 19, _ComputationType.SEND_F, (3,), 4, None, None, None, None, None),
        
        
        
        
        _Action(1, 3, 20, _ComputationType.RECV_B, (0,), 4, None, None, None, None, None),
        _Action(1, 3, 21, _ComputationType.FULL_BACKWARD, (0), None, None, None, None, None, ["audio","vision","text"]),
        _Action(1, 3, 22, _ComputationType.SEND_B, (0,), 0, None, None, None, None, ["audio"]),
        _Action(1, 3, 23, _ComputationType.SEND_B, (0,), 1, None, None, None, None, ["vision"]),
        _Action(1, 3, 24, _ComputationType.SEND_B, (0,), 2, None, None, None, None, ["text"]),
        
        _Action(1, 3, 25, _ComputationType.RECV_B, (1,), 4, None, None, None, None, None),
        _Action(1, 3, 26, _ComputationType.FULL_BACKWARD, (1), None, None, None, None, None, ["audio","vision","text"]),
        _Action(1, 3, 27, _ComputationType.SEND_B, (1,), 0, None, None, None, None, ["audio"]),
        _Action(1, 3, 28, _ComputationType.SEND_B, (1,), 1, None, None, None, None, ["vision"]),
        _Action(1, 3, 29, _ComputationType.SEND_B, (1,), 2, None, None, None, None, ["text"]),
        
        _Action(1, 3, 30, _ComputationType.RECV_B, (2,), 4, None, None, None, None, None),
        _Action(1, 3, 31, _ComputationType.FULL_BACKWARD, (2), None, None, None, None, None, ["audio","vision","text"]),
        _Action(1, 3, 32, _ComputationType.SEND_B, (2,), 0, None, None, None, None, ["audio"]),
        _Action(1, 3, 33, _ComputationType.SEND_B, (2,), 1, None, None, None, None, ["vision"]),
        _Action(1, 3, 34, _ComputationType.SEND_B, (2,), 2, None, None, None, None, ["text"]),
        
        _Action(1, 3, 35, _ComputationType.RECV_B, (3,), 4, None, None, None, None, None),
        _Action(1, 3, 36, _ComputationType.FULL_BACKWARD, (3), None, None, None, None, None, ["audio","vision","text"]),
        _Action(1, 3, 37, _ComputationType.SEND_B, (3,), 0, None, None, None, None, ["audio"]),
        _Action(1, 3, 38, _ComputationType.SEND_B, (3,), 1, None, None, None, None, ["vision"]),
        _Action(1, 3, 39, _ComputationType.SEND_B, (3,), 2, None, None, None, None, ["text"]),
    ]
    
    rank4_actions = [
        _Action(2, 4, 0, _ComputationType.RECV_F, (0,), 3, None, None, None, None, None),
        _Action(2, 4, 1, _ComputationType.FORWARD, (0), None, None, None, None, None, None),
        _Action(2, 4, 2, _ComputationType.SEND_F, (0,), 5, None, None, None, None, None),

        _Action(2, 4, 3, _ComputationType.RECV_F, (1,), 3, None, None, None, None, None),
        _Action(2, 4, 4, _ComputationType.FORWARD, (1), None, None, None, None, None, None),
        _Action(2, 4, 5, _ComputationType.SEND_F, (1,), 5, None, None, None, None, None),

        _Action(2, 4, 6, _ComputationType.RECV_F, (2,), 3, None, None, None, None, None),
        _Action(2, 4, 7, _ComputationType.FORWARD, (2), None, None, None, None, None, None),
        _Action(2, 4, 8, _ComputationType.SEND_F, (2,), 5, None, None, None, None, None),

        _Action(2, 4, 9, _ComputationType.RECV_F, (3,), 3, None, None, None, None, None),
        _Action(2, 4, 10, _ComputationType.FORWARD, (3), None, None, None, None, None, None),
        _Action(2, 4, 11, _ComputationType.SEND_F, (3,), 5, None, None, None, None, None),




        _Action(2, 4, 12, _ComputationType.RECV_B, (0,), 5, None, None, None, None, None),
        _Action(2, 4, 13, _ComputationType.FULL_BACKWARD, (0), None, None, None, None, None, None),
        _Action(2, 4, 14, _ComputationType.SEND_B, (0,), 3, None, None, None, None, None),

        _Action(2, 4, 15, _ComputationType.RECV_B, (1,), 5, None, None, None, None, None),
        _Action(2, 4, 16, _ComputationType.FULL_BACKWARD, (1), None, None, None, None, None, None),
        _Action(2, 4, 17, _ComputationType.SEND_B, (1,), 3, None, None, None, None, None),

        _Action(2, 4, 18, _ComputationType.RECV_B, (2,), 5, None, None, None, None, None),
        _Action(2, 4, 19, _ComputationType.FULL_BACKWARD, (2), None, None, None, None, None, None),
        _Action(2, 4, 20, _ComputationType.SEND_B, (2,), 3, None, None, None, None, None),

        _Action(2, 4, 21, _ComputationType.RECV_B, (3,), 5, None, None, None, None, None),
        _Action(2, 4, 22, _ComputationType.FULL_BACKWARD, (3), None, None, None, None, None, None),
        _Action(2, 4, 23, _ComputationType.SEND_B, (3,), 3, None, None, None, None, None),
    ]
    
    rank5_actions = [
        _Action(3, 5, 0, _ComputationType.RECV_F, (0,), 4, None, None, None, None, None),
        _Action(3, 5, 1, _ComputationType.FORWARD, (0), None, None, None, None, None, None),

        _Action(3, 5, 2, _ComputationType.RECV_F, (1,), 4, None, None, None, None, None),
        _Action(3, 5, 3, _ComputationType.FORWARD, (1), None, None, None, None, None, None),

        _Action(3, 5, 4, _ComputationType.RECV_F, (2,), 4, None, None, None, None, None),
        _Action(3, 5, 5, _ComputationType.FORWARD, (2), None, None, None, None, None, None),

        _Action(3, 5, 6, _ComputationType.RECV_F, (3,), 4, None, None, None, None, None),
        _Action(3, 5, 7, _ComputationType.FORWARD, (3), None, None, None, None, None, None),

        _Action(3, 5, 8, _ComputationType.FULL_BACKWARD, (0), None, None, None, None, None, None),
        _Action(3, 5, 9, _ComputationType.SEND_B, (0,), 4, None, None, None, None, None),
        
        _Action(3, 5, 10, _ComputationType.FULL_BACKWARD, (1), None, None, None, None, None, None),
        _Action(3, 5, 11, _ComputationType.SEND_B, (1,), 4, None, None, None, None, None),

        _Action(3, 5, 12, _ComputationType.FULL_BACKWARD, (2), None, None, None, None, None, None),
        _Action(3, 5, 13, _ComputationType.SEND_B, (2,), 4, None, None, None, None, None),

        _Action(3, 5, 14, _ComputationType.FULL_BACKWARD, (3), None, None, None, None, None, None),
        _Action(3, 5, 15, _ComputationType.SEND_B, (3,), 4, None, None, None, None, None),
    ]


    return {0: rank0_actions, 1: rank1_actions, 2: rank2_actions, 3: rank3_actions, 4: rank4_actions, 5: rank5_actions}

parser = argparse.ArgumentParser()
parser.add_argument("--train_steps", type=int, default=1,
                    help="The total number of steps for training. If omitted, run the entire DataLoader.")
parser.add_argument("--batch_size", type=int,
                    default=int(os.getenv("BATCH_SIZE", 20)),
                    help="The batch size of each rank (the environment variable BATCH_SIZE can be overridden)")
parser.add_argument("--microbatch_num", type=int,
                    default=int(os.getenv("MICROBATCH_NUM", 5)),
                    help="Micro-batch number (the environment variable MICROBATCH_NUM can be overridden)")
parser.add_argument("--sudo_pass", default=os.getenv("SUDO_PASS"),
                    help='Write the password of root')
parser.add_argument("--upstream", default=os.getenv("upstream"),
                    help='Write the upstream in mbps')
parser.add_argument("--plan_loc", type=str, required=True,
                    help='the json file that stores the sharding plans...')
args = parser.parse_args()
def main():
    import torch  # ensure local binding to avoid UnboundLocalError in some environments
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
        stage_mod = AudioStage(audio_enc)
        stage_mod.to(device)
        stage = PipelineStage_Multimodality(stage_mod, stage_index=0,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=None, this_group=[0], next_group=[3],
                            model_type = "audio",
                            mm_prev_groups = None)
        setattr(stage, "modal_type", "audio")
    elif rank == 1:
        stage_mod = VisionStage(vision_enc)
        stage_mod.to(device)
        stage = PipelineStage_Multimodality(stage_mod, stage_index=0,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=None, this_group=[1], next_group=[3],
                            model_type = "vision",
                            mm_prev_groups = None)
        setattr(stage, "modal_type", "vision")
    elif rank == 2:
        stage_mod = TextStage(text_model)
        stage_mod.to(device)
        stage = PipelineStage_Multimodality(stage_mod, stage_index=0,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=None, this_group=[2], next_group=[3],
                            model_type = "text",
                            mm_prev_groups = None)
        setattr(stage, "modal_type", "text")
        
    elif rank == 3:
        stage_mod = Stage1(text_model, L1)
        stage_mod.to(device)
        stage = PipelineStage_Multimodality(stage_mod, stage_index=1,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=[0,1,2], this_group=[3], next_group=[4],
                            model_type = "packing",
                            mm_prev_groups = {"audio":[0],"vision":[1],"text":[2]})
        setattr(stage, "modal_type", "packing")
    elif rank == 4:
        stage_mod = Stage2(text_model, L1, L2)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=2,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=[3], this_group=[4], next_group=[5])
    elif rank == 5:
        stage_mod = Stage3(thinker, text_model, L2)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=3,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=[4], this_group=[5], next_group=None)
    
    del thinker                        
    import gc; gc.collect()

    raw = load_dataset("jxie/flickr8k", split="train")

    def pick_caption(example):
        text = example.get("caption_0", None)
        if text is None:
            caps = example.get("captions")
            text = caps[0] if isinstance(caps, list) and caps else ""
        return {"text": text}

    keep_cols = {"image", "text"}
    raw = raw.map(pick_caption, remove_columns=[c for c in raw.column_names if c not in keep_cols])

    from PIL import Image

    noise_path = os.path.join(os.getcwd(), "noise.mp3")

    def collate_fn(batch):
        conversations = []
        for ex in batch:
            img = ex["image"]
            txt = ex.get("text", "") if isinstance(ex.get("text", ""), str) else ""
            conversations.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": noise_path},
                        {"type": "image", "image": img},
                        {"type": "text",  "text": txt}
                    ],
                }
            ])

        pack = proc.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            padding="max_length",
            max_length=512,
            truncation=True
        )

        # Debug once or twice: what keys does processor return, and are there audio keys?
        try:
            if not hasattr(collate_fn, "_dbg_count"):
                collate_fn._dbg_count = 0
            if collate_fn._dbg_count < 2:
                def _shape(x):
                    if isinstance(x, torch.Tensor):
                        return (tuple(x.shape), str(x.dtype))
                    return type(x).__name__
                keys = list(pack.keys()) if hasattr(pack, 'keys') else []
                print(f"[rank0] collate_fn: processor pack keys: {keys}")
                for k in ("pixel_values", "image_grid_thw", "input_values", "input_features", "audio_values", "feature_attention_mask"):
                    if hasattr(pack, 'get'):
                        v = pack.get(k, None)
                        if v is not None:
                            print(f"[rank0] collate_fn: pack[{k}] -> {_shape(v)}")
                        else:
                            print(f"[rank0] collate_fn: pack[{k}] -> None")
                collate_fn._dbg_count += 1
        except Exception:
            pass

        input_ids = pack["input_ids"]
        attention_mask = pack["attention_mask"]
        labels = input_ids.clone()

        special_ids = set([
            tok.pad_token_id,
            getattr(tok, "eos_token_id", None),
            getattr(tok, "bos_token_id", None),
            getattr(cfg, "image_token_id", None),
            getattr(cfg, "video_token_id", None),
            getattr(cfg, "audio_token_id", None),
        ])
        special_ids = {i for i in special_ids if i is not None}
        for sid in special_ids:
            labels[labels == sid] = -100

        # Vision packing
        vision_inputs = None
        pixel_values = pack.get("pixel_values", None)
        image_grid_thw = pack.get("image_grid_thw", None)
        if image_grid_thw is not None:
            image_grid_thw = torch.as_tensor(image_grid_thw, dtype=torch.long)
        if pixel_values is not None and image_grid_thw is not None and image_grid_thw.numel() > 0:
            counts = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2])
            slices, off = [], 0
            for n in counts.tolist():
                slices.append(pixel_values[off: off + n])
                off += n
            assert off == pixel_values.size(0), f"visual tokens mismatch: {off} != {pixel_values.size(0)}"
            vision_inputs = {
                "pixel_values_list": slices,
                "grid_thw": image_grid_thw,
            }

        # Audio packing: pass through whatever processor returned
        audio_inputs = None
        if hasattr(pack, 'keys'):
            for k in ("input_values", "audio_values", "input_features"):
                if k in pack:
                    audio_inputs = {k: pack[k]}
                    if "feature_attention_mask" in pack:
                        audio_inputs["feature_attention_mask"] = pack["feature_attention_mask"]
                    break

        # Debug once or twice: what did we build for audio_inputs?
        try:
            if not hasattr(collate_fn, "_dbg_audio_count"):
                collate_fn._dbg_audio_count = 0
            if collate_fn._dbg_audio_count < 2:
                if audio_inputs is None:
                    print(f"[rank0] collate_fn: built audio_inputs=None")
                else:
                    kv = {k: (tuple(v.shape), str(v.dtype)) if isinstance(v, torch.Tensor) else type(v).__name__
                          for k, v in audio_inputs.items()}
                    print(f"[rank0] collate_fn: built audio_inputs keys/shapes: {kv}")
                collate_fn._dbg_audio_count += 1
        except Exception:
            pass

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "vision_inputs": vision_inputs,
            "audio_inputs": audio_inputs,
        }

    batch_size = args.batch_size
    microbatch_num = args.microbatch_num
    block = 512

    if rank == 0:
        loader = torch.utils.data.DataLoader(
            raw,
            batch_size=batch_size,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def loss_fn(output, target):
        if output is None or target is None:
            return None
        vocab_size = output.size(-1)
        T_logits = output.size(1)
        T_labels = target.size(1)
        T = min(T_logits, T_labels)
        logits = output[:, :T-1, :].reshape(-1, vocab_size)
        labels = target[:, 1:T].reshape(-1)
        valid_mask = (labels >= -100) & (labels < vocab_size)
        if not valid_mask.all():
            invalid_labels = labels[~valid_mask]
            print(f"[rank{dist.get_rank()}] WARNING: Found invalid labels: {invalid_labels[:10]}...")
        return F.cross_entropy(logits, labels, ignore_index=-100)

    sched = PipelineScheduleRuntimeWithDirection([stage], n_microbatches=microbatch_num,
                                                loss_fn=loss_fn, root_pass=args.sudo_pass)
    actions = create_pipeline_actions()
    sched._load_actions(actions, format="compute_comms")

    opt = optim.Adam(stage_mod.parameters(), lr=1e-4)
    prev_loss = None
    
    
    for epoch in range(1):
        if rank == 0:
            steps_tensor = torch.tensor(len(loader) if args.train_steps is None else args.train_steps, device=device)
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

            # Prepare per-head inputs on rank 0 then broadcast
            if rank == 0:
                batch = next(data_iter)
                inp_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                vis_pack = batch.get("vision_inputs", None)
                aud_pack = batch.get("audio_inputs", None)
                tgt = batch["labels"].to(device)

                # Broadcast target tensor to all ranks
                dist.broadcast(tgt, src=0)

                # Broadcast text tensors (for other heads to use splitting if needed)
                dist.broadcast(inp_ids, src=0)
                dist.broadcast(attn, src=0)

                # Broadcast vision/audio packs as Python objects
                buf_vis = [vis_pack]
                dist.broadcast_object_list(buf_vis, src=0)
                buf_aud = [aud_pack]
                dist.broadcast_object_list(buf_aud, src=0)

                # Debug first two steps: what will we broadcast for audio
                if step < 2:
                    if aud_pack is None:
                        print(f"[rank0] train-step {step}: aud_pack=None (no audio_inputs)")
                    else:
                        try:
                            kv = {k: (tuple(v.shape), str(v.dtype)) if isinstance(v, torch.Tensor) else type(v).__name__
                                  for k, v in aud_pack.items()}
                            print(f"[rank0] train-step {step}: aud_pack keys/shapes: {kv}")
                        except Exception:
                            print(f"[rank0] train-step {step}: aud_pack present but cannot summarize")

                # Local step for audio head：作为 kwargs 传入，并带上 attention_mask 以便 microbatch 大小推断
                sched.step(audio_inputs=aud_pack, attention_mask=attn, target=tgt)

            else:
                # Receive target
                tgt = torch.zeros(batch_size, block, dtype=torch.long, device=device)
                dist.broadcast(tgt, src=0)

                # Receive text tensors for rank 2 (text head) or keep but ignore on others
                inp_ids = torch.zeros(batch_size, block, dtype=torch.long, device=device)
                attn = torch.zeros(batch_size, block, dtype=torch.long, device=device)
                dist.broadcast(inp_ids, src=0)
                dist.broadcast(attn, src=0)

                # Receive vision/audio packs
                buf_vis = [None]
                dist.broadcast_object_list(buf_vis, src=0)
                vis_pack = buf_vis[0]
                buf_aud = [None]
                dist.broadcast_object_list(buf_aud, src=0)
                aud_pack = buf_aud[0]

                # Debug first two steps: confirm reception on non-zero ranks
                if step < 2:
                    rk = rank
                    if aud_pack is None:
                        print(f"[rank{rk}] train-step {step}: received aud_pack=None")
                    else:
                        try:
                            kv = {k: (tuple(v.shape), str(v.dtype)) if isinstance(v, torch.Tensor) else type(v).__name__
                                  for k, v in aud_pack.items()}
                            print(f"[rank{rk}] train-step {step}: received aud_pack keys/shapes: {kv}")
                        except Exception:
                            print(f"[rank{rk}] train-step {step}: received aud_pack present but cannot summarize")

                if rank == 1:
                    # Vision head executes with vision inputs：作为 kwargs 传入，并带上 attention_mask 以便 microbatch 大小推断
                    sched.step(vision_inputs=vis_pack, attention_mask=attn, target=tgt)
                elif rank == 2:
                    # Text head executes with text inputs
                    sched.step(inp_ids, attention_mask=attn, target=tgt)
                else:
                    # Packing and later stages only need target to drive schedule
                    sched.step(target=tgt)

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

    # ===== Gather states and merge full model =====
    # Collect from each rank the submodule state_dict
    buf_audio = [stage_mod.state_dict()] if rank == 0 else [None]
    dist.broadcast_object_list(buf_audio, src=0)
    if rank == 0:
        audio_state = buf_audio[0]

    buf_vision = [stage_mod.state_dict()] if rank == 1 else [None]
    dist.broadcast_object_list(buf_vision, src=1)
    if rank == 0:
        vision_state = buf_vision[0]

    buf_text = [stage_mod.state_dict()] if rank == 2 else [None]
    dist.broadcast_object_list(buf_text, src=2)
    if rank == 0:
        text_state = buf_text[0]

    buf_s1 = [stage_mod.state_dict()] if rank == 3 else [None]
    dist.broadcast_object_list(buf_s1, src=3)
    if rank == 0:
        s1_state = buf_s1[0]

    buf_s2 = [stage_mod.state_dict()] if rank == 4 else [None]
    dist.broadcast_object_list(buf_s2, src=4)
    if rank == 0:
        s2_state = buf_s2[0]

    buf_s3 = [stage_mod.state_dict()] if rank == 5 else [None]
    dist.broadcast_object_list(buf_s3, src=5)
    if rank == 0:
        s3_state = buf_s3[0]

    if rank == 0:
        print("\nMerging and saving model (divide-head version)...")
        merged = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=True)
        merged_state = merged.state_dict()

        # 1) Text embed_tokens from rank2
        for k, v in text_state.items():
            if k.startswith("embed_tokens."):
                newk = "model.embed_tokens." + k[len("embed_tokens."):]
                merged_state[newk] = v

        # 2) Rotary from Stage1 (rank3) if present
        for k, v in s1_state.items():
            if k.startswith("rotary_emb."):
                newk = "model.rotary_emb." + k[len("rotary_emb."):]
                if newk in merged_state:
                    merged_state[newk] = v

        # 3) Audio tower from rank0
        for k, v in audio_state.items():
            if k.startswith("audio_enc."):
                newk = "audio_tower." + k[len("audio_enc."):]
                if newk in merged_state:
                    merged_state[newk] = v

        # 4) Vision tower from rank1
        for k, v in vision_state.items():
            if k.startswith("vision_enc."):
                newk = "visual." + k[len("vision_enc."):]
                if newk in merged_state:
                    merged_state[newk] = v

        # Helper to map layers
        def _map_layer_key(local_key: str, global_offset: int) -> str:
            parts = local_key.split(".")
            assert parts[0] == "layers", f"unexpected key {local_key}"
            li = int(parts[1]) + global_offset
            rest = ".".join(parts[2:])
            return f"model.layers.{li}.{rest}"

        # 5) Layers 0..L1-1 from Stage1
        for k, v in s1_state.items():
            if k.startswith("layers."):
                newk = _map_layer_key(k, 0)
                merged_state[newk] = v

        # 6) Layers L1..L2-1 from Stage2
        for k, v in s2_state.items():
            if k.startswith("layers."):
                newk = _map_layer_key(k, L1)
                merged_state[newk] = v

        # 7) Layers L2..end + norm + lm_head from Stage3
        for k, v in s3_state.items():
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

        merged.load_state_dict(merged_state, strict=False)
        save_dir = "trained_qwen_pp_devide_head"
        merged.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        print(f"Saved merged model to ./{save_dir}")

    dist.destroy_process_group()



if __name__ == "__main__":
    main()
