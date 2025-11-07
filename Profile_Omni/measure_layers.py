#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-module forward/backward latency profiling for Qwen-Omni.

Profiles:
 - vision encoder
 - audio encoder
 - thinker backbone
"""

import torch, time, json
import torch.nn as nn
#from transformers import AutoModelForSpeechTextVision, AutoProcessor
from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from transformers import AutoProcessor
import torch, torch.nn as nn, torch.nn.functional as F

# ----------------------------
# Utility: register hooks
# ----------------------------
def hook_module_list(modules, label):
    fwd_times, bwd_times = [0.0]*len(modules), [0.0]*len(modules)
    start_fwd, start_bwd = [0.0]*len(modules), [0.0]*len(modules)

    def pre_f(i):
        def fn(_, __):
            start_fwd[i] = time.perf_counter()
        return fn
    def post_f(i):
        def fn(_, __, ___):
            fwd_times[i] += time.perf_counter() - start_fwd[i]
        return fn
    def pre_b(i):
        def fn(_, __):
            start_bwd[i] = time.perf_counter()
        return fn
    def post_b(i):
        def fn(_, __, ___):
            bwd_times[i] += time.perf_counter() - start_bwd[i]
        return fn

    handles = []
    for i, blk in enumerate(modules):
        handles.append(blk.register_forward_pre_hook(pre_f(i)))
        handles.append(blk.register_forward_hook(post_f(i)))
        if hasattr(blk, "register_full_backward_pre_hook"):
            handles.append(blk.register_full_backward_pre_hook(pre_b(i)))
            handles.append(blk.register_full_backward_hook(post_b(i)))
    return handles, fwd_times, bwd_times


def print_summary(name, fwd, bwd, iters):
    print(f"\n=== {name} ===")
    for i, (f,b) in enumerate(zip(fwd, bwd)):
        print(f"Layer {i:02d}: fwd {f/iters*1000:.2f} ms, bwd {b/iters*1000:.2f} ms")
    print(f"Total {name} forward: {sum(f)/iters*1000:.2f} ms")
    print(f"Total {name} backward: {sum(b)/iters*1000:.2f} ms")

# ----------------------------
# Main
# ----------------------------
def main():
    model_name = "Qwen/Qwen2.5-Omni-3B"
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    #proc = AutoProcessor.from_pretrained(model_name)

    vision = model.thinker.visual
    audio  = model.thinker.audio_tower
    thinker = model.thinker.model

    # --- dummy inputs similar to your dataset ---
    img = torch.randn(1, 3, 224, 224)
    aud = torch.randn(1, 80, 2048)
    aud_mask = torch.ones(1, 2048, dtype=torch.int32)
    seq_len = 256
    hidden = thinker.embed_tokens.embedding_dim
    ids = torch.randint(0, 32000, (1, seq_len))
    attn = torch.ones_like(ids)
    embeds = torch.randn(1, seq_len, hidden)

    # --- find layers ---
    v_blocks = list(vision.encoder.blocks)
    a_blocks = list(audio.layers)
    t_blocks = list(thinker.layers)

    hv, fv, bv = hook_module_list(v_blocks, "Vision")
    ha, fa, ba = hook_module_list(a_blocks, "Audio")
    ht, ft, bt = hook_module_list(t_blocks, "Thinker")

    iters, warmup = 1, 0

    for i in range(iters):
        if i < warmup:
            with torch.no_grad():
                _ = vision(pixel_values=img)
                _ = audio(input_features=aud, feature_attention_mask=aud_mask)
                _ = thinker(inputs_embeds=embeds, attention_mask=attn)
            continue

        # Vision encoder
        out_v = vision(pixel_values=img)
        loss_v = out_v[0].sum()
        loss_v.backward()

        # Audio encoder
        out_a = audio(input_features=aud, feature_attention_mask=aud_mask)
        loss_a = out_a[0].sum()
        loss_a.backward()

        # Thinker backbone
        out_t = thinker(inputs_embeds=embeds, attention_mask=attn)
        loss_t = out_t.last_hidden_state.sum()
        loss_t.backward()

    # --- summarize ---
    print_summary("Vision Encoder", fv, bv, iters)
    print_summary("Audio Encoder", fa, ba, iters)
    print_summary("Thinker Backbone", ft, bt, iters)

    for h in hv+ha+ht:
        h.remove()

if __name__ == "__main__":
    main()
