#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug Qwen2.5-Omni audio tower input expectations.

Runs local probes (no internet browsing) to check:
- Processor outputs (keys and shapes)
- Audio tower forward with synthetic features under different layouts
- Optional: test a given audio file via processor

Usage examples:
  python debug_qwen_omni_audio.py --model Qwen/Qwen2.5-Omni-3B
  python debug_qwen_omni_audio.py --model Qwen/Qwen2.5-Omni-3B --audio /path/to/audio.wav
  python debug_qwen_omni_audio.py --batch 1 --mels 128 --frames 160
"""

import argparse
import inspect
import sys
import traceback
from typing import Tuple

import torch


def try_import_transformers():
    try:
        import transformers  # type: ignore
        from transformers import AutoProcessor  # type: ignore
        from transformers.models.qwen2_5_omni import (  # type: ignore
            Qwen2_5OmniThinkerForConditionalGeneration,
        )
        return transformers, AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    except Exception as e:
        print("ERROR: cannot import transformers or Qwen2.5-Omni:", repr(e))
        traceback.print_exc()
        sys.exit(1)


def summarize_exc(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)}"


def test_audio_tower(audio_tower, feats: torch.Tensor, lens_dim_choice: int, with_aftercnn: bool) -> Tuple[bool, str]:
    """Run a single probe against audio_tower.forward.

    feats: [B, *, *]
    lens_dim_choice: 1 or 2 -> which dimension to use for feature_lens length
    with_aftercnn: whether to also pass aftercnn_lens (same shape as feature_lens)
    """
    B = feats.size(0)
    lens_len = int(feats.size(lens_dim_choice))
    feature_lens = torch.tensor([lens_len] * B, dtype=torch.long, device=feats.device)
    kwargs = {"input_features": feats, "feature_lens": feature_lens}
    if with_aftercnn:
        kwargs["aftercnn_lens"] = feature_lens.clone()

    try:
        out = audio_tower(**kwargs)
        if isinstance(out, torch.Tensor):
            last = out
        elif hasattr(out, "last_hidden_state") and isinstance(out.last_hidden_state, torch.Tensor):
            last = out.last_hidden_state
        elif isinstance(out, (list, tuple)):
            last = next((x for x in out if isinstance(x, torch.Tensor)), None)
        elif isinstance(out, dict):
            last = out.get("last_hidden_state", None)
        else:
            last = None
        shape_last = tuple(last.shape) if isinstance(last, torch.Tensor) else None
        return True, f"OK shape={shape_last}, lens_dim={lens_dim_choice}, aftercnn={with_aftercnn}"
    except Exception as e:
        return False, f"FAIL lens_dim={lens_dim_choice}, aftercnn={with_aftercnn}, error=({summarize_exc(e)})"


def main():
    transformers, AutoProcessor, Qwen = try_import_transformers()
    print("transformers version:", transformers.__version__)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--audio", default=None, help="optional path to audio file to probe via processor")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--mels", type=int, default=128)
    ap.add_argument("--frames", type=int, default=160)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print("Loading:", args.model)
    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen.from_pretrained(args.model, trust_remote_code=True)
    model.to(args.device)
    audio_tower = getattr(model, "audio_tower", None)
    if audio_tower is None:
        print("ERROR: model.audio_tower is None")
        return
    audio_tower.to(args.device)
    print("audio_tower:", audio_tower.__class__.__name__)
    try:
        print("audio_tower.forward signature:", inspect.signature(audio_tower.forward))
    except Exception:
        print("audio_tower.forward signature unavailable")

    # Processor probe (optional)
    if args.audio is not None:
        print("\n=== Processor audio probe ===")
        try:
            pack = proc(audio=[args.audio], return_tensors="pt")
            for k in ["input_features", "input_values", "feature_attention_mask"]:
                v = pack.get(k, None)
                if isinstance(v, torch.Tensor):
                    print(f"proc[{k}] -> shape={tuple(v.shape)}, dtype={v.dtype}")
                else:
                    print(f"proc[{k}] -> {type(v).__name__}")
            feats = pack.get("input_features", None)
            if isinstance(feats, torch.Tensor):
                feats = feats.to(args.device)
                for lens_dim in [1, 2]:
                    for wacc in [False, True]:
                        ok, msg = test_audio_tower(audio_tower, feats, lens_dim, wacc)
                        print("proc_input_features:", msg)
            else:
                print("processor did not produce input_features; skipping direct test")
        except Exception as e:
            print("processor audio probe failed:", summarize_exc(e))
            traceback.print_exc()

    # Synthetic probes
    print("\n=== Synthetic probes ===")
    B, M, T = args.batch, args.mels, args.frames

    # Case A: [B, n_mels, n_frames] (dim1=mels, dim2=frames)
    feats_A = torch.randn(B, M, T, device=args.device)
    print(f"Case A: feats_A shape={tuple(feats_A.shape)} (dim1=mels={M}, dim2=frames={T})")
    for lens_dim in [1, 2]:
        for wacc in [False, True]:
            ok, msg = test_audio_tower(audio_tower, feats_A, lens_dim, wacc)
            print("Case A:", msg)

    # Case B: [B, n_frames, n_mels] (dim1=frames, dim2=mels)
    feats_B = torch.randn(B, T, M, device=args.device)
    print(f"\nCase B: feats_B shape={tuple(feats_B.shape)} (dim1=frames={T}, dim2=mels={M})")
    for lens_dim in [1, 2]:
        for wacc in [False, True]:
            ok, msg = test_audio_tower(audio_tower, feats_B, lens_dim, wacc)
            print("Case B:", msg)

    print("\n=== Summary hints ===")
    print("- If you see errors like 'split_with_sizes expects ... at dimension 1', the tower splits along dim=1.")
    print("  In that case, feature_lens must sum to feats.size(1). Choose lens_dim=1 and feed feats with dim1=the sequence length it expects.")
    print("- Try matching processor-produced input_features layout (often [B, 128, frames]) and set feature_lens=[128] if it splits on dim=1.")


if __name__ == "__main__":
    main()

