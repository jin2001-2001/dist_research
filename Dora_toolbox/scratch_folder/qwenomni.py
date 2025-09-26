import torch
import io
import math
import numpy as np
from PIL import Image

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
#from qwen_omni_utils import process_mm_info

# default: Load the model on the available device(s)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-3B", torch_dtype="auto", device_map="auto")


torch.onnx.export(
    model,
    (sample["input_ids"], sample["attention_mask"]),
    "qwen25omni.onnx",
    input_names=["input_ids","attention_mask"],
    output_names=["logits"],
    opset_version=17,
    dynamic_axes={"input_ids": {0: "batch",1: "seq"}}
)

# Fast, coarse view
print(model)  # shows a nested nn.Module tree
