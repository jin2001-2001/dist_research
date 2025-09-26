import torch
import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names
from datasets import Dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from huggingface_hub import login
import time
import csv
import argparse
import os
import torch.distributed as dist
from transformers import TrainerCallback
#from bitsandbytes.optim import AdamW8bit
from torch.optim import SGD
from pynvml import *

login(token="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")


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    return info.used//1024**2

class DevicePrint(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        for name, p in list(model.named_parameters()):
            print(f"{name} -> {p.device}")
        #control.should_training_stop = True  # optional: stop after printing

class CudaCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()         # Release cached blocks to OS
        torch.cuda.synchronize()         # Force CUDA to finish frees


def training_estimation(model_id = "Qwen/Qwen3-0.6B", output_file_path = "./offloading.csv"):
    # Configuration
    vocab_size = 1024
    batch_size = 24
    seq_len = 256

    # Generate synthetic training data
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_dict({
        "input_ids": x,
        "labels": y
    })

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

#    optimizer = SGD(model.parameters(), lr=5e-5)
    training_args = TrainingArguments(
        output_dir="./hf_results",
        per_device_train_batch_size=3,
        num_train_epochs=1,
        logging_steps=1,
        gradient_accumulation_steps=1, 
        #gradient_checkpointing=True,
        #optim="adafactor",
        save_strategy="no",
        report_to="none",
        deepspeed="./ds_config.json"
    )


    ####construct 8-bit adam: 
    #decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    #decay_parameters = [name for name in decay_parameters if "bias" not in name]
    #optimizer_grouped_parameters = [
    #    {
    #        "params": [p for n, p in model.named_parameters() if n in decay_parameters],
    #        "weight_decay": training_args.weight_decay,
    #    },
    #    {
    #        "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
    #        "weight_decay": 0.0,
    #    },
    #]
    #
    #optimizer_kwargs = {
    #    "betas": (training_args.adam_beta1, training_args.adam_beta2),
    #    "eps": training_args.adam_epsilon,
    #}
    #optimizer_kwargs["lr"] = training_args.learning_rate
    #adam_bnb_optim = bnb.optim.Adam8bit(
    #    optimizer_grouped_parameters,
    #    betas=(training_args.adam_beta1, training_args.adam_beta2),
    #    eps=training_args.adam_epsilon,
    #    lr=training_args.learning_rate,
    #)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset
        #optimizers=(adam_bnb_optim, None)
        #callbacks=[DevicePrint, CudaCleanupCallback]
    )

    result = trainer.train()
    print_gpu_utilization()
    print(result)
    dist.destroy_process_group()

    headers = ["device","model", "mode","status","train_steps_per_second"]
    #line = f"{args.device}, {args.model_id}, {args.maxt}, {throughput}, {total_new_tokens}, {total_time}, {duration}, {cpu_energy}, {gpu_energy}, {ram_energy}, {energy_consumed}\n"
    duration = result.metrics['train_steps_per_second']
    row = ["5070",model_id, "dp","False",duration]
    write_header = not os.path.exists(output_file_path)

    with open(output_file_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerow(row)

if __name__ == "__main__":
    model_ids=[
  "Qwen/Qwen3-0.6B",
  "Qwen/Qwen3-1.7B",
  "Qwen/Qwen3-4B",
  "Qwen/Qwen3-14B"]
    training_estimation(model_id = "Qwen/Qwen3-1.7B")
