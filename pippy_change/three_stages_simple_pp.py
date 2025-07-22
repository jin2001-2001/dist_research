#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two-stage pipeline-parallel training for Qwen-0.6B  (PyTorch ≥ 2.5)

GPU-0:   embedding + layers 0-13
GPU-1:   layers 14-27 + ln + lm_head
"""

import argparse, os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torch.distributed as dist

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import time

from stage_with_mutiple_ranks import PipelineStage_with_mutiple_ranks
from schedule_runtime import PipelineScheduleRuntimeWithDirection
from torch.distributed.pipelining.schedules import _Action, _ComputationType

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from simple_1F1B_Action import generate_1f1b_pipeline_actions

class Part1(nn.Module):  # rank 0
    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layers = nn.ModuleList(model.model.layers[:7])  # 0-6
        self.rotary_emb = model.model.rotary_emb

    def forward(self, input_ids):
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
        hidden = self.embed_tokens(input_ids)
        pos_emb = self.rotary_emb(hidden, position_ids)

        attn_mask = torch.triu(torch.full((seqlen, seqlen), float('-inf'), device=device), 1)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).contiguous()

        for layer in self.layers:
            hidden = layer(hidden_states=hidden,
                           attention_mask=attn_mask,
                           position_ids=position_ids,
                           position_embeddings=pos_emb,
                           output_attentions=False,
                           use_cache=False)[0]

        return hidden.contiguous(), attn_mask.contiguous()



class Part2(nn.Module):  # rank 1
    def __init__(self, model):
        super().__init__()
        self.layers = nn.ModuleList(model.model.layers[7:21])  # 7-18
        self.rotary_emb = model.model.rotary_emb

    def forward(self, hidden, attn_mask):
        bsz, seqlen = hidden.shape[:2]
        device = hidden.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
        pos_emb = self.rotary_emb(hidden, position_ids)

        if attn_mask.dim() == 2:
            attn_mask = torch.triu(torch.full((seqlen, seqlen), float('-inf'), device=device), 1)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).contiguous()
        elif not attn_mask.is_contiguous():
            attn_mask = attn_mask.contiguous()

        for layer in self.layers:
            hidden = layer(hidden_states=hidden,
                           attention_mask=attn_mask,
                           position_ids=position_ids,
                           position_embeddings=pos_emb,
                           output_attentions=False,
                           use_cache=False)[0]
        return hidden.contiguous(), attn_mask

class Part3(nn.Module):  # rank 2
    def __init__(self, model):
        super().__init__()
        self.layers = nn.ModuleList(model.model.layers[21:])  # 21-27
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.rotary_emb = model.model.rotary_emb

    def forward(self, hidden, attn_mask):
        bsz, seqlen = hidden.shape[:2]
        device = hidden.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
        pos_emb = self.rotary_emb(hidden, position_ids)

        if attn_mask.dim() == 2:
            attn_mask = torch.triu(torch.full((seqlen, seqlen), float('-inf'), device=device), 1)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).contiguous()
        elif not attn_mask.is_contiguous():
            attn_mask = attn_mask.contiguous()

        for layer in self.layers:
            hidden = layer(hidden_states=hidden,
                           attention_mask=attn_mask,
                           position_ids=position_ids,
                           position_embeddings=pos_emb,
                           output_attentions=False,
                           use_cache=False)[0]

        hidden = self.norm(hidden)
        return self.lm_head(hidden)  # logits


parser = argparse.ArgumentParser()
parser.add_argument("--train_steps", type=int, default=None,
                    help="The total number of steps for training. If omitted, run the entire DataLoader.")
parser.add_argument("--batch_size", type=int,
                    default=int(os.getenv("BATCH_SIZE", 16)),
                    help="The batch size of each rank (the environment variable BATCH_SIZE can be overridden)")
parser.add_argument("--microbatch_num", type=int,
                    default=int(os.getenv("MICROBATCH_NUM", 4)),
                    help="Micro-batch number (the environment variable MICROBATCH_NUM can be overridden)")
parser.add_argument("--profile_batch", default=os.getenv("PROFILE_BATCH", "0"),
                    help='Write the value of the environment variable PROFILE_BATCH')
args = parser.parse_args()
os.environ["PROFILE_BATCH"] = args.profile_batch
def main():

    dist.init_process_group("gloo", init_method="env://")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    device = torch.device("cpu")     

    name = "Qwen/Qwen3-0.6B-Base"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    full = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)

    if rank == 0:
        stage_mod = Part1(full)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=0,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=None, this_group=[0], next_group=[1,2])
        
    elif rank == 1:
        stage_mod = Part2(full)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=1,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=[0], this_group=[1], next_group=[2])
    else:
        stage_mod = Part3(full)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=2,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=[1], this_group=[2], next_group=None)

    del full                        
    import gc; gc.collect()
    
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    block = 128
    def tok_fn(ex): 
        return tok(ex["text"], return_attention_mask=False)
    def grp_fn(ex):
        concat = sum(ex["input_ids"], [])
        tot = len(concat) // block * block
        ids = [concat[i:i+block] for i in range(0, tot, block)]
        return {"input_ids": ids, "labels": [x[:] for x in ids]}
    
    ds = (raw.map(tok_fn, batched=True, remove_columns=["text"])
              .map(grp_fn, batched=True))
    ds.set_format("torch", columns=["input_ids", "labels"])
    
    batch_size = args.batch_size
    microbatch_num = args.microbatch_num
    
    if rank == 0:
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    def loss_fn(output, target):
        if output is None or target is None:
            return None
        vocab_size = output.size(-1)
        output = output[:, :-1, :].reshape(-1, vocab_size)
        target = target[:, 1:].reshape(-1)
        return F.cross_entropy(output, target)

    
    sched = PipelineScheduleRuntimeWithDirection([stage], n_microbatches=microbatch_num, loss_fn=loss_fn)
    actions = generate_1f1b_pipeline_actions(3,microbatch_num) # 3 stages, 8 microbatchs 
    sched._load_actions(actions, format="compute_comms")
    
    opt = optim.Adam(stage_mod.parameters(), lr=1e-4)            
    
    for epoch in range(1):
        if rank == 0:
            if args.train_steps is None:
                # 跑满一个 epoch
                steps_tensor = torch.tensor(len(loader), device=device)
            else:
                steps_tensor = torch.tensor(args.train_steps, device=device)

            dist.broadcast(steps_tensor, 0)
            data_iter = iter(loader)
            print(f"Total training steps: {steps_tensor.item()}")
        else:
            steps_tensor = torch.tensor(0, device=device)
            dist.broadcast(steps_tensor, 0)

        total_steps = int(steps_tensor.item())

        if rank == 0:
            pbar = tqdm(total=int(total_steps), 
                       desc=f"Training Epoch {epoch+1}",
                       unit="step",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            start_time = time.time()
        
        for step in range(total_steps):
            step_start_time = time.time()
            opt.zero_grad(set_to_none=True)
            
            if rank == 0:
                batch = next(data_iter)
                inp = batch["input_ids"].to(device)
                tgt = batch["labels"].to(device)
                dist.broadcast(tgt, src=0)
                sched.step(inp, target=tgt)
            else:
               
                tgt = torch.empty(batch_size, block, dtype=torch.long, device=device)
                dist.broadcast(tgt, src=0)
                sched.step(target=tgt)
            
            if (step + 1) % 50 == 0:
                sched.timeline_rec.events.clear()
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
            
            dist.barrier()
        
        if rank == 0:
            pbar.close()
            total_time = time.time() - start_time
            print(f"\nEpoch {epoch+1} completed in {total_time:.2f}s")
            print(f"Average speed: {total_steps / total_time:.2f} steps/s")

    if rank == 0:
        print("\nMerging and saving model...")
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        merged = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        merged_state = merged.state_dict()
        
        part1_state = stage_mod.state_dict()
        for key in part1_state:
            if key.startswith("embed_tokens"):
                merged_state[f"model.{key}"] = part1_state[key]
            elif key.startswith("layers"):
                merged_state[f"model.{key}"] = part1_state[key]
            elif key.startswith("rotary_emb"):
                merged_state[f"model.{key}"] = part1_state[key]
        
        recv = [None]
        dist.broadcast_object_list(recv, src=1)
        part2_state = recv[0]
        
        for key in part2_state:
            if key.startswith("layers"):
                layer_idx = int(key.split(".")[1]) + 14
                new_key = f"model.layers.{layer_idx}" + key[key.find(".", 7):]
                merged_state[new_key] = part2_state[key]
            elif key == "norm.weight":
                merged_state["model.norm.weight"] = part2_state[key]
            elif key == "lm_head.weight":
                merged_state["lm_head.weight"] = part2_state[key]
                
        merged.load_state_dict(merged_state, strict=False)
        merged.save_pretrained("trained_qwen_pp")
        tok.save_pretrained("trained_qwen_pp")
        print("Saved to ./trained_qwen_pp")
    else:
        dist.broadcast_object_list([stage_mod.state_dict()], src=1)

    
    sched.save_timeline("qwen_pp")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
