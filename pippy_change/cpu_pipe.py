#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two-stage pipeline-parallel training for Qwen-0.6B  (PyTorch ≥ 2.5)

GPU-0:   embedding + layers 0-13
GPU-1:   layers 14-27 + ln + lm_head
"""

import os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torch.distributed as dist

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import time

from stage_with_mutiple_ranks import PipelineStage_with_mutiple_ranks
from schedule_runtime import PipelineScheduleRuntimeWithDirection
from torch.distributed.pipelining.schedules import _Action, _ComputationType

def create_pipeline_actions():

    
    # Rank 0 (Stage 0) 的操作序列
    rank0_actions = [
        
        _Action(0, 0, _ComputationType.FORWARD, 0, None),
        _Action(0, 0, _ComputationType.SEND_F, 0, 1),
        
        _Action(0, 0, _ComputationType.FORWARD, 1, None),
        _Action(0, 0, _ComputationType.SEND_F, 1, 1),
        
        _Action(0, 0, _ComputationType.FORWARD, 2, None),
        _Action(0, 0, _ComputationType.SEND_F, 2, 1),
        
        _Action(0, 0, _ComputationType.RECV_B, 0, 1),
        _Action(0, 0, _ComputationType.FULL_BACKWARD, 0, None),
        
        _Action(0, 0, _ComputationType.FORWARD, 3, None),
        _Action(0, 0, _ComputationType.SEND_F, 3, 1),
        
        _Action(0, 0, _ComputationType.RECV_B, 1, 1),
        _Action(0, 0, _ComputationType.FULL_BACKWARD, 1, None),

        _Action(0, 0, _ComputationType.RECV_B, 2, 1),
        _Action(0, 0, _ComputationType.FULL_BACKWARD, 2, None),

        _Action(0, 0, _ComputationType.RECV_B, 3, 1),
        _Action(0, 0, _ComputationType.FULL_BACKWARD, 3, None),
        
                
    ]
    
    # Rank 1 (Stage 1) 的操作序列
    rank1_actions = [

        _Action(1, 1, _ComputationType.RECV_F, 0, 0),
        _Action(1, 1, _ComputationType.FORWARD, 0, None),
        _Action(1, 1, _ComputationType.FULL_BACKWARD, 0, None),
        _Action(1, 1, _ComputationType.SEND_B, 0, 0),
        _Action(1, 1, _ComputationType.SEND_B, 3, 0),
        
        _Action(1, 1, _ComputationType.RECV_F, 1, 0),
        _Action(1, 1, _ComputationType.FORWARD, 1, None),
        _Action(1, 1, _ComputationType.FULL_BACKWARD, 1, None),
        _Action(1, 1, _ComputationType.SEND_B, 1, 0),
        
        _Action(1, 1, _ComputationType.RECV_F, 2, 0),
        _Action(1, 1, _ComputationType.FORWARD, 2, None),
        _Action(1, 1, _ComputationType.FULL_BACKWARD, 2, None),
        _Action(1, 1, _ComputationType.SEND_B, 2, 0),
        
        _Action(1, 1, _ComputationType.RECV_F, 3, 0),
        _Action(1, 1, _ComputationType.FORWARD, 3, None),
        _Action(1, 1, _ComputationType.FULL_BACKWARD, 3, None),
        
        
                
        
    ]
    
    return {0: rank0_actions, 1: rank1_actions}

class Part1(nn.Module):  # rank 0
    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layers = nn.ModuleList(model.model.layers[:14])
        self.rotary_emb = model.model.rotary_emb

    def forward(self, input_ids):
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1).contiguous()
        hidden = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden, position_ids)
        attention_mask = torch.triu(
            torch.full((seqlen, seqlen), float('-inf'), device=device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).contiguous()

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                output_attentions=False,
                use_cache=False,
            )
            hidden = layer_outputs[0]

        return hidden.contiguous(), attention_mask.contiguous()


class Part2(nn.Module):  # rank 1
    def __init__(self, model):
        super().__init__()
        self.layers = nn.ModuleList(model.model.layers[14:])
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.rotary_emb = model.model.rotary_emb

    def forward(self, hidden, attention_mask):
        bsz, seqlen = hidden.shape[:2]
        device = hidden.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1).contiguous()
        position_embeddings = self.rotary_emb(hidden, position_ids)

        if attention_mask.dim() == 2:
            seqlen = attention_mask.shape[-1]
            attention_mask = torch.triu(
                torch.full((seqlen, seqlen), float('-inf'), device=device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0).expand(hidden.shape[0], 1, -1, -1).contiguous()
        elif not attention_mask.is_contiguous():
            attention_mask = attention_mask.contiguous()

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                output_attentions=False,
                use_cache=False,
            )
            hidden = layer_outputs[0]

        hidden = self.norm(hidden)
        return self.lm_head(hidden)


def main():

    dist.init_process_group("gloo", init_method="env://")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    device = torch.device("cpu")     

    name = "Qwen/Qwen3-0.6B-Base"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    full = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)

    stage_mod = Part1(full) if rank == 0 else Part2(full)
    stage_mod.to(device)
    stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=rank,
                          num_stages=world, device=device,
                          group=dist.group.WORLD)

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
    
    if rank == 0:
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)

    def loss_fn(output, target):
        if output is None or target is None:
            return None
        vocab_size = output.size(-1)
        output = output[:, :-1, :].reshape(-1, vocab_size)
        target = target[:, 1:].reshape(-1)
        return F.cross_entropy(output, target)

    sched = PipelineScheduleRuntimeWithDirection([stage], n_microbatches=4, loss_fn=loss_fn)
    actions = create_pipeline_actions()
    sched._load_actions(actions, format="compute_comms")
    
    opt = optim.Adam(stage_mod.parameters(), lr=1e-4)

    for epoch in range(1):
        if rank == 0:
            steps = torch.tensor(len(loader), device=device)
            dist.broadcast(steps, 0)
            data_iter = iter(loader)
            print(f"Total training steps: {steps.item()}")
        else:
            steps = torch.tensor(0, device=device)
            dist.broadcast(steps, 0)

        if rank == 0:
            pbar = tqdm(total=int(steps.item()), 
                       desc=f"Training Epoch {epoch+1}",
                       unit="step",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            start_time = time.time()
        
        for step in range(int(steps.item())):
            step_start_time = time.time()
            opt.zero_grad(set_to_none=True)
            
            if rank == 0:
                batch = next(data_iter)
                inp = batch["input_ids"].to(device)
                tgt = batch["labels"].to(device)
                dist.broadcast(tgt, src=0)
                sched.step(inp, target=tgt)
            else:
               
                tgt = torch.empty(8, block, dtype=torch.long, device=device)
                dist.broadcast(tgt, src=0)
                sched.step(target=tgt)
            
            opt.step()
            
            if rank == 0:
                step_time = time.time() - step_start_time
                tokens_processed = 8 * block
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
            print(f"Average speed: {steps.item() / total_time:.2f} steps/s")

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
        print("✅ Saved to ./trained_qwen_pp")
    else:
        dist.broadcast_object_list([stage_mod.state_dict()], src=1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
