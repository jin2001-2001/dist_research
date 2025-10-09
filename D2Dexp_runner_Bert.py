#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_cpu_hybrid_pp.py

This is the main experiment script for training Qwen-0.6B using a custom 
Hybrid Pipeline Parallelism framework built on top of PiPPy.

Key Features:
- CPU-only execution with 3 ranks
- Manual microbatch scheduling via custom action list
- Used for testing correctness and loss behavior of the hybrid PP runtime
"""
import json
from typing import Any, Dict, List, Union
import argparse, os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torch.distributed as dist

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import BertTokenizerFast, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling

from datasets import load_dataset
from tqdm import tqdm
import time

from stage_with_mutiple_ranks import PipelineStage_with_mutiple_ranks
from schedule_runtime import PipelineScheduleRuntimeWithDirection
from pipelining_source_code.schedules import _Action, _ComputationType

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from simple_1F1B_Action import generate_1f1b_pipeline_actions, generate_1f1b_pipeline_actions_pro

# ---------- Helpers ----------
def make_pad_attention_mask(input_ids, pad_id):
    # (B,T) -> attention_mask with 1 for real tokens, 0 for pad
    return (input_ids != pad_id).long()

#@torch.no_grad()
#def extend_mask(bert_module, attention_mask, input_shape, device):
    # HF BERT uses this extended mask in self-attention
    # attention_mask: (B,T) with 1 for keep, 0 for pad
#    return bert_module.get_extended_attention_mask(attention_mask, input_shape, device)

# ---------- Part 1: Embeddings + first N layers ----------
class PartStart(nn.Module):
    def __init__(self, bert_mlm: BertForMaskedLM, end: int):
        """
        end: number of encoder layers to include in this part (e.g., 0..end-1)
        """
        super().__init__()
        self.config = bert_mlm.config
        self.mask = bert_mlm.bert.get_extended_attention_mask                 # access helpers (mask extension)
        self.embeddings = bert_mlm.bert.embeddings
        self.layers = nn.ModuleList(bert_mlm.bert.encoder.layer[:end])  # 0..end-1

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Returns:
          hidden: (B,T,H)
          ext_mask: (B,1,1,T) extended attention mask
        """
        device = input_ids.device
        if attention_mask is None:
            attention_mask = make_pad_attention_mask(input_ids, pad_id=self.config.pad_token_id)

        # Embeddings (adds word + position + token_type)
        hidden = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        # Build extended mask once and carry it downstream
        ext_mask = self.mask(self.bert, attention_mask, input_ids.shape, device)

        # Run the first chunk of layers
        for layer in self.layers:
            hidden = layer(hidden_states=hidden, attention_mask=ext_mask)[0]

        return hidden.contiguous(), ext_mask.contiguous()

# ---------- Part 2: Middle layers only ----------
class PartMiddle(nn.Module):
    def __init__(self, bert_mlm: BertForMaskedLM, start: int, end: int):
        super().__init__()
        self.layers = nn.ModuleList(bert_mlm.bert.encoder.layer[start:end])

    def forward(self, hidden, ext_mask):
        for layer in self.layers:
            hidden = layer(hidden_states=hidden, attention_mask=ext_mask)[0]
        return hidden.contiguous(), ext_mask

# ---------- Part 3: Last layers + MLM head ----------
class PartEnd(nn.Module):
    def __init__(self, bert_mlm: BertForMaskedLM, start: int):
        super().__init__()
        self.layers = nn.ModuleList(bert_mlm.bert.encoder.layer[start:])
        self.cls = bert_mlm.cls  # BertOnlyMLMHead (tied to word embeddings)

    def forward(self, hidden, ext_mask):
        for layer in self.layers:
            hidden = layer(hidden_states=hidden, attention_mask=ext_mask)[0]
        logits = self.cls(hidden)  # (B,T,V)
        return logits

class PartWhole(nn.Module):                                # rank 4：25-27 + norm + lm_head
    def __init__(self, bert_mlm: BertForMaskedLM):

        super().__init__()
        self.config = bert_mlm.config
        self.mask = bert_mlm.bert.get_extended_attention_mask                 # access helpers (mask extension)
        self.embeddings = bert_mlm.bert.embeddings
        self.layers = nn.ModuleList(bert_mlm.bert.encoder.layer[:])  # 0..end-1
        self.cls = bert_mlm.cls  # BertOnlyMLMHead (tied to word embeddings)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Returns:
          hidden: (B,T,H)
          ext_mask: (B,1,1,T) extended attention mask
        """
        device = input_ids.device
        if attention_mask is None:
            attention_mask = make_pad_attention_mask(input_ids, pad_id=self.config.pad_token_id)

        # Embeddings (adds word + position + token_type)
        hidden = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        # Build extended mask once and carry it downstream
        ext_mask = self.mask(self.bert, attention_mask, input_ids.shape, device)

        # Run the first chunk of layers
        for layer in self.layers:
            hidden = layer(hidden_states=hidden, attention_mask=ext_mask)[0]
        logits = self.cls(hidden)
        return logits

def load_config(cfg:str) -> Dict[str, Any]:
    """Accepts a dict or a JSON file path."""
    if isinstance(cfg, str):
        with open(cfg, "r") as f:
            cfg = json.load(f)
    return cfg

def plan_parser(rank: int, world: int, cfg: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given the pipeline config (dict or JSON path) and a rank id,
    return the stage that contains this rank, plus the immediate
    previous/next stages' rank groups.
    """
    cfg = load_config(cfg)
    #if world != cfg["total_devices"]:
        #raise ValueError(f"Number of devices mismatch...")
    stages: List[Dict[str, Any]] = sorted(cfg["stage_info"], key=lambda s: s["index"])
    total_stages = cfg["total_stages"]
    total_batchs = cfg["total_batchs"]
    total_stage = cfg["total_stages"]
    # Find the stage containing the rank
    stage = next((s for s in stages if rank in s.get("ranks", [])), None)
    if stage is None:
        raise ValueError(f"Rank {rank} not found in any stage's ranks.")

    idx = stage["index"]
    shard_stage = idx

    this_g = stages[idx]["ranks"]
    prev_g = stages[idx - 1]["ranks"] if idx - 1 >= 0 else None
    next_g = stages[idx + 1]["ranks"] if idx + 1 < len(stages) else None
    
    is_group = 1 if len(this_g)>1 else 0
    is_final_stage = 1 if idx == total_stage-1 else 0

    shard_from = stages[idx]["shard_layer"][0]
    shard_to = stages[idx]["shard_layer"][1]

    return is_group,is_final_stage, shard_stage, shard_from, shard_to, prev_g, this_g, next_g, total_stages, total_batchs


def plan_batch_parser(cfg: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given the pipeline config (dict or JSON path) and a rank id,
    return the stage that contains this rank, plus the immediate
    previous/next stages' rank groups.
    """
    cfg = load_config(cfg)
    stages: List[Dict[str, Any]] = sorted(cfg["stage_info"], key=lambda s: s["index"])

    # Find the stage containing the rank
    # format of batch info: [[[0,1,2],[3,8,6]],...]

    #final [[[0,[0,1,2]],[1,[3,4,5,6,7,8,9,10]]..],...]
    final_batch_info = []
    final_group_info = []
    for s in stages:
        rank_group = s["ranks"]
        batch_alloc = s["batch_allocate"]
        smallest_rank = rank_group[0]
        final_group_info.append(rank_group)
        stage_l=[]
        cur_sample_index = 0
        id = 0
        for rank_index in rank_group:
            cur_sample_chunk = list(range(cur_sample_index, cur_sample_index+batch_alloc[id]))
            cur_sample_index = cur_sample_index+batch_alloc[id]
            stage_l.append([rank_index, cur_sample_chunk])
            id+=1
            

        final_batch_info.append(stage_l)

    return final_batch_info, final_group_info



parser = argparse.ArgumentParser()
parser.add_argument("--train_steps", type=int, default=2,
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

    dist.init_process_group("gloo", init_method="env://")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    device = torch.device("cpu")     

    name = "bert-base-uncased"

    tok = BertTokenizerFast.from_pretrained(name, trust_remote_code=True)
    full = BertForMaskedLM.from_pretrained(name, trust_remote_code=True)
    #tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    #tok.pad_token = tok.eos_token
    #full = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)


    is_group,is_final_stage, shard_stage, shard_from, shard_to, prev_g, this_g, next_g, total_stages, total_batchs = plan_parser(rank, world, args.plan_loc)

    if shard_stage == 0:
        print(f"shard_to {shard_to}")
        if total_stages != 1:
            stage_mod = PartStart(full,shard_to)
        else:
            stage_mod = PartWhole(full)
    elif is_final_stage == 1:
        print(f"shard_from {shard_from}")
        stage_mod = PartEnd(full,shard_from)
    else:
        print(f"shard_from {shard_from}")
        print(f"shard_to {shard_to}")
        stage_mod = PartMiddle(full, shard_from, shard_to)


    if is_group == 0:
        dist.barrier()
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=shard_stage,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=prev_g, this_group=this_g, next_group=next_g)
        
    else:
        dist.barrier()
        time.sleep(shard_stage)
        dp_group = dist.new_group(ranks=this_g, backend="gloo")
        print(f"dp组 {this_g} 初始化完成")
        stage_mod.to(device)
        print("进入DDP初始化")
        #Using DDP as the data parallelism component of our frame 
        time.sleep(shard_stage)
        stage_mod = DDP(
            stage_mod,
            device_ids=None,        # CPU don' use device_ids
            process_group=dp_group, # Used for local dp
            find_unused_parameters=False,
            init_sync = False,

            bucket_cap_mb=50,
            broadcast_buffers=False,
            gradient_as_bucket_view=True
        )        
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=shard_stage,
                                num_stages=world, device=device,
                                group=dist.group.WORLD,  # Used for world pp 
                                prev_group=prev_g, this_group=this_g, next_group=next_g)

    
    del full                        
    import gc; gc.collect()

    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    block = 128

    ##info:::
    def tok_fn(ex):
        # no special tokens here; we’ll add them after chunking
        return tok(ex["text"], add_special_tokens=False, return_attention_mask=False)

    def grp_fn(ex):
        concat = sum(ex["input_ids"], [])
        inner = block - 2  # room for [CLS] and [SEP]
        tot = (len(concat) // inner) * inner
        chunks = [concat[i:i+inner] for i in range(0, tot, inner)]
        # wrap each chunk with CLS/SEP → final length exactly 128
        ids = [[tok.cls_token_id] + c + [tok.sep_token_id] for c in chunks]
        return {"input_ids": ids}

    ds = (raw.map(tok_fn, batched=True, remove_columns=["text"])
              .map(grp_fn, batched=True))
    ds.set_format("torch", columns=["input_ids"])

    # ----- 2) Collator: dynamic MLM masking (creates attention_mask + labels) -----
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok,
        mlm=True,
        mlm_probability=0.15,   # standard BERT masking rate
    )

    # ----- 3) Dataloader (same shape semantics as your code) -----
    batch_size = args.batch_size
    microbatch_num = args.microbatch_num

    if stage.is_first:
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collator
        )
    
    def loss_fn(output, target):
        """
        output: logits, shape (B, T, V)
        target: labels, shape (B, T), with masked positions = original token id,
                and all other positions = -100 (ignored by CE)
        """
        if output is None or target is None:
            return None
        V = output.size(-1)
        return F.cross_entropy(output.reshape(-1, V),
                               target.reshape(-1),
                               ignore_index=-100)


    # jin: we get the total_batchs from plans, but make sure args input is scynized...
    if total_batchs!= int(args.microbatch_num):
        raise ValueError(f"Mbatch {total_batchs} not equal to {int(args.microbatch_num)},misbatch plan's assumption")


    print(f"n_microbatches {args.batch_size}")
    sched = PipelineScheduleRuntimeWithDirection([stage], n_microbatches=args.batch_size, loss_fn=loss_fn, root_pass=args.sudo_pass)

    # === Memory monitor: start & register containers (CPU/Gloo safe) ===

    rank_env = int(os.environ.get("RANK", str(rank)))  # 若已定义 rank 变量，可直接用
    # monitor = MemoryMonitor(
    #     log_path=f"/tmp/mem_rank{rank_env}.jsonl",
    #     interval_s=0.5,
    #     include_tensors=True,      # 若性能有感知，可改为 False
    #     top_tensors=50,            # 多列一些大对象
    #     cuda_only_tensors=False,   # 关键：CPU-only 必须 False
    # )
    # 逐容器统计（尽量多挂一些你关心的 dict）
    # try:
    #     if hasattr(stage, "fwd_cache"):
    #        # monitor.register_container("fwd_cache", stage.fwd_cache)
    #     if hasattr(stage, "bwd_cache"):
    #        # monitor.register_container("bwd_cache", stage.bwd_cache)
    # except Exception as e:
    #     print(f"[rank {rank_env}] register stage caches failed: {e}")

    # sched 内部“打包后的前向缓存”（如果存在就监控）
    # try:
    #    # monitor.register_container("big_fwd_cache", getattr(sched, "_big_fwd_cache"))
    # except Exception as e:
    #     print(f"[rank {rank_env}] register sched big cache failed: {e}")

   # monitor.start()
    # === end ===

    
    
    batch_info,group_info = plan_batch_parser(args.plan_loc)
    actions = generate_1f1b_pipeline_actions_pro(num_stages= total_stages, total_samples = args.batch_size, num_microbatches= args.microbatch_num,
                                                 group_info=group_info, batch_info=batch_info,
                                                  upstream = args.upstream)
    print(f"对应action {actions[dist.get_rank()]}")
    
    sched._load_actions(actions, format="compute_comms")
    
    opt = optim.Adam(stage_mod.parameters(), lr=1e-4)            
    prev_loss = None
    
    
    for epoch in range(1):
        if stage.is_first:
            if args.train_steps is None:
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
        
        try:
            for step in range(total_steps):
                step_start_time = time.time()
                opt.zero_grad(set_to_none=True)

               # with monitor.section("sched.step"):
                if stage.is_first:
                    batch = next(data_iter)
                    inp = batch["input_ids"].to(device)
                    tgt = batch["labels"].to(device)
                    dist.broadcast(tgt, src=0)
                    sched.step(inp, target=tgt)
                else:
                    tgt = torch.empty(batch_size, block, dtype=torch.long, device=device)
                    dist.broadcast(tgt, src=0)
                    sched.step(None, target=tgt)

              #  with monitor.section("opt.step"):
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
                
                if cur_loss is not None:
                    delta = (cur_loss - prev_loss) if prev_loss is not None else None

                    print(f"[rank0] step {step+1} loss {cur_loss:.4f}"
                            + ("" if prev_loss is None else (" down" if cur_loss < prev_loss else " up" if cur_loss > prev_loss else " flat")))
                    prev_loss = cur_loss
                
                
                
                dist.barrier()
            # try:
            #    # monitor.stop()
            # except Exception:
            #     pass    
            
        except Exception as e:
            # 出错时抓一份即时快照到 stdout / log
          #  snap = monitor.snapshot()
          #  print("[MEM-SNAPSHOT-ON-ERROR]", snap)
            raise
        
        if rank == 0:
            pbar.close()
            total_time = time.time() - start_time
            print(f"\nEpoch {epoch+1} completed in {total_time:.2f}s")
            print(f"Average speed: {total_steps / total_time:.2f} steps/s")
            
    

    #if rank == 0:
    #    print("\nMerging and saving model...")
    #    cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
    #    merged = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    #    merged_state = merged.state_dict()
    #    
    #    part1_state = stage_mod.state_dict()
    #    for key in part1_state:
    #        if key.startswith("embed_tokens"):
    #            merged_state[f"model.{key}"] = part1_state[key]
    #        elif key.startswith("layers"):
    #            merged_state[f"model.{key}"] = part1_state[key]
    #        elif key.startswith("rotary_emb"):
    #            merged_state[f"model.{key}"] = part1_state[key]
    #    
    #    recv = [None]
    #    dist.broadcast_object_list(recv, src=1)
    #    part2_state = recv[0]
    #    
    #    for key in part2_state:
    #        if key.startswith("layers"):
    #            layer_idx = int(key.split(".")[1]) + 14
    #            new_key = f"model.layers.{layer_idx}" + key[key.find(".", 7):]
    #            merged_state[new_key] = part2_state[key]
    #        elif key == "norm.weight":
    #            merged_state["model.norm.weight"] = part2_state[key]
    #        elif key == "lm_head.weight":
    #            merged_state["lm_head.weight"] = part2_state[key]
    #            
    #    merged.load_state_dict(merged_state, strict=False)
    #    merged.save_pretrained("trained_qwen_pp")
    #    tok.save_pretrained("trained_qwen_pp")
    #    print("Saved to ./trained_qwen_pp")
    #else:
    #    dist.broadcast_object_list([stage_mod.state_dict()], src=1)

    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
    #test correctness of plan_parser...
    #print(plan_parser(0, 4, "./D2Dexp_scratch.json"))