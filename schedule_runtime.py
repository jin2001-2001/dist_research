import copy
import csv
import itertools
import logging
from pathlib import Path
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from enum import Enum
import sys
from typing import Any, Callable, NamedTuple, Optional, Union, Dict, Tuple, List, Set
import json
from dataclasses import dataclass, asdict
import copy, time
import os, socket, subprocess, functools, fcntl, tempfile
ROOT_PASS = None              
_last_rate = None

import threading
import torch
import torch.distributed as dist
import pipelining_source_code.schedules as schedule
from pipelining_source_code.schedules import _Action, _ComputationType
from pipelining_source_code.stage import _normalize_model_output_as_tuple
from pipelining_source_code._utils import flatten_args
from torch.distributed.distributed_c10d import _get_default_store
import atexit, signal, threading, json
logger = logging.getLogger(__name__)

FORWARD = _ComputationType.FORWARD
BACKWARD_INPUT = _ComputationType.BACKWARD_INPUT
BACKWARD_WEIGHT = _ComputationType.BACKWARD_WEIGHT
UNSHARD = _ComputationType.UNSHARD
RESHARD = _ComputationType.RESHARD
SEND_F = _ComputationType.SEND_F
RECV_F = _ComputationType.RECV_F
SEND_B = _ComputationType.SEND_B
RECV_B = _ComputationType.RECV_B
FULL_BACKWARD = _ComputationType.FULL_BACKWARD
ALL_REDUCE = _ComputationType.ALL_REDUCE

# Convenience shorthand for compute actions only since they are used in 'simple schedule format'
F = FORWARD
I = BACKWARD_INPUT
W = BACKWARD_WEIGHT
B = FULL_BACKWARD

_EXEC_DONE: Set[int] = set()

_STORE = None

master_addr = os.environ.get("MASTER_ADDR", "10.10.0.2")  # 你的主节点IP
master_port = 29501
world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])
start_daemon = (rank == 0)
def _get_store():
    global _STORE
    if _STORE is None:
        print("初始化")
        _STORE = dist.FileStore("/local/desk/rdzv_file", world_size=3)
        print("初始化结束")
        #_STORE = _get_default_store()   # Only the first real get
    return _STORE

def _reset_exec_done():
    _EXEC_DONE.clear()

def _mark_done(batch_id: int, action_id: int | None):
    if action_id is not None:
        _EXEC_DONE.add(action_id)
        key = f"batch_{batch_id}_done_{dist.get_rank()}_{action_id}"
        _get_store().set(key, b"1")


def _has_done(action_id: int) -> bool:
    return action_id in _EXEC_DONE

def _wait_remote_id(batch_id: int, owner_rank: int, dep_id: int, timeout: float | None = None):
    """
    Only block the current rank; Wait for owner_rank to mark dep_id as completed.
    """
    key = f"batch_{batch_id}_done_{owner_rank}_{dep_id}"
    store = _get_store()
    if timeout is None:
        store.wait([key])      
    else:
        start = time.time()
        while True:
            try:
                store.get([key])
                break
            except RuntimeError:
                if time.time() - start > timeout:
                    raise TimeoutError(f"wait_remote_id timeout on {key}")

def _mark_done_chunk(batch_id: int, action_id: int, chunk_idx: int):
    
    key = f"batch_{batch_id}_done_{dist.get_rank()}_{action_id}_c{chunk_idx}"
    print(f"已经做完了 {key}")
    _get_store().set(key, b"1")

def _wait_remote_chunk(batch_id: int, owner_rank: int, dep_id: int, dep_chunk: int, timeout: float | None = None):
    key = f"batch_{batch_id}_done_{owner_rank}_{dep_id}_c{dep_chunk}"
    store = _get_store()
    if timeout is None:
        if key == "batch_0_done_2_0_c0":
            val = store.get("batch_0_done_2_0_c0")
            print(f"✅✅✅ {val}")

            import threading, time
            from datetime import timedelta
            def monitor_key(store, key):
                import time
                while True:
                    try:
                        print("开始查询")
                        # 等待 0.1 秒，如果超时说明 key 不存在
                        store.wait([key])
                        print(f"{key} 已存在")
                    except RuntimeError:
                        print(f"{key} 不存在")
                    time.sleep(1)  # 每秒检查一次


            # 启动一个后台线程来打印
            t = threading.Thread(target=monitor_key, args=(store, key), daemon=True)
            t.start()
        store.wait([key])
    else:
        start = time.time()
        while True:
            try:
                store.wait([key], timeout=timeout)
                break
            except RuntimeError:
                if time.time() - start > timeout:
                    raise TimeoutError(f"wait_remote_chunk timeout on {key}")

NIC = os.getenv("PP_BW_IF", "eth0")   # The network card can be specified through environment variables
_LOCK = f"/tmp/pp_bw_lock_{NIC}"
_ORIG_FILE = f"/tmp/pp_orig_qdisc_{NIC}.txt"
_last_rate = None 
_restore_done = False
_lock = threading.Lock() 

def _run_tc(cmd: list[str]):
    base = ["sudo", "-n"] if os.geteuid() else []
    subprocess.run(base + cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _save_original_qdisc():
    if os.path.exists(_ORIG_FILE):
        return
    with open(_ORIG_FILE, "w") as f:
        subprocess.run(["tc", "qdisc", "show", "dev", NIC], check=True, stdout=f)

def _restore_qdisc():
    global _restore_done
    if _restore_done or not os.path.exists(_ORIG_FILE):
        return
    with _lock:
        _restore_done = True
        subprocess.run(["sudo", "-n", "tc", "qdisc", "del", "dev", NIC, "root"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(_ORIG_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("qdisc"):
                    _run_tc(line.split())
        os.remove(_ORIG_FILE)
        print(f"[bw] restored original qdisc on {NIC}")

# Register and exit hook
atexit.register(_restore_qdisc)
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda *_: (_restore_qdisc(), sys.exit(1)))

def _tc_set_rate(mbps: int | None):
    """
    System-level rate limiting: Change the root qdisc to tbf rate X mbit.
    mbps = None indicates no speed limit (using 40Gbit).
    Only LOCAL_RANK==0 is called; the rest of the ranks return directly.
    """
    global _last_rate
    if int(os.getenv("LOCAL_RANK", "0")) != 0:
        return
    rate = 40000 if mbps is None else mbps         
    if rate == _last_rate:
        return

    with open(_LOCK, "w") as fp:                
        fcntl.flock(fp, fcntl.LOCK_EX)
        _save_original_qdisc()
        cmd = ["tc","qdisc","replace","dev", NIC, "root", "tbf",
               "rate", f"{rate}mbit",
               "burst","32kbit","latency","400ms"]
        _run_tc(cmd)
        _last_rate = rate
        print(f"[bw] set {NIC} rate -> {rate} mbit/s")


def _sudo(cmd: list[str]):
    """以 root 执行；能用 -n 就用；否则喂 password"""
    if os.geteuid() == 0:     
        subprocess.run(cmd, check=True)
        return
    base = ["sudo", "-n"] if ROOT_PASS is None else ["sudo", "-S"]
    subprocess.run(base + cmd,
                   check=True,
                   input=(ROOT_PASS + "\n").encode() if ROOT_PASS else None)



def _cat_like(items: list[Any]) -> Any:
    """
    递归拼接：支持 Tensor / 标量0维Tensor / tuple / list / dict / None。
    Tensor 默认按 dim=0 cat；0维用 stack。
    容器按键或下标对齐后递归调用本函数。
    """
    first = items[0]
    # --- 新增：dict 支持 ---
    if isinstance(first, dict):
        out = {}
        keys = first.keys()
        # 断言所有字典 key 集合一致（或做一次交集/并集按需处理）
        for it in items[1:]:
            assert isinstance(it, dict) and it.keys() == keys, "kwargs dict keys mismatch across microbatches"
        for k in keys:
            out[k] = _cat_like([it[k] for it in items])
        return out

    # 原有逻辑
    if torch.is_tensor(first):
        if first.dim() == 0:
            return torch.stack(items, dim=0)
        else:
            return torch.cat(items, dim=0)
    elif isinstance(first, (tuple, list)):
        pieces = [_cat_like([itm[i] for itm in items]) for i in range(len(first))]
        return type(first)(pieces)
    else:
        # 非张量非容器：取第一个（要求各 MB 一致）
        for it in items[1:]:
            assert it == first, "non-tensor kwarg values must be identical across packed MBs"
        return first


# Safely split the tensor into n parts in the batch dimension; If the number of blocks is less than n, the last block is copied to make up for it
def _safe_chunk(t: torch.Tensor, n: int) -> List[torch.Tensor]:
    chunks = list(torch.chunk(t, min(n, t.size(0)), dim=0))
    if len(chunks) < n:
        pad = chunks[-1].detach().clone()
        chunks += [pad] * (n - len(chunks))
    return chunks

# Recursively split the "packaged object" (tensor/tuple/list)
def _chunk_like(big_obj, n: int):
    if torch.is_tensor(big_obj):
        return _safe_chunk(big_obj, n)

    if isinstance(big_obj, (tuple, list)):
        # Recursively split each element; The result is list[list[Tensor]]
        split_per_elem = [_chunk_like(o, n) for o in big_obj]
        out = []
        for i in range(n):
            out.append(tuple(split_per_elem[j][i] for j in range(len(big_obj))))
        return out

    # Make n copies of other types exactly as they are
    return [big_obj for _ in range(n)]

# Recursively complete the gradient: Make the structure and length of grad exactly the same as those of ref; Missing or None → zeros_like
def _fill_grads_like(ref, grad):
    if torch.is_tensor(ref):
        if grad is None:
            return torch.zeros_like(ref, requires_grad=False)
        return grad

    if isinstance(ref, (tuple, list)):
        g_list = list(grad) if isinstance(grad, (tuple, list)) else []
        while len(g_list) < len(ref):
            g_list.append(None)
        filled = [_fill_grads_like(r, g) for r, g in zip(ref, g_list)]
        return type(ref)(filled)

    # 非张量非序列：保持原样或用 ref 占位
    return grad if grad is not None else ref

# ---------------------------------------------------------------------------
# Action-string helpers (UPDATED FORMAT: [stage],[rank],[action type],[microbatch],[dest_rank],[upstream])
# ---------------------------------------------------------------------------

# ORIGINAL: _action_regex = re.compile(r"(\d+)(F|I|B|W|UNSHARD|RESHARD|SEND_F|RECV_F|SEND_B|RECV_B)(\d*)")
# REPLACED with a pattern that understands the *comma separated* five‑field format.


class PipelineScheduleRuntimeWithDirection(schedule.PipelineScheduleMulti):

    def __init__(self, stages, n_microbatches, loss_fn = None, args_chunk_spec = None, kwargs_chunk_spec = None, output_merge_spec = None, use_full_backward = None, scale_grads = True, root_pass: str | None = None):
        super().__init__(stages, n_microbatches, loss_fn, args_chunk_spec, kwargs_chunk_spec, output_merge_spec, use_full_backward, scale_grads)
        self._internal_losses: dict[int, torch.Tensor] = {}
        self.last_backward = False
        self.grad_recv_info_copy = None
        self.last_step_loss = None
        # 异步 SEND 的 work 容器（按 batch 存），以及互斥锁
        self._async_send_works = defaultdict(list)   # key: batch_id -> List[List[dist.Work]]
        self._async_send_lock = threading.Lock()
        # —— 异步 RECV 的容器（按 (stage_idx, mb) 存），以及互斥锁 ——
        self._async_recv_lock = threading.Lock()
        # RECV_F
        self._fwd_recv_works: dict[tuple[int,int], list[dist.Work]] = {}
        self._fwd_recv_posted: dict[tuple[int,int], threading.Event] = {}
        # RECV_B
        self._bwd_recv_works: dict[tuple[int,int], list[dist.Work]] = {}
        self._bwd_recv_posted: dict[tuple[int,int], threading.Event] = {}

        
        
        self._big_fwd_cache: dict[tuple[int, int], tuple[tuple[torch.Tensor, ...], list[torch.Tensor]]] = {}
        
        global ROOT_PASS
        ROOT_PASS = root_pass
        
        self._split_parts = int(os.getenv("PP_COMM_SPLITS", "8"))

        
        #Each node prepares HTB at one time
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            _save_original_qdisc() 
        if dist.is_initialized():
            dist.barrier()
    
    def _spawn_chunked_recv_worker(self, *, kind: str, action, ops, plan, current_batch: int, stage_idx: int, mb_index: int, action_id: int):
        """
        在独立线程中执行：
        for chunk in plan:
            立即 post 该 chunk 的 irecv (batch_isend_irecv) -> 得到 works_k
            将 works_k 追加进对应 key 的 works 列表
            recorder.record_async(..., chunk_idx=chunk_idx) 异步轮询完成，并 _mark_done_chunk
        循环结束 → 置位 “已全部 post 完成” 的 Event
        """
        assert kind in ("RECV_F", "RECV_B")
        key = (stage_idx, mb_index)

        def worker():
            pos = 0
            for chunk_idx, cnt in enumerate(plan):
                if cnt <= 0:
                    continue
                sub_ops = ops[pos:pos+cnt]; pos += cnt

                # 立即 post 该 chunk 的 irecv
                works_k = schedule._batch_p2p(sub_ops)
                print(f"[{dist.get_rank()}] POST {kind} st{stage_idx} mb{mb_index} "
                        f"chunk{chunk_idx} ops={len(works_k)}")
                
                # 记录到异步容器
                with self._async_recv_lock:
                    if kind == "RECV_F":
                        self._fwd_recv_works.setdefault(key, []).extend(works_k)
                    else:
                        self._bwd_recv_works.setdefault(key, []).extend(works_k)

                # 逐 chunk 记录（完成后 _mark_done_chunk，供 SEND 的 chunk 依赖等待）
                start_ns_k = time.time_ns()
                self._rec.record_async(
                    current_batch+1, action_id, kind, stage_idx, mb_index,
                    works_k, start_ns_k, chunk_idx=chunk_idx
                )

            # 全部 chunk 的 irecv 都已 post
            with self._async_recv_lock:
                if kind == "RECV_F":
                    ev = self._fwd_recv_posted.get(key)
                else:
                    ev = self._bwd_recv_posted.get(key)
            if ev is not None:
                ev.set()

        t = threading.Thread(
            target=worker, name=f"{kind}_st{stage_idx}_mb{mb_index}_b{current_batch+1}", daemon=True
        )
        t.start()

    
    def _spawn_chunked_send_worker(self, *, kind: str, action, ops, plan, chunk_deps,
                                current_batch: int, stage_idx: int, mb_index: int):
        """
        在独立线程中执行：
        for chunk in plan:
            等待依赖(如果有) -> batch_isend_irecv(sub_ops) -> 把 works 放入 self._async_send_works[current_batch+1]
            （可选）record_async 仅用于日志 (chunk_idx=None)，避免误写完成标记
        """
        assert kind in ("SEND_F", "SEND_B")

        def worker():
            pos = 0
            for chunk_idx, cnt in enumerate(plan):
                if cnt <= 0:
                    continue
                sub_ops = ops[pos:pos+cnt]; pos += cnt

                # 只允许 SEND 依赖 RECV 的 chunk 完成
                if chunk_deps and chunk_idx in chunk_deps:
                    for (dep_rank, dep_action_id, dep_chunk) in chunk_deps[chunk_idx]:
                        dep_key = f"batch_{current_batch+1}_done_{dep_rank}_{dep_action_id}_c{dep_chunk}"
                        print(f"[{dist.get_rank()}] SEND {kind} st{stage_idx} mb{mb_index} "
                            f"chunk{chunk_idx} waiting {dep_key}")
                        try:
                            _wait_remote_chunk(current_batch+1, dep_rank, dep_action_id, dep_chunk)
                        except TimeoutError:
                            print(f"[{dist.get_rank()}] TIMEOUT waiting {dep_key} "
                                f"(check remote RECV posting/completion)")
                            raise
                        print(f"[{dist.get_rank()}] SEND {kind} st{stage_idx} mb{mb_index} "
                            f"chunk{chunk_idx} dep OK: {dep_key}")
                print(f"\n这里mb {mb_index} chunk {chunk_idx}")
                # 提交该 chunk 的发送
                works_k = schedule._batch_p2p(sub_ops)

                # 记录到异步容器，批尾统一等待
                with self._async_send_lock:
                    self._async_send_works[current_batch+1].append(works_k)

                # （可选）如果你也想记录发送侧 chunk 的时间线，可解开下面注释
                # start_ns_k = time.time_ns()
                # schedule._rec.record_async(current_batch+1, action.id, kind, stage_idx, mb_index,
                #                            works_k, start_ns_k, chunk_idx=None)  # 注意 None：不写 mark_done_chunk

        t = threading.Thread(
            target=worker, name=f"{kind}_st{stage_idx}_mb{mb_index}_b{current_batch+1}", daemon=True
        )
        t.start()

    
    
    def _maybe_compute_loss(self, stage, output, target_mbs, mb_index):
        if stage.is_last and self._has_backward:
            loss = self._compute_loss(output, target_mbs[mb_index])  # type: ignore[index]
            self._internal_losses[mb_index] = loss     # ← 用 mb_index 做键
    
    def _maybe_get_loss(self, stage, mb_index):
        if stage.is_last and self._has_backward:
            if mb_index in self._internal_losses:
                return self._internal_losses[mb_index]
            else:
                raise RuntimeError(
                    f"Loss for microbatch {mb_index} is not available. "
                    f"Available losses for microbatches: {list(self._internal_losses.keys())}"
                )
        else:
            return None

    def _update_losses(self, stages, losses):
        if not isinstance(stages, list):
            stages = [stages]

        contains_last_stage = any(stage.is_last for stage in stages)

        if contains_last_stage and losses is not None:
            if len(self._internal_losses) != self._n_microbatches:
                raise RuntimeError(
                    f"Expecting {self._n_microbatches} losses but got "
                    f"{len(self._internal_losses)}"
                )

            losses.clear()
            for idx in range(self._n_microbatches):
                losses.append(self._internal_losses[idx])

        self._internal_losses.clear()

        
    def _load_actions(
        self,
        actions: dict[int, list[Optional[_Action]]],
        format: str = "compute_only",
    ):
        
        # validate the provided actions are valid and overrides the default stage_index_to_group_rank
        super()._validate_and_set_stage_mapping(actions)

        self.pipeline_order_with_comms: dict[int, list[_Action]] = {}
        if format == "compute_comms":
            for rank in actions:
                self.pipeline_order_with_comms[rank] = []
                for action in actions[rank]:
                    assert action is not None
                    self.pipeline_order_with_comms[rank].append(action)
            # TODO what level of validation should we offer for compute+comms schedule?
        elif format == "compute_only":
            # Perform schedule lowering
            for rank in actions:
                self.pipeline_order_with_comms[rank] = (
                    actions[rank]
                )

            self.pipeline_order_with_comms = schedule._add_send_recv(
                self.pipeline_order_with_comms,
                stage_to_rank=lambda s: self.stage_index_to_group_rank[s],
                num_stages=self._num_stages,
            )
        else:
            raise NotImplementedError(f"{format=} is not implemented")

    def _load_csv(self, filename: str, format: str = "compute_only"):
        """Loads a csv in simple format and then lowers it to include communication actions"""
        if format == "compute_only":
            super()._load_csv(filename)
            self._load_actions(self.pipeline_order)
        elif format == "compute_comms":
            actions = {}
            with open(filename, newline="", encoding="utf-8-sig") as csvfile:
                reader = csv.reader(csvfile)
                for rank, row in enumerate(reader):
                    action_list = []
                    for s in row:
                        s_clean = s.strip().strip('"')
                        if s_clean.count(",") != 4:
                            raise RuntimeError(f"Invalid action string (not 5 fields): {s_clean}")
                        try:
                            action = _Action.from_str(s_clean)
                        except Exception as e:
                            raise RuntimeError(f"Failed to parse action string: {s_clean}") from e
                        action_list.append(action)
                    actions[rank] = action_list
            self._load_actions(actions, format=format)
        else:
            raise NotImplementedError(f"{format=} is not implemented")


    def _dump_csv(self, filename: str):
        """Dump a CSV representation of the compute + comms schedule into a file with the provided filename."""
        # TODO should there be an option to dump the compute_only schedule from PipelineScheduleRuntime? It's possible
        # that it does not exist if it was created from a compute_comms schedule.
        assert self.pipeline_order_with_comms is not None, (
            "Must initialize compute_comms schedule before dump_csv"
        )
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for rank in self.pipeline_order_with_comms:
                writer.writerow(self.pipeline_order_with_comms[rank])

    def _simulate(self):
        pass
        # return schedule._simulate_comms_compute(
        #     self.pipeline_order_with_comms,
        #     lambda s: self.stage_index_to_group_rank[s],
        #     self._num_stages,
        # )

    def _step_microbatches(
        self,
        arg_mbs: Optional[list] = None,
        kwarg_mbs: Optional[list] = None,
        target_mbs: Optional[list] = None,
        losses: Optional[list] = None,
    ):
        """
        Operate on the microbatches for looped schedules (multiple stages on each rank).

        TODO: Does not use sorted_batch_isend_irecv(). As a result, this schedule does
        not support models with skip connections.
        """ 
        _reset_exec_done()          # At the beginning of each batch, clear the completion table
        
        local_bwd_budget: Counter[int] = Counter()
        for act in self.pipeline_order_with_comms[self.rank]:
            if act.computation_type in (FULL_BACKWARD, BACKWARD_WEIGHT):
                local_bwd_budget[act.stage_index] += 1
                
        self._pack_groups: Dict[tuple[int, int], int] = getattr(self, "_pack_groups", {})
        self._mb_to_group: Dict[tuple[int, int], tuple[int, int]] = getattr(self, "_mb_to_group", {})
                      
        if not hasattr(self, "_global_batch"):
            # env_batch = os.getenv("PROFILE_BATCH")         
            # self._profile_batch = int(env_batch) if env_batch is not None else None

            from recorder import Recorder
            self._rec = Recorder(rank=self.rank,net_sample_interval_ms=10)       
            self._timeline_saved = False
            self._global_batch = -1
        
        current_batch = self._global_batch
        self._global_batch += 1
        
        # record_this_batch = (
        #     self._profile_batch is not None                   
        #     and current_batch == self._profile_batch        
        # )
        
        record_this_batch = True
        
        self._rec.set_enabled(record_this_batch)
        
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        if not self._stages_initialized:
            self._initialize_stages(arg_mbs[0], kwarg_mbs[0])
        
        # Based on the plan in Step 1 created in __init__:
        # 2. Perform communication based on the pipeline_order
        stage_index_to_stage: dict[int, schedule._PipelineStageBase] = {
            stage.stage_index: stage for stage in self._stages
        }
        
        assert self.pipeline_order_with_comms is not None, (
            "Must call _load_actions() before calling _step_microbatches()"
        )

        # recv ops indexed by (stage_idx, mb_idx) need to be waited on before use
        bwd_recv_ops: dict[tuple[int, int], list[dist.Work]] = {}
        fwd_recv_ops: dict[tuple[int, int], list[dist.Work]] = {}

        # send ops should be waited on before step() exists, mainly for hygeine
        send_ops: list[list[dist.Work]] = []

        # we track which stages are 'active' when used with FSDP, and wait on unshard ops before computing on stages
        unshard_ops: dict[int, schedule.UnshardHandle] = {}
        unsharded_stages = set()

        def _assert_unsharded(stage_idx: int):
            """If an unshard is active for `stage_idx`, wait() it and mark `stage_idx` unshared."""
            if stage_idx in unshard_ops:
                unshard_ops[stage_idx].wait()
                del unshard_ops[stage_idx]
                unsharded_stages.add(stage_idx)
            assert stage_idx in unsharded_stages, (
                f"Attempted to compute on sharded {stage_idx=}"
            )

        # count either full_backward or backward_weight together, to determine when to sync DP grads
        backward_counter: Counter[int] = Counter()
        for time_step, action in enumerate(self.pipeline_order_with_comms[self.rank]):
            try:
                deps = action.dependency
                #print(f"{action.id} {action.computation_type} has dependency {deps}")
                if deps:                              # Dependency format: {rank: (id1, id2, ...)}
                    # Local Dependency
                    while not all(_has_done(d) for d in deps.get(self.rank, ())):
                        print(f"{action.computation_type} local wait")
                        time.sleep(0.001)

                    # Remote Dependency 
                    for dep_rank, ids in deps.items():
                        if dep_rank == self.rank:
                            continue
                        for dep_id in ids:
                            print(f"{action.computation_type} remote wait")
                            _wait_remote_id(current_batch+1,dep_rank, dep_id)
                
                comp_type = action.computation_type
                mb_field = action.microbatch_index
                action_id = action.id
                if mb_field is None:
                    mb_ids: tuple[int, ...] = ()           
                elif isinstance(mb_field, (tuple, list)):
                    mb_ids = tuple(mb_field)                  
                else: 
                    mb_ids = (mb_field,)

                mb_index: int = mb_ids[0] if mb_ids else -1   
               
                rank = action.rank
                dest_rank = action.dest_rank

                if comp_type not in (UNSHARD, RESHARD, ALL_REDUCE):
                    assert mb_ids and all(m >= 0 for m in mb_ids), \
                        f"{action=} missing mb_index"
                stage_idx = action.stage_index
                stage = stage_index_to_stage[stage_idx]
                stage_uses_fsdp = isinstance(stage.submod, schedule.FSDPModule)
                
                is_next_stage_on_this_rank = stage_idx + 1 in stage_index_to_stage
                is_prev_stage_on_this_rank = stage_idx - 1 in stage_index_to_stage
                
                if(self.grad_recv_info_copy is None):
                    self.grad_recv_info_copy = copy.deepcopy(stage.grad_recv_info)

                
                logger.debug(
                    "_PipelineScheduleRuntime running time_step %d, action %s",
                    time_step,
                    action,
                )
             
                if comp_type == SEND_F:
                    _tc_set_rate(action.upstream)
                    if action.upstream is not None:
                        print(f"[{dist.get_rank()}]: batch {current_batch+1} SEND_F microbatch {mb_index}, upstream bandwidth {action.upstream} mbps")
                    else:
                        print(f"[{dist.get_rank()}]: batch {current_batch+1} SEND_F microbatch {mb_index}")
                    
                    # num_splits = action.split_parts or 1

                    # ops = (
                    #     stage.get_fwd_send_ops(mb_index, rank=rank, dest_rank=dest_rank, num_splits=num_splits)
                    #     if rank is not None and dest_rank is not None
                    #     else stage.get_fwd_send_ops(mb_index, rank=None, dest_rank=None, num_splits=num_splits)
                    # )
                    # # 读取 stage 填写的计划（每块的 op 数）
                    # plan = stage._last_comm_plan.get(("SEND_F", mb_index), [len(ops)])  # 后备：一整块
                    # pos = 0
                    # for chunk_idx, cnt in enumerate(plan):
                    #     sub_ops = ops[pos:pos+cnt]; pos += cnt

                    #     # === 二级依赖：只允许依赖 RECV_* 的“分块完成” ===
                    #     if action.chunk_deps and chunk_idx in action.chunk_deps:
                    #         for (dep_rank, dep_action_id, dep_chunk) in action.chunk_deps[chunk_idx]:
                    #             _wait_remote_chunk(current_batch+1, dep_rank, dep_action_id, dep_chunk)

                    #     # [SEND-GATE] 你的阻塞/整形逻辑可继续写在 stage 的 send 循环里（不冲突）
                    #     works_k = schedule._batch_p2p(sub_ops)
                    #     send_ops.append(works_k)
                    num_splits = action.split_parts or 1
                    ops = (
                        stage.get_fwd_send_ops(mb_index, rank=rank, dest_rank=dest_rank, num_splits=num_splits)
                        if rank is not None and dest_rank is not None
                        else stage.get_fwd_send_ops(mb_index, rank=None, dest_rank=None, num_splits=num_splits)
                    )
                    plan = stage._last_comm_plan.get(("SEND_F", mb_index), [len(ops)])

                    # 带宽限速的设置仍在主线程做（如果你这里有 _tc_set_rate）
                    # self._tc_set_rate(action.upstream)  # 若你已有这行，保留；没有可忽略

                    # 起线程，里面按 chunk 等依赖 -> 发送 -> 落 self._async_send_works
                    self._spawn_chunked_send_worker(
                        kind="SEND_F", action=action, ops=ops, plan=plan, chunk_deps=(action.chunk_deps or {}),
                        current_batch=current_batch, stage_idx=stage_idx, mb_index=mb_index
                    )


                    

                elif comp_type == SEND_B:
                    _tc_set_rate(action.upstream)
                    if action.upstream is not None:
                        print(f"[{dist.get_rank()}]: batch {current_batch+1} SEND_B microbatch {mb_index}, upstream bandwidth {action.upstream} mbps")
                    else:
                        print(f"[{dist.get_rank()}]: batch {current_batch+1} SEND_B microbatch {mb_index}")
                    
                    # num_splits = action.split_parts or 1

                    # ops = (
                    #     stage.get_bwd_send_ops(mb_index, rank=rank, dest_rank=dest_rank, num_splits=num_splits)
                    #     if rank is not None and dest_rank is not None
                    #     else stage.get_bwd_send_ops(mb_index, rank=None, dest_rank=None, num_splits=num_splits)
                    # )
                    # plan = stage._last_comm_plan.get(("SEND_B", mb_index), [len(ops)])
                    # pos = 0
                    # for chunk_idx, cnt in enumerate(plan):
                    #     sub_ops = ops[pos:pos+cnt]; pos += cnt

                    #     if action.chunk_deps and chunk_idx in action.chunk_deps:
                    #         for (dep_rank, dep_action_id, dep_chunk) in action.chunk_deps[chunk_idx]:
                    #             _wait_remote_chunk(current_batch+1, dep_rank, dep_action_id, dep_chunk)

                    #     works_k = schedule._batch_p2p(sub_ops)
                    #     send_ops.append(works_k)
                    num_splits = action.split_parts or 1
                    ops = (
                        stage.get_bwd_send_ops(mb_index, rank=rank, dest_rank=dest_rank, num_splits=num_splits)
                        if rank is not None and dest_rank is not None
                        else stage.get_bwd_send_ops(mb_index, rank=None, dest_rank=None, num_splits=num_splits)
                    )
                    plan = stage._last_comm_plan.get(("SEND_B", mb_index), [len(ops)])

                    # self._tc_set_rate(action.upstream)  # 若你在 SEND_B 也有带宽设置，保留

                    self._spawn_chunked_send_worker(
                        kind="SEND_B", action=action, ops=ops, plan=plan, chunk_deps=(action.chunk_deps or {}),
                        current_batch=current_batch, stage_idx=stage_idx, mb_index=mb_index
)




                elif comp_type == RECV_F:
                    # print(f"[{dist.get_rank()}]: batch {current_batch+1} RECV_F microbatch {mb_index}")
                    # assert (stage_idx, mb_index) not in fwd_recv_ops, (
                    #     f"Recv twice for stage_idx={stage_idx} mb_index={mb_index} without executing forward"
                    # )
                    # num_splits = action.split_parts or 1
                    # ops = (
                    #     stage.get_fwd_recv_ops(mb_index, rank=rank, dest_rank=dest_rank, num_splits=num_splits)
                    #     if rank is not None and dest_rank is not None
                    #     else stage.get_fwd_recv_ops(mb_index, rank=None, dest_rank=None, num_splits=num_splits)
                    # )
                    # plan = stage._last_comm_plan.get(("RECV_F", mb_index), [len(ops)])
                    # pos = 0
                    # all_works = []
                    # for chunk_idx, cnt in enumerate(plan):
                    #     sub_ops = ops[pos:pos+cnt]; pos += cnt
                    #     works_k = schedule._batch_p2p(sub_ops)
                    #     # 分块级完成打点（后台轮询 -> _mark_done_chunk）
                    #     start_ns = time.time_ns()
                    #     self._rec.record_async(current_batch+1, action_id, "RECV_F", stage_idx, mb_index, works_k, start_ns, chunk_idx=chunk_idx)
                    #     all_works.extend(works_k)

                    # # 如你仍想保持“RECV_F 立等”语义，可继续等待全部
                    # schedule._wait_batch_p2p(all_works)
                    # fwd_recv_ops[(stage_idx, mb_index)] = []   # 卫生：清空挂账
                    print(f"[{dist.get_rank()}]: batch {current_batch+1} RECV_F microbatch {mb_index}")
                    key = (stage_idx, mb_index)
                    assert key not in self._fwd_recv_posted, (
                        f"Recv twice for stage_idx={stage_idx} mb_index={mb_index} without executing forward"
                    )
                    num_splits = action.split_parts or 1
                    ops = (
                        stage.get_fwd_recv_ops(mb_index, rank=rank, dest_rank=dest_rank, num_splits=num_splits)
                        if rank is not None and dest_rank is not None
                        else stage.get_fwd_recv_ops(mb_index, rank=None, dest_rank=None, num_splits=num_splits)
                    )
                    plan = stage._last_comm_plan.get(("RECV_F", mb_index), [len(ops)])

                    # 建立“全部 irecv 已 post”的 Event，占位 works 容器
                    with self._async_recv_lock:
                        self._fwd_recv_works[key] = []
                        self._fwd_recv_posted[key] = threading.Event()

                    # 启动接收线程
                    self._spawn_chunked_recv_worker(
                        kind="RECV_F", action=action, ops=ops, plan=plan,
                        current_batch=current_batch, stage_idx=stage_idx, mb_index=mb_index, action_id=action_id
                    )



                elif comp_type == RECV_B:
                    # print(f"[{dist.get_rank()}]: batch {current_batch+1} RECV_B microbatch {mb_index}")
                    # assert (stage_idx, mb_index) not in bwd_recv_ops, (
                    #     f"Recv twice for stage_idx={stage_idx} mb_index={mb_index} without executing backward"
                    # )
                    # num_splits = action.split_parts or 1
                    # ops = (
                    #     stage.get_bwd_recv_ops(mb_index, rank=rank, dest_rank=dest_rank, num_splits=num_splits)
                    #     if rank is not None and dest_rank is not None
                    #     else stage.get_bwd_recv_ops(mb_index, rank=None, dest_rank=None, num_splits=num_splits)
                    # )
                    # plan = stage._last_comm_plan.get(("RECV_B", mb_index), [len(ops)])
                    # pos = 0
                    # works_all = []
                    # for chunk_idx, cnt in enumerate(plan):
                    #     sub_ops = ops[pos:pos+cnt]; pos += cnt
                    #     works_k = schedule._batch_p2p(sub_ops)
                    #     start_ns = time.time_ns()
                    #     self._rec.record_async(current_batch+1, action_id, "RECV_B", stage_idx, mb_index, works_k, start_ns, chunk_idx=chunk_idx)
                    #     works_all.extend(works_k)

                    # bwd_recv_ops[(stage_idx, mb_index)] = works_all  # 真正等待仍在 BACKWARD_INPUT 触发前
                    print(f"[{dist.get_rank()}]: batch {current_batch+1} RECV_B microbatch {mb_index}")
                    key = (stage_idx, mb_index)
                    assert key not in self._bwd_recv_posted, (
                        f"Recv twice for stage_idx={stage_idx} mb_index={mb_index} without executing backward"
                    )
                    num_splits = action.split_parts or 1
                    ops = (
                        stage.get_bwd_recv_ops(mb_index, rank=rank, dest_rank=dest_rank, num_splits=num_splits)
                        if rank is not None and dest_rank is not None
                        else stage.get_bwd_recv_ops(mb_index, rank=None, dest_rank=None, num_splits=num_splits)
                    )
                    plan = stage._last_comm_plan.get(("RECV_B", mb_index), [len(ops)])

                    with self._async_recv_lock:
                        self._bwd_recv_works[key] = []
                        self._bwd_recv_posted[key] = threading.Event()

                    self._spawn_chunked_recv_worker(
                        kind="RECV_B", action=action, ops=ops, plan=plan,
                        current_batch=current_batch, stage_idx=stage_idx, mb_index=mb_index, action_id=action_id
                    )



                elif comp_type == UNSHARD:
                    if stage_uses_fsdp:
                        assert (
                            stage_idx not in unsharded_stages
                            and stage_idx not in unshard_ops
                        ), f"Unsharding the same {stage_idx=} twice"
                        unshard_ops[stage_idx] = stage.submod.unshard(async_op=True)  # type: ignore[operator]
                elif comp_type == RESHARD:
                    if stage_uses_fsdp:
                        assert stage_idx in unsharded_stages, (
                            f"Resharding {stage_idx=} without unsharding"
                        )
                        assert stage_idx not in unshard_ops, (
                            f"Resharding {stage_idx=} before finishing unshard"
                        )
                        stage.submod.reshard()  # type: ignore[operator]
                
                elif comp_type == FORWARD:
                    print(f"[{dist.get_rank()}]: batch {current_batch+1} FORWARD microbatch {mb_field}")
                    
                    mb_ids: tuple[int, ...] = (mb_field,) if isinstance(mb_field, int) else tuple(mb_field)
                    rep_id = mb_ids[0]

                    
                    if stage_uses_fsdp:
                        _assert_unsharded(stage_idx)
                    
                    if stage.is_first:
                        if len(mb_ids) > 1:
                            for mid in mb_ids:
                                if mid not in arg_mbs and rep_id in arg_mbs:
                                    arg_mbs[mid] = arg_mbs[rep_id]
                                if kwarg_mbs and mid not in kwarg_mbs and rep_id in kwarg_mbs:
                                    kwarg_mbs[mid] = kwarg_mbs[rep_id]
                        
                        cat_args = tuple(
                            torch.cat([arg_mbs[mid][i] for mid in mb_ids], dim=0)
                            for i in range(len(arg_mbs[rep_id]))
                        )
                        
                        if kwarg_mbs and kwarg_mbs[rep_id]:
                            cat_kwargs: dict[str, Any] = {}
                            for k in kwarg_mbs[rep_id]:
                                vals = [kwarg_mbs[mid][k] for mid in mb_ids]
                                cat_kwargs[k] = _cat_like(vals)

                        else:
                            cat_kwargs = {}
                            
                    else:
                        if not is_prev_stage_on_this_rank:
                            for mid in mb_ids:
                                key = (stage_idx, mid)
                                # 等待“所有 irecv 已 post”
                                assert key in self._fwd_recv_posted, f"Computing {action=} before RECV_F posted"
                                self._fwd_recv_posted[key].wait()
                                # 再等待“全部 works 完成”
                                with self._async_recv_lock:
                                    works = self._fwd_recv_works.pop(key, [])
                                print(f"[{dist.get_rank()}] FORWARD wait st{stage_idx} mb{mid} "
                                        f"posted={self._fwd_recv_posted[key].is_set()} "
                                        f"works={len(self._fwd_recv_works.get(key, []))}")
                                schedule._wait_batch_p2p(works)
                                

                                # 清理 event
                                self._fwd_recv_posted.pop(key, None)

                        
                        if len(mb_ids) == 1:
                            composite_args = stage._retrieve_recv_activations(mb_ids[0])
                            cat_args = composite_args
                        else:
                            all_activations = []
                            for mid in mb_ids:
                                act = stage._retrieve_recv_activations(mid)
                                all_activations.append(act)
                            
                            if all_activations and all_activations[0]:
                                num_args = len(all_activations[0])
                                cat_args = tuple(
                                    torch.cat([all_activations[i][j] for i in range(len(mb_ids))], dim=0)
                                    for j in range(num_args)
                                )
                            else:
                                cat_args = ()
                        
                        cat_kwargs = {}
                    
                    
                    if len(mb_ids) > 1:
                        g_id = (stage_idx, rep_id)
                        self._pack_groups[g_id] = len(mb_ids)
                        for mid in mb_ids:
                            self._mb_to_group[(stage_idx, mid)] = g_id
                            
                    
                    with self._rec.record(current_batch+1,action_id,"FORWARD", stage_idx, mb_ids):
                        output = stage.forward_one_chunk(rep_id, cat_args, cat_kwargs, len(mb_ids))
                    
                    big_key = (stage_idx, rep_id)
                    big_entry = stage.fwd_cache.get(rep_id)
                    assert big_entry is not None, f"missing big forward entry for stage {stage_idx}, mb {rep_id}"
                    self._big_fwd_cache[big_key] = big_entry

                    
                    # Split output
                    if isinstance(output, tuple):
                        split_out = list(zip(*[
                            torch.chunk(t, len(mb_ids), dim=0) if isinstance(t, torch.Tensor)
                            else (t,) * len(mb_ids)
                            for t in output
                        ]))
                    else:
                        split_out = list(torch.chunk(output, len(mb_ids), dim=0))
                    
                    flat_args = flatten_args(cat_args)
                    flat_kwargs = flatten_args(cat_kwargs)
                    
                    for idx, mid in enumerate(mb_ids):
                        if len(mb_ids) > 1:
                            mb_inputs = []
                            for i, arg in enumerate(cat_args):
                                split_args = torch.chunk(arg, len(mb_ids), dim=0)
                                mb_inputs.append(split_args[idx])
                            
                            if cat_kwargs:
                                for k, v in cat_kwargs.items():
                                    if isinstance(v, torch.Tensor):
                                        split_v = torch.chunk(v, len(mb_ids), dim=0)
                                        mb_inputs.append(split_v[idx])
                                    else:
                                        mb_inputs.append(v)
                        else:
                            mb_inputs = flat_args + flat_kwargs
                        
                        stage.fwd_cache[mid] = (
                            _normalize_model_output_as_tuple(split_out[idx]),
                            mb_inputs
                        )
                    
                    if stage.is_first:
                        split_args_per_dim = [
                            torch.chunk(a, len(mb_ids), dim=0) for a in cat_args
                        ]
                        for idx, mid in enumerate(mb_ids):
                            old_shape = [a.shape for a in arg_mbs[mid]] if mid in arg_mbs else []
                            arg_mbs[mid] = tuple(split_args_per_dim[d][idx] for d in range(len(cat_args)))
                            new_shape = [a.shape for a in arg_mbs[mid]]
                        
                        if kwarg_mbs and kwarg_mbs[rep_id]:
                            for idx, mid in enumerate(mb_ids):
                                kwarg_mbs[mid] = {
                                    k: (torch.chunk(cat_kwargs[k], len(mb_ids), dim=0)[idx]
                                    if isinstance(cat_kwargs[k], torch.Tensor) else cat_kwargs[k])
                                    for k in cat_kwargs
                                }
                    
                    for idx, mid in enumerate(mb_ids):
                        self._maybe_compute_loss(stage, split_out[idx], target_mbs, mid)
                    
                    if is_next_stage_on_this_rank:
                        for idx, mid in enumerate(mb_ids):
                            stage_index_to_stage[stage_idx + 1].set_local_fwd_input(
                                split_out[idx], mid
                            )
                
                    for mid in mb_ids:
                        if mid in stage.fwd_cache:
                            outputs, inputs = stage.fwd_cache[mid]

                elif comp_type == FULL_BACKWARD:
                    mb_ids = (mb_field,) if isinstance(mb_field, int) else tuple(mb_field)
                    rep_id = mb_ids[0]
                    print(f"[{dist.get_rank()}]: batch {current_batch+1} FULL_BACKWARD microbatch {mb_ids}")

                    if stage_uses_fsdp:
                        _assert_unsharded(stage_idx)

                    if not stage.is_last and not is_next_stage_on_this_rank:
                        for mid in mb_ids:
                            key = (stage_idx, mid)
                            if key in self._bwd_recv_posted:
                                self._bwd_recv_posted[key].wait()
                                with self._async_recv_lock:
                                    works = self._bwd_recv_works.pop(key, [])
                                schedule._wait_batch_p2p(works)
                                self._bwd_recv_posted.pop(key, None)


                    for mid in mb_ids:
                        stage.fwd_cache.pop(mid, None)

                    big_key   = (stage_idx, rep_id)
                    big_entry = self._big_fwd_cache.pop(big_key, None)
                    if big_entry is None:
                        big_entry = stage.fwd_cache.get(rep_id)
                    assert big_entry is not None, f"big forward entry missing for {big_key}"

                    big_out_tuple, big_flat_inputs = big_entry

                    if stage.is_last:
                        loss_lst = [ self._maybe_get_loss(stage, mid) for mid in mb_ids ]
                        cat_loss = _cat_like(loss_lst).mean()
                        cat_gradout = None
                    else:
                        grad_out_lst = [ stage._retrieve_recv_grads(mid) for mid in mb_ids ]
                        cat_loss = None
                        cat_gradout = _cat_like(grad_out_lst)

                    stage.fwd_cache[rep_id] = (big_out_tuple, big_flat_inputs)

                    backward_counter[stage_idx] += 1
                    self.last_backward = backward_counter[stage_idx] == local_bwd_budget[stage_idx]
                    if not stage.is_last:
                        stage.set_local_bwd_input(cat_gradout, rep_id)

                    with self._rec.record(current_batch+1, action_id, "FULL_BACKWARD", stage_idx, mb_ids):
                        stage.backward_one_chunk(
                            rep_id,
                            loss=cat_loss,
                            full_backward=True,
                            last_backward=False,
                            retain_graph_for_packed_mbs=False,
                        )

                    big_grads_in = stage.bwd_cache.pop(rep_id)
                    big_grads_in = _fill_grads_like(big_flat_inputs, big_grads_in) 

                    split_grads = _chunk_like(big_grads_in, len(mb_ids))
                    for mid, g in zip(mb_ids, split_grads):
                        stage.bwd_cache[mid] = g

                    if is_prev_stage_on_this_rank:
                        for mid in mb_ids:
                            stage_index_to_stage[stage_idx - 1].set_local_bwd_input(
                                stage.get_local_bwd_output(mid), mid
                            )

                    if self.last_backward:
                        grad_scale_factor = backward_counter.total() if self.scale_grads else 1
                        stage.scale_grads(grad_scale_factor)
                
                elif comp_type == BACKWARD_INPUT:
                    if stage_uses_fsdp:
                        _assert_unsharded(stage_idx)

                    if not stage.is_last and not is_next_stage_on_this_rank:
                        assert (
                            stage_idx,
                            mb_index,
                        ) in bwd_recv_ops, (
                            f"Attempted to run compute {action=} before receiving input"
                        )
                        schedule._wait_batch_p2p(bwd_recv_ops.pop((stage_idx, mb_index)))
                    loss = self._maybe_get_loss(stage, mb_index)
                    with self._rec.record(current_batch+1,action_id,"BACKWARD_INPUT", stage_idx, mb_index):
                        stage.backward_one_chunk(
                            mb_index,
                            loss=loss,
                            full_backward=False,
                            last_backward=False,
                        )

                    if is_prev_stage_on_this_rank:
                        stage_index_to_stage[stage_idx - 1].set_local_bwd_input(
                            stage.get_local_bwd_output(mb_index), mb_index
                        )
                elif comp_type == BACKWARD_WEIGHT:
                    if stage_uses_fsdp:
                        _assert_unsharded(stage_idx)
                    backward_counter[stage_idx] += 1
                    
                    with self._rec.record(current_batch+1,action_id,"BACKWARD_WEIGHT", stage_idx, mb_index):
                        stage.backward_weight_one_chunk(
                            mb_index,
                            last_backward=backward_counter[stage_idx]
                            == self._n_microbatches,
                        )
                elif comp_type == _ComputationType.ALL_REDUCE:
                    _tc_set_rate(action.upstream)
                    if action.upstream is not None:
                        print(f"[{dist.get_rank()}]: batch {current_batch+1} ALL_REDUCE for stage {stage_idx}, upstream bandwidth {action.upstream} mbps")
                    else:
                        print(f"[{dist.get_rank()}]: batch {current_batch+1} ALL_REDUCE for stage {stage_idx}")
                    with self._rec.record(current_batch+1,action_id,"ALL_REDUCE", stage_idx, -1):  # -1 not specific microbatch
                        stage._execute_allreduce()
                                
                else:
                    raise ValueError(f"{action=} is unknown or unsupported")
            except Exception as e:
                logger.error(
                    "_PipelineScheduleRuntime caught exception at step %s when running action %s.  Full Schedule:",
                    time_step,
                    action,
                )
              
                print(
                    schedule._format_pipeline_order(
                        self.pipeline_order_with_comms,  # type: ignore[arg-type]
                        error_step_number=time_step,
                    )
                )
                raise e
       
        while len(send_ops):
            schedule._wait_batch_p2p(send_ops.pop())
        
        # 也回收异步 SEND 的 works
        with self._async_send_lock:
            async_list = self._async_send_works.pop(current_batch+1, [])
        for works_k in async_list:
            schedule._wait_batch_p2p(works_k)

        # —— 保险：回收任何未消费的 RECV works，并清理 posted events ——
        with self._async_recv_lock:
            # FWD
            for key, ev in list(self._fwd_recv_posted.items()):
                ev.wait()
                works = self._fwd_recv_works.pop(key, [])
                schedule._wait_batch_p2p(works)
                self._fwd_recv_posted.pop(key, None)
            # BWD
            for key, ev in list(self._bwd_recv_posted.items()):
                ev.wait()
                works = self._bwd_recv_works.pop(key, [])
                schedule._wait_batch_p2p(works)
                self._bwd_recv_posted.pop(key, None)

        
        assert len(unshard_ops) == 0, "Unused unshard operations"

                
        # === Compute global average loss for this training step ===
        local_sum = torch.tensor(0.0)
        local_cnt = torch.tensor(0, dtype=torch.long)

        # 只有“包含最后一段”的 rank 才会在本地累到 loss
        if any(stage.is_last for stage in self._stages) and len(self._internal_losses) > 0:
            # _internal_losses: {mb_index -> loss_tensor}
            vals = [v if torch.is_tensor(v) else torch.tensor(float(v)) for v in self._internal_losses.values()]
            local_sum = torch.stack(vals).sum().to(torch.float32)
            local_cnt = torch.tensor(len(vals), dtype=torch.long)

        # 在 WORLD 维度做求和，再统一求均值
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_cnt, op=dist.ReduceOp.SUM)

        self.last_step_loss = (local_sum / local_cnt).item() if local_cnt.item() > 0 else None
        # === End of loss aggregation ===

        
        
        
        
        self._update_losses(self._stages, losses)
        
        if (
            record_this_batch
            # and self._profile_batch is not None
            # and not self._timeline_saved
        ):
            fname_prefix = f"timeline_batch{current_batch + 1}"
            self.save_timeline(fname_prefix)
            # self._timeline_saved = True
            # self._rec.set_enabled(False)   # 后续 minibatch 不再计时
        
        # clear cache
        stage.grad_recv_info = copy.deepcopy(self.grad_recv_info_copy)
        self._clear_communication_state()
        
    def _clear_communication_state(self):
        self._internal_losses.clear()
        self._pack_groups.clear()
        self._mb_to_group.clear()
        self._big_fwd_cache.clear() 

        for stage in self._stages:
            stage.fwd_cache.clear()
            stage.bwd_cache.clear()
            if hasattr(stage, "grad_send_info"):
                stage.grad_send_info = None
            if hasattr(stage, "_local_grads"):
                stage._local_grads.clear()
            if hasattr(stage, "_local_acts"):
                stage._local_acts.clear()




    def save_timeline(self, fname_prefix="timeline"):
        if dist.is_initialized():
            if dist.get_world_size() == 1:
                self._rec.dump(f"{fname_prefix}_rank0.json")
            else:
                gathered = [None] * dist.get_world_size()
                dist.gather_object(self._rec.events, gathered if self.rank == 0 else None, dst=0)
                if self.rank == 0:
                    all_ev = sum(gathered, [])
                    with open(f"{fname_prefix}_all.json", "w") as f:
                        json.dump([asdict(e) for e in all_ev], f, indent=2)
        else:
            self._rec.dump(f"{fname_prefix}_solo.json")
        self._rec.events.clear()