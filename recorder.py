import time, json, threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Iterable, Optional
import psutil          
import torch.distributed as dist
from schedule_runtime import _mark_done, _mark_done_chunk

@dataclass
@dataclass
class TraceEvent:
    batch_id:  int
    rank:       int
    action:     str
    stage_idx:  int
    mb_idx:     int
    start_ns:   int
    end_ns:     int
    net_series: List[Tuple[int, float, float]] = field(default_factory=list)
    
    # 新增：分块编号（一级命令无分块时为 None）
    chunk:      Optional[int] = None
    # 新增：执行状态（completed / error:...）
    status:     str = "completed"
    # (ts_ns, up_mbps, down_mbps)


import os, sys, traceback, threading, time
from datetime import datetime

def _now_ns():
    return time.time_ns()

def _fmt_ts(ns: int):
    # 人类可读 + ns
    return f"{datetime.fromtimestamp(ns/1e9).strftime('%H:%M:%S.%f')[:-3]} ({ns})"

def _dist_backend_info():
    try:
        if dist.is_available() and dist.is_initialized():
            # PyTorch 2.x: str(dist.get_backend()) -> 'gloo' / 'nccl' / 'mpi'
            be = str(dist.get_backend())
            ws = dist.get_world_size()
            rk = dist.get_rank()
            return {"backend": be, "world_size": ws, "rank": rk}
    except Exception as e:
        return {"backend": f"<err: {e}>", "world_size": -1, "rank": -1}
    return {"backend": "<uninit>", "world_size": -1, "rank": -1}

def _gloo_env_info():
    # 常见的 Gloo 相关环境变量（有就打印）
    keys = [
        "TORCH_DISTRIBUTED_DEBUG",   # DETAIL/INFO 有助于底层日志
        "GLOO_SOCKET_IFNAME",        # 指定网卡
        "GLOO_DEVICE_TRANSPORT",     # tcp / ib
        "MASTER_ADDR", "MASTER_PORT",
        "WORLD_SIZE", "RANK",
    ]
    kv = {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}
    return kv

def _summarize_works(works):
    # Work 对象基本是黑盒；repr 通常包含 id。这里给出长度+repr 截断，避免过多 IO
    try:
        items = []
        for i, w in enumerate(works):
            s = repr(w)
            if len(s) > 120:
                s = s[:117] + "..."
            items.append(f"[{i}] {s}")
        return items
    except Exception as e:
        return [f"<summarize_works error: {e}>"]

def _dbg_print(*args, **kwargs):
    # 仅当打开 RECORDER_DEBUG=1 才打印，避免污染正常运行
    if os.environ.get("RECORDER_DEBUG", "0") == "1":
        print(*args, **kwargs, flush=True)



class Recorder:
    _NET_ACTIONS = {"SEND_F", "RECV_F", "SEND_B", "RECV_B", "ALL_REDUCE"}

    def __init__(
        self,
        rank: int,
        net_sample_interval_ms: int = 10,
        net_actions: Optional[Iterable[str]] = None,
        measure_net: bool = True,
    ):
        self.rank = rank
        self.events: List[TraceEvent] = []
        self.enabled = True
        self.measure_net = measure_net
        self.sample_interval = max(net_sample_interval_ms, 1) / 1000.0   # 秒
        self.net_actions = set(net_actions) if net_actions else self._NET_ACTIONS

    def set_enabled(self, flag: bool):
        self.enabled = flag

    @contextmanager
    def record(self, batch_id: int, action_id: int,action: str, stage_idx: int, mb_idx: int):
        if not self.enabled:
            yield
            return

        need_net = self.measure_net and action in self.net_actions
        samples: List[Tuple[int, float, float]] = []
        stop_evt = threading.Event()

        if need_net:
            # 初始累计值
            prev_ts = time.time_ns()
            prev_io = psutil.net_io_counters()
            prev_sent, prev_recv = prev_io.bytes_sent, prev_io.bytes_recv

            def sampler():
                nonlocal prev_ts, prev_sent, prev_recv
                while not stop_evt.is_set():
                    time.sleep(self.sample_interval)
                    curr_ts = time.time_ns()
                    io = psutil.net_io_counters()
                    dt = (curr_ts - prev_ts) / 1e9 or 1e-9
                    up_mbps   = (io.bytes_sent - prev_sent) * 8 / dt / 1e6
                    down_mbps = (io.bytes_recv - prev_recv) * 8 / dt / 1e6
                    samples.append((curr_ts, up_mbps, down_mbps))
                    prev_ts, prev_sent, prev_recv = curr_ts, io.bytes_sent, io.bytes_recv

            thr = threading.Thread(target=sampler, daemon=True)
            thr.start()

        start_ns = time.time_ns()
        try:
            yield
        finally:
            end_ns = time.time_ns()
            if need_net:
                stop_evt.set()
                thr.join()

            # print(f"{action} {mb_idx}计时结束，{action_id}添加表")
            _mark_done(batch_id=batch_id,action_id=action_id)
            
            self.events.append(
                TraceEvent(
                    batch_id, self.rank, action, stage_idx, mb_idx,
                    start_ns, end_ns,
                    chunk=None,
                    status="completed",
                    net_series=samples,
                )
            )

    ...
    def record_async(
        self,
        batch_id: int,
        action_id: int, 
        action: str,
        stage_idx: int,
        mb_idx: int,
        works: List[dist.Work],
        start_ns: int,
        poll_interval: float = 0.001,
        chunk_idx: Optional[int] = None,
    ):
        if not self.enabled:
            return
        
        need_net = self.measure_net and action in self.net_actions
        samples, stop_evt = [], threading.Event()

        if action in ("SEND_F", "SEND_B"):

            self.events.append(
                TraceEvent(
                    batch_id, self.rank, action, stage_idx, mb_idx,
                    start_ns, start_ns,  # Use start time as end time for now
                    chunk=chunk_idx,
                    status="posted",  
                    net_series=[],
                )
            )
            return  # Exit early
        
        if need_net:
            def sampler():
                prev_ts = time.time_ns()
                io = psutil.net_io_counters()
                prev_sent, prev_recv = io.bytes_sent, io.bytes_recv
                while not stop_evt.is_set():
                    time.sleep(self.sample_interval)
                    curr_ts = time.time_ns()
                    io = psutil.net_io_counters()
                    dt = (curr_ts - prev_ts) / 1e9 or 1e-9
                    up   = (io.bytes_sent - prev_sent) * 8 / dt / 1e6
                    down = (io.bytes_recv - prev_recv) * 8 / dt / 1e6
                    samples.append((curr_ts, up, down))
                    prev_ts, prev_sent, prev_recv = curr_ts, io.bytes_sent, io.bytes_recv
            threading.Thread(target=sampler, daemon=True).start()

        # def waiter():
        #     status = "completed"
        #     try:
        #         # print(f"[{dist.get_rank()}] Waiting for {len(works)} works (action={action_id}, chunk={chunk_idx})")
        #         while not all(w.is_completed() for w in works):
        #             time.sleep(poll_interval)
        #         # for w in works:
        #         #     w.wait()
                    
        #         end_ns = time.time_ns()
        #         if need_net:
        #             stop_evt.set()
                
        #         # Mark done after waiting completes
        #         if chunk_idx is None:
        #             _mark_done(batch_id=batch_id, action_id=action_id)
        #         else:
        #             # print(f"[{dist.get_rank()}] Marking done chunk for batch={batch_id}, action={action_id}, chunk={chunk_idx}")
        #             _mark_done_chunk(batch_id=batch_id, action_id=action_id, chunk_idx=chunk_idx)
        #             # print(f"[{self.rank}] DONE {action} st={stage_idx} mb={mb_idx} chunk={chunk_idx} "
        #             #     f"works={len(works)}")
        #     except Exception as e:
        #         status = f"error:{type(e).__name__}"
        #         end_ns = time.time_ns()
        #     finally:
        #         self.events.append(
        #             TraceEvent(
        #                 batch_id, self.rank, action, stage_idx, mb_idx,
        #                 start_ns, end_ns,
        #                 chunk=chunk_idx,
        #                 status=status,
        #                 net_series=samples,
        #             )
        #         )

        def waiter():
            # --- 基础上下文（无锁） ---
            status = "completed"
            be_info = _dist_backend_info()      # {'backend': 'gloo'|'nccl'|..., 'world_size':..., 'rank':...}
            gloo_env = _gloo_env_info()         # MASTER_ADDR/PORT、GLOO_SOCKET_IFNAME 等
            th_name = threading.current_thread().name
            rk = be_info.get("rank", -1)
            t_enter = _now_ns()

            _dbg_print(
                f"[recorder/waiter][rank={rk}][thr={th_name}] ENTER  "
                f"action={action} stage={stage_idx} mb={mb_idx} chunk={chunk_idx} "
                f"backend={be_info.get('backend')} world_size={be_info.get('world_size')} "
                f"t0={_fmt_ts(t_enter)}"
            )
            _dbg_print(f"[recorder/waiter][rank={rk}] GLOO_ENV: {gloo_env}")
            _dbg_print(f"[recorder/waiter][rank={rk}] works({len(works)}):")
            for line in _summarize_works(works):
                _dbg_print(f"    {line}")

            try:
                # ===== 只用 wait()，完全去掉轮询 =====
                # while not all(w.is_completed() for w in works):
                #     time.sleep(poll_interval)

                for i, w in enumerate(works):
                    t1 = _now_ns()
                    _dbg_print(f"[recorder/waiter][rank={rk}] wait begin for work[{i}] at { _fmt_ts(t1) }")
                    w.wait()  # PyTorch 这一步会释放 GIL，只阻塞当前线程
                    t2 = _now_ns()
                    _dbg_print(f"[recorder/waiter][rank={rk}] wait done  for work[{i}] at { _fmt_ts(t2) }, dt={(t2 - t1)/1e6:.3f} ms")

                status = "completed"

            except Exception as e:
                status = f"error: {type(e).__name__}: {e}"
                _dbg_print(
                    f"[recorder/waiter][rank={rk}] EXCEPTION during wait: {status}\n"
                    + "".join(traceback.format_exc())
                )
                raise
            finally:
                t_exit = _now_ns()
                _dbg_print(
                    f"[recorder/waiter][rank={rk}][thr={th_name}] EXIT   "
                    f"action={action} stage={stage_idx} mb={mb_idx} chunk={chunk_idx} "
                    f"status={status} t1={_fmt_ts(t_exit)} dt={(t_exit - t_enter)/1e6:.3f} ms"
                )

        
        threading.Thread(target=waiter, daemon=True).start()


    
    def dump(self, fname: str = None):
        fname = fname or f"timeline_rank{self.rank}.json"
        with open(fname, "w") as f:
            json.dump([asdict(e) for e in self.events], f, indent=2)
