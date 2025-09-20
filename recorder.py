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
            status = "completed"
            end_ns = None

            try:
                # 并发等待每个 Work：为每个 w 启一个监听线程，全部 join 后再继续
                n = len(works)
                done_flags = [False] * n
                errors = [None] * n
                threads = []

                def _wait_one(i: int, w):
                    try:
                        w.wait()              # 只阻塞该监听线程，会释放 GIL
                        done_flags[i] = True
                    except Exception as e:
                        errors[i] = e

                for i, w in enumerate(works):
                    t = threading.Thread(target=_wait_one, args=(i, w), daemon=True, name=f"waiter-work-{i}")
                    t.start()
                    threads.append(t)

                # 等所有监听线程结束
                for t in threads:
                    t.join()

                # 记录结束时间（在 mark 前，便于统计）
                end_ns = time.time_ns()

                # 停止网速采样（如启用）
                if need_net:
                    stop_evt.set()

                # 如果有任一监听线程出错，抛出第一个异常
                first_err = next((e for e in errors if e is not None), None)
                if first_err is not None:
                    raise first_err

                # 全部完成后再标记 done
                if chunk_idx is None:
                    _mark_done(batch_id=batch_id, action_id=action_id)
                else:
                    _mark_done_chunk(batch_id=batch_id, action_id=action_id, chunk_idx=chunk_idx)

            except Exception as e:
                status = f"error:{type(e).__name__}:{e}"
                if end_ns is None:
                    end_ns = time.time_ns()

            finally:
                # 落事件（保持你原有的 TraceEvent 字段顺序）
                self.events.append(
                    TraceEvent(
                        batch_id, self.rank, action, stage_idx, mb_idx,
                        start_ns, end_ns,
                        chunk=chunk_idx,
                        status=status,
                        net_series=samples,
                    )
                )


        
        threading.Thread(target=waiter, daemon=True).start()


    
    def dump(self, fname: str = None):
        fname = fname or f"timeline_rank{self.rank}.json"
        with open(fname, "w") as f:
            json.dump([asdict(e) for e in self.events], f, indent=2)
