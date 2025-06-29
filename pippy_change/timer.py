# timeline.py
import time, json, os, torch.distributed as dist
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class TraceEvent:
    rank:        int
    action:      str          
    stage_idx:   int
    mb_idx:      int
    start_ns:    int
    end_ns:      int

class TimelineRecorder:
    """
    线程安全足够，perf_counter_ns 分辨率纳秒，适合 CPU / NCCL.
    """
    def __init__(self, rank: int):
        self.rank    = rank
        self.events: list[TraceEvent] = []
        self.enabled = True     

    def set_enabled(self, flag: bool):
        """打开 / 关闭计时；关闭时 record() 直接透传"""
        self.enabled = flag

    @contextmanager
    def record(self, action: str, stage_idx: int, mb_idx: int):
        
        if not self.enabled:
            yield
            return
        
        st = time.time_ns()
        yield      # 真正执行目标代码
        ed = time.time_ns()        
        self.events.append(
            TraceEvent(self.rank, action, stage_idx, mb_idx, st, ed)
        )

    def dump(self, fname: str = None):
        if fname is None:
            fname = f"timeline_rank{self.rank}.json"
        with open(fname, "w") as f:
            json.dump([asdict(e) for e in self.events], f, indent=2)
