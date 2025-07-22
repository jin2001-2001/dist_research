import time, json, threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Iterable, Optional
import psutil          

@dataclass
class TraceEvent:
    rank:       int
    action:     str
    stage_idx:  int
    mb_idx:     int
    start_ns:   int
    end_ns:     int
    # (ts_ns, up_mbps, down_mbps)
    net_series: List[Tuple[int, float, float]] = field(default_factory=list)

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
    def record(self, action: str, stage_idx: int, mb_idx: int):
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

            self.events.append(
                TraceEvent(
                    self.rank, action, stage_idx, mb_idx,
                    start_ns, end_ns, samples
                )
            )

    def dump(self, fname: str = None):
        fname = fname or f"timeline_rank{self.rank}.json"
        with open(fname, "w") as f:
            json.dump([asdict(e) for e in self.events], f, indent=2)
