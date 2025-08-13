# Demo: Using Algorithm 1 (micro-batch allocation within a stage) 
# and Algorithm 2 (DP-based HPP planning) together on a toy model.
#
# What you'll see:
# 1) Definitions of both algorithms (clean, runnable).
# 2) A small synthetic model + device/bandwidth profiles.
# 3) Run Algorithm 2 -> which internally calls Algorithm 1 to build each stage.
# 4) Two tables:
#    - Stage summary (layers, devices, Algorithm-1 allocations Ef/Eb/Ta)
#    - Per-step timing breakdown (Tw, Te, Ta, totals) and round latency.
#
# No external internet access is used.

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
import pandas as pd
# from ace_tools import display_dataframe_to_user


# ----------------------------- Algorithm 1 -----------------------------

@dataclass(frozen=True)
class Device:
    name: str
    memory_budget: int   # bytes or arbitrary units
    capacity: float      # higher is faster (used for proportional allocation)


def allocate_microbatch_samples(
    devices: List[Device],
    micro_batch_size: int,
    memory_of: Callable[[Device, int], int],
    latency_of: Callable[[Device, int], float],
    block_size: int = 1,
) -> Dict[str, int]:
    """
    Allocate a micro-batch's samples across a stage's DP group (Algorithm 1).

    Parameters
    ----------
    devices : List[Device]
        Devices in the *same stage's DP group*. Each provides a memory budget
        and a relative capacity value (used for proportional allocation).
    micro_batch_size : int
        B, total samples to allocate among the devices.
    memory_of : Callable[[Device, int], int]
        Peak memory function for the stage on device 'd' given y samples.
        Must be monotonic non-decreasing in y.
    latency_of : Callable[[Device, int], float]
        Stage-level FP+BP latency on device 'd' for y samples.
    block_size : int
        Unit to move per offloading step when relieving the straggler.

    Returns
    -------
    Dict[str, int]
        Mapping device_name -> allocated samples y_d, summing to B.
    """
    def max_batch_size_under_budget(d: Device) -> int:
        lo, hi = 0, micro_batch_size
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if memory_of(d, mid) <= d.memory_budget:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # Feasibility check
    if sum(max_batch_size_under_budget(d) for d in devices) < micro_batch_size:
        print(f"✅{sum(max_batch_size_under_budget(d) for d in devices)}           {micro_batch_size}")
        raise ValueError("Micro-batch cannot fit into the group's memory budgets.")

    y: Dict[str, int] = {d.name: 0 for d in devices}
    dev_by_name = {d.name: d for d in devices}

    # ---- Phase 1: Memory-Aware Balancing (proportional to capacity) ----
    def memory_aware_balancing(group: List[Device], beta: int) -> None:
        if beta == 0:
            return
        # Compute headroom and total capacity among devices that can still take >0
        headroom: Dict[str, int] = {}
        caps: Dict[str, float] = {}
        total_cap = 0.0
        for d in group:
            avail = max_batch_size_under_budget(d) - y[d.name]
            if avail > 0:
                headroom[d.name] = avail
                caps[d.name] = d.capacity
                total_cap += d.capacity

        if total_cap == 0.0:
            raise ValueError("No remaining memory headroom for allocation.")

        # Floor of proportional shares
        allocated = 0
        remainders: List[Tuple[str, float]] = []
        for d in group:
            if d.name not in headroom:
                continue
            ideal = beta * (caps[d.name] / total_cap)
            take = min(int(ideal), headroom[d.name])
            y[d.name] += take
            allocated += take
            remainders.append((d.name, ideal - int(ideal)))

        # Distribute leftovers by largest fractional remainder
        leftover = beta - allocated
        if leftover > 0:
            for name, _rem in sorted(remainders, key=lambda x: x[1], reverse=True):
                if leftover == 0:
                    break
                avail = max_batch_size_under_budget(dev_by_name[name]) - y[name]
                if avail > 0:
                    y[name] += 1
                    leftover -= 1

        # Recurse if still leftover
        if leftover > 0:
            next_group = [d for d in group if (max_batch_size_under_budget(d) - y[d.name]) > 0]
            memory_aware_balancing(next_group, leftover)

    memory_aware_balancing(devices, micro_batch_size)

    # ---- Phase 2: Straggler Workload Offloading ----
    def straggler_offload() -> None:
        def dlat(d: Device) -> float:
            return latency_of(d, y[d.name])

        group_sorted = sorted(devices, key=dlat)  # fastest first
        if not group_sorted:
            return

        while True:
            old_straggler = max(group_sorted, key=dlat)
            old_lat = dlat(old_straggler)

            # pick fastest recipient with headroom
            recipient = None
            for cand in group_sorted:
                if cand.name == old_straggler.name:
                    continue
                if y[cand.name] + block_size <= max_batch_size_under_budget(cand):
                    recipient = cand
                    break

            if recipient is None or y[old_straggler.name] <= 0:
                break

            move = min(block_size, y[old_straggler.name])
            y[old_straggler.name] -= move
            y[recipient.name] += move

            group_sorted = sorted(devices, key=dlat)
            new_straggler = max(group_sorted, key=dlat)
            new_lat = dlat(new_straggler)
            if new_lat >= old_lat:
                # revert if not improved
                y[old_straggler.name] += move
                y[recipient.name] -= move
                break

    straggler_offload()

    assert sum(y.values()) == micro_batch_size, "Allocation sum mismatch."
    return y


# ----------------------------- Algorithm 2 -----------------------------

@dataclass(frozen=True)
class StageSpec:
    layer_start: int           # inclusive
    layer_end: int             # exclusive
    device_names: List[str]
    y_allocation: Dict[str, int]
    Ef: float                  # forward time per step
    Eb: float                  # backward time per step
    Ta: float                  # AllReduce time per step


@dataclass(frozen=True)
class PipelinePlan:
    stages: List[StageSpec]
    round_latency: float
    meta: dict


def plan_hpp_dynamic(
    *,
    num_layers: int,
    weights_size: List[int],
    activation_size: List[int],
    devices: List[Device],
    bandwidth_matrix: Dict[Tuple[str, str], float],  # bytes/s
    micro_batch_size: int,
    num_micro_batches: int,
    latency_of_layer: Callable[[str, int, int], Tuple[float, float]],
    # returns (t_f, t_b) seconds for a single layer on a device at batch size
    memory_of_stage: Callable[[str, Tuple[int, int], int], int],
    allocate_fn: Callable[
        [List[Device], int, Callable[[Device, int], int], Callable[[Device, int], float], int],
        Dict[str, int]
    ],
    block_size: int = 1,
    max_stages: Optional[int] = None,
) -> PipelinePlan:
    """
    Dynamic Programming HPP Planning (Algorithm 2).

    Parameters (key ones)
    ---------------------
    num_layers : total layer count L.
    weights_size : per-layer parameter sizes (bytes).
    activation_size : per-layer activation sizes (bytes).
    devices : all candidate devices (will be sorted by memory desc).
    bandwidth_matrix : pairwise bandwidths in bytes/s.
    micro_batch_size : B.
    num_micro_batches : M.
    latency_of_layer : (dev_name, layer_idx, batch) -> (tf, tb) seconds.
    memory_of_stage : (dev_name, (start,end), batch) -> peak memory.
    allocate_fn : Algorithm 1 function (intra-stage allocation).
    block_size : Algorithm 1 offloading granularity.
    max_stages : optional cap on number of stages.

    Returns
    -------
    PipelinePlan with stages and timing breakdown.
    """

    def bw(u: str, v: str) -> float:
        b1 = bandwidth_matrix.get((u, v))
        b2 = bandwidth_matrix.get((v, u))
        candidates = [b for b in (b1, b2) if b is not None]
        if not candidates:
            raise KeyError(f"No bandwidth between {u} and {v}")
        return min(candidates)

    def min_group_bw(group: List[str]) -> float:
        if len(group) <= 1:
            return float("inf")
        mn = float("inf")
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                mn = min(mn, bw(group[i], group[j]))
        return mn

    def between_groups_bw(g1: List[str], g2: List[str]) -> float:
        mn = float("inf")
        for u in g1:
            for v in g2:
                mn = min(mn, bw(u, v))
        return mn

    def stage_latency_of_device(dev: Device, layer_span: Tuple[int, int], batch: int) -> float:
        s, e = layer_span
        total = 0.0
        for l in range(s, e):
            tf, tb = latency_of_layer(dev.name, l, batch)
            total += tf + tb
        return total

    def build_stage(layer_span: Tuple[int, int], group_devs: List[Device]) -> StageSpec:
        start, end = layer_span
        group_names = [d.name for d in group_devs]

        def mem_of(d: Device, y: int) -> int:
            return memory_of_stage(d.name, layer_span, y)

        def lat_of(d: Device, y: int) -> float:
            return stage_latency_of_device(d, layer_span, y)

        y_alloc = allocate_fn(group_devs, micro_batch_size, mem_of, lat_of, block_size)

        # Ef/Eb: max over devices of per-layer sums under y_d
        Ef, Eb = 0.0, 0.0
        for d in group_devs:
            y = y_alloc[d.name]
            tf_sum, tb_sum = 0.0, 0.0
            for l in range(start, end):
                tf, tb = latency_of_layer(d.name, l, y)
                tf_sum += tf
                tb_sum += tb
            Ef = max(Ef, tf_sum)
            Eb = max(Eb, tb_sum)

        # AllReduce per step (ring lower bound): 2(|G|-1)/|G| * sum(weights)/min_bw
        sum_w = sum(weights_size[l] for l in range(start, end))
        if len(group_names) <= 1:
            Ta = 0.0
        else:
            ar_bw = min_group_bw(group_names)
            Ta = (2.0 * (len(group_names) - 1) * sum_w) / (len(group_names) * ar_bw)

        return StageSpec(start, end, group_names, y_alloc, Ef, Eb, Ta)

    def compose_round_latency(stages: List[StageSpec]) -> Tuple[float, dict]:
        p = len(stages)
        S = 2 * p - 1
        Ef = [0.0] * S
        Eb = [0.0] * S
        Ta = [0.0] * S

        # exec steps at even indices
        for i, st in enumerate(stages):
            Ef[2 * i] = st.Ef
            Eb[2 * i] = st.Eb
            Ta[2 * i] = st.Ta

        # comm steps at odd indices (between stages i and i+1)
        for i in range(p - 1):
            boundary_layer = stages[i].layer_end - 1
            a = activation_size[boundary_layer]
            g1 = stages[i].device_names
            g2 = stages[i + 1].device_names
            link_bw = between_groups_bw(g1, g2)
            t_comm = (micro_batch_size * a) / link_bw
            Ef[2 * i + 1] = t_comm
            Eb[2 * i + 1] = t_comm
            Ta[2 * i + 1] = 0.0

        # Tw: prefix sum of Ef
        Tw = [0.0] * S
        cum = 0.0
        for s in range(S):
            Tw[s] = cum
            cum += Ef[s]

        # Dominant step = argmax(Ef+Eb)
        sums = [Ef[s] + Eb[s] for s in range(S)]
        dm = max(range(S), key=lambda s: sums[s])

        # Te: M*(Ef+Eb at dm) +/- offsets
        pref = [0.0] * (S + 1)
        for s in range(S):
            pref[s + 1] = pref[s] + (Ef[s] + Eb[s])
        base = num_micro_batches * (Ef[dm] + Eb[dm])
        Te = [0.0] * S
        for s in range(S):
            if s < dm:
                Te[s] = base + (pref[dm] - pref[s])
            else:
                Te[s] = base - (pref[s] - pref[dm])

        totals = [Tw[s] + Te[s] + Ta[s] for s in range(S)]
        obj = max(totals)
        meta = dict(Ef=Ef, Eb=Eb, Ta=Ta, Tw=Tw, Te=Te, totals=totals, dominant_step_index=dm)
        return obj, meta

    # DP over suffixes of layers/devices:
    L = num_layers
    devs_sorted = sorted(devices, key=lambda d: d.memory_budget, reverse=True)
    N = len(devs_sorted)
    P_cap = max_stages or min(L, N)

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def solve(l: int, n: int, p: int) -> Tuple[float, Tuple[StageSpec, ...]]:
        # Last l layers on last n devices, split into p stages
        if l <= 0 or n <= 0 or p <= 0 or p > l or p > n:
            return float("inf"), tuple()
        if p == 1:
            span = (L - l, L)
            group = devs_sorted[N - n : N]
            st = build_stage(span, group)
            obj, _ = compose_round_latency([st])
            return obj, (st,)

        best_obj = float("inf")
        best: Tuple[StageSpec, ...] = tuple()

        # Enumerate head-split: head uses l_head, n_head; rest uses l_sub, n_sub
        for l_sub in range(1, l):       # ensure both parts non-empty
            for n_sub in range(1, n):
                # sub-pipeline with p-1 stages on (l_sub, n_sub)
                sub_obj, sub_stages = solve(l_sub, n_sub, p - 1)
                if sub_obj == float("inf"):
                    continue

                # head stage on (l - l_sub) layers, (n - n_sub) devices
                span_head = (L - l, L - l_sub)
                group_head = devs_sorted[N - n : N - n_sub]
                st_head = build_stage(span_head, group_head)

                obj, _ = compose_round_latency([st_head, *sub_stages])
                if obj < best_obj:
                    best_obj = obj
                    best = (st_head, *sub_stages)

        return best_obj, best

    best_overall = float("inf")
    best_plan: Tuple[StageSpec, ...] = tuple()
    for p in range(1, P_cap + 1):
        obj, sts = solve(L, N, p)
        if obj < best_overall:
            best_overall = obj
            best_plan = sts

    round_latency, meta = compose_round_latency(list(best_plan))
    return PipelinePlan(stages=list(best_plan), round_latency=round_latency, meta=meta)

