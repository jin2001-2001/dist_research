from dataclasses import dataclass
from typing import Callable, Dict, List

@dataclass(frozen=True)
class Device:
    """
    A single device in the device group.

    Attributes
    ----------
    name : str
        Unique device identifier.
    memory_budget : int
        Peak memory budget for this step (in bytes or arbitrary units).
    capacity : float
        Computing capacity v_d (higher = faster). In the paper it's defined as
        the inverse of the FP+BP latency with a micro-batch. You may precompute
        it from your profiler and fill here.
    """
    name: str
    memory_budget: int
    capacity: float


def allocate_microbatch_samples(
    devices: List[Device],
    micro_batch_size: int,
    memory_of: Callable[[Device, int], int],
    latency_of: Callable[[Device, int], float],
    block_size: int = 1,
) -> Dict[str, int]:
    """
    Allocate the samples of one micro-batch across a heterogeneous device group
    under per-device memory budgets, following Algorithm 1.

    Parameters
    ----------
    devices : List[Device]
        The device group G_s of this execution step. Each device provides
        its memory budget and capacity v_d.
    micro_batch_size : int
        The size B of the (single) micro-batch to be allocated.
    memory_of : Callable[[Device, int], int]
        Peak memory function for this step: given a device and the number of
        samples y allocated to that device, return the peak memory footprint
        Mem_s(y) on that device. Must be monotonic non-decreasing in y.
    latency_of : Callable[[Device, int], float]
        Execution latency function for this step: given a device and the number
        of samples y allocated to that device, return the FP+BP time on that
        device (used to detect the straggler).
    block_size : int, optional (default=1)
        The unit “block” moved per iteration during straggler offloading.
        Increase to reduce planning overhead at the cost of coarser balancing.

    Returns
    -------
    Dict[str, int]
        Allocation Y_s as a mapping: device_name -> samples (y_d), with
        sum(y_d) == micro_batch_size.

    Raises
    ------
    ValueError
        If the micro-batch cannot fit into the group memory budgets (i.e.,
        sum(max_batch_size_under_budget(d)) < B).

    Notes
    -----
    - Phase 1 (MemoryAwareBalancing): proportional to capacity within memory.
    - Phase 2 (StragglerWorkloadOffloading): iteratively move blocks from the
      slowest device to the fastest device that still has memory headroom,
      while the slowest device’s latency improves.
    """
    # ---- helpers -------------------------------------------------------------

    def max_batch_size_under_budget(d: Device) -> int:
        """Binary search the maximum y such that memory_of(d, y) <= d.memory_budget."""
        lo, hi = 0, micro_batch_size  # never need > B on a single device
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if memory_of(d, mid) <= d.memory_budget:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # Pre-check feasibility: total capacity under memory budgets must hold B
    if sum(max_batch_size_under_budget(d) for d in devices) < micro_batch_size:
        raise ValueError("Insufficient memory across devices to place the micro-batch.")

    # The allocation to build.
    y: Dict[str, int] = {d.name: 0 for d in devices}

    # For convenience maps
    dev_by_name = {d.name: d for d in devices}

    # ---- Phase 1: Memory-Aware Balancing -----------------------------------
    def memory_aware_balancing(group: List[Device], beta: int) -> None:
        """
        Recursively distribute 'beta' samples across 'group' by proportional
        capacity while respecting each device's remaining memory headroom.

        Parameters
        ----------
        group : List[Device]
            Current device subset with remaining memory headroom.
        beta : int
            Unallocated sample count to place in this recursion.
        """
        if not group and beta > 0:
            # Exit with Fail (T = +inf in paper). Here we raise to surface it.
            raise ValueError("No devices left but still have samples to allocate.")
        if beta == 0:
            return

        # Compute total capacity only over devices that still can take >0 samples.
        caps = {}
        total_cap = 0.0
        headroom: Dict[str, int] = {}
        for d in group:
            bs_total = max_batch_size_under_budget(d)
            avail = bs_total - y[d.name]
            if avail > 0:
                headroom[d.name] = avail
                caps[d.name] = d.capacity
                total_cap += d.capacity

        if total_cap == 0.0:
            # No headroom anywhere but beta > 0 -> impossible
            raise ValueError("No remaining memory headroom for allocation.")

        # First pass: floor of proportional shares, clipped by headroom
        allocated_this_round = 0
        remainders: List[tuple[str, float]] = []
        for d in group:
            avail = headroom.get(d.name, 0)
            if avail <= 0:
                continue
            ideal = beta * (caps[d.name] / total_cap)
            take = min(int(ideal), avail)
            y[d.name] += take
            allocated_this_round += take
            remainders.append((d.name, ideal - int(ideal)))

        # Greedy distribute leftovers by fractional remainder, still respecting headroom
        leftover = beta - allocated_this_round
        if leftover > 0:
            for dname, _rem in sorted(remainders, key=lambda x: x[1], reverse=True):
                if leftover == 0:
                    break
                avail = headroom[dname] - (y[dname] - (max_batch_size_under_budget(dev_by_name[dname]) - headroom[dname]))
                # Above: compute per-device take in this round as y_now - y_before; re-derive residual headroom
                # Simpler: recompute headroom fresh:
                avail = max_batch_size_under_budget(dev_by_name[dname]) - y[dname]
                if avail > 0:
                    y[dname] += 1
                    leftover -= 1

        # Recurse with unallocated portion on devices that still have headroom
        beta_prime = beta - (beta - leftover)
        if leftover > 0:
            next_group = [d for d in group if (max_batch_size_under_budget(d) - y[d.name]) > 0]
            memory_aware_balancing(next_group, leftover)

    # Kick off Phase 1
    memory_aware_balancing(devices, micro_batch_size)

    # ---- Phase 2: Straggler Workload Offloading -----------------------------
    def straggler_workload_offloading() -> None:
        """
        Iteratively move 'block_size' samples from the slowest device to the
        fastest device that still has memory headroom, as long as the slowest
        device's latency strictly improves after the move.
        """
        # initial ordering by current latency with Y_s
        def dev_latency(d: Device) -> float:
            return latency_of(d, y[d.name])

        # Pre-sort (ascending latency: fastest first)
        group_sorted = sorted(devices, key=dev_latency)
        if not group_sorted:
            return

        while True:
            # Identify current straggler (slowest device) and its latency
            old_straggler = max(group_sorted, key=dev_latency)
            old_slowest_latency = dev_latency(old_straggler)

            # Pick the fastest device with enough headroom to receive one block
            recipient = None
            for cand in group_sorted:  # ascending latency
                if cand.name == old_straggler.name:
                    continue
                if (y[cand.name] + block_size) <= max_batch_size_under_budget(cand):
                    recipient = cand
                    break

            # No valid recipient or nothing to move
            if recipient is None or y[old_straggler.name] <= 0:
                break

            # Apply the transfer
            move = min(block_size, y[old_straggler.name])
            y[old_straggler.name] -= move
            y[recipient.name] += move

            # Re-order and check improvement
            group_sorted = sorted(devices, key=dev_latency)
            new_straggler = max(group_sorted, key=dev_latency)
            new_slowest_latency = dev_latency(new_straggler)

            # Continue only if slowest improved; otherwise revert and stop
            if new_slowest_latency >= old_slowest_latency:
                y[old_straggler.name] += move
                y[recipient.name] -= move
                break

    # Run Phase 2
    straggler_workload_offloading()

    # Final sanity
    assert sum(y.values()) == micro_batch_size, "Allocation does not sum to B."
    return y
