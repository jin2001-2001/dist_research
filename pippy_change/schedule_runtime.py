import copy
import csv
import itertools
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Callable, NamedTuple, Optional, Union
import json
from dataclasses import dataclass, asdict
import os 

import torch
import torch.distributed as dist
import torch.distributed.pipelining.schedules as schedule
from torch.distributed.pipelining.schedules import _Action, _ComputationType
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

# Convenience shorthand for compute actions only since they are used in 'simple schedule format'
F = FORWARD
I = BACKWARD_INPUT
W = BACKWARD_WEIGHT
B = FULL_BACKWARD

# ---------------------------------------------------------------------------
# Action-string helpers (UPDATED FORMAT: [stage],[rank],[action type],[microbatch],[dest_rank])
# ---------------------------------------------------------------------------

# ORIGINAL: _action_regex = re.compile(r"(\d+)(F|I|B|W|UNSHARD|RESHARD|SEND_F|RECV_F|SEND_B|RECV_B)(\d*)")
# REPLACED with a pattern that understands the *comma separated* five‑field format.


class PipelineScheduleRuntimeWithDirection(schedule.PipelineScheduleMulti):

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
        if not hasattr(self, "_global_batch"):
            env_batch = os.getenv("PROFILE_BATCH")          # export PROFILE_BATCH=17
            self._profile_batch = int(env_batch) if env_batch is not None else None

            from timer import TimelineRecorder
            self._rec = TimelineRecorder(self.rank)        
            self._timeline_saved = False
            self._global_batch = -1
        
        current_batch = self._global_batch
        self._global_batch += 1

        record_this_batch = (
            self._profile_batch is not None                   
            and current_batch == self._profile_batch        
        )
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
                comp_type = action.computation_type
                mb_index: int = (
                    action.microbatch_index
                    if action.microbatch_index is not None
                    else -1
                )
                rank = action.rank
                dest_rank = action.dest_rank
                assert mb_index >= 0 or comp_type in (
                    UNSHARD,
                    RESHARD,
                ), f"{action=} missing mb_index"
                stage_idx = action.stage_index
                stage = stage_index_to_stage[stage_idx]
                stage_uses_fsdp = isinstance(stage.submod, schedule.FSDPModule)
                # see [Note: V-schedule special case]
                is_next_stage_on_this_rank = stage_idx + 1 in stage_index_to_stage
                is_prev_stage_on_this_rank = stage_idx - 1 in stage_index_to_stage

                logger.debug(
                    "_PipelineScheduleRuntime running time_step %d, action %s",
                    time_step,
                    action,
                )

             
                if comp_type == SEND_F:
                    with self._rec.record("SEND_F", stage_idx, mb_index):
                        ops = (
                            stage.get_fwd_send_ops(mb_index, rank=rank, dest_rank=dest_rank)
                            if rank is not None and dest_rank is not None
                            else stage.get_fwd_send_ops(mb_index)
                        )
                        send_ops.append(schedule._batch_p2p(ops))

                elif comp_type == SEND_B:
                    with self._rec.record("SEND_B", stage_idx, mb_index):
                        ops = (
                            stage.get_bwd_send_ops(mb_index, rank=rank, dest_rank=dest_rank)
                            if rank is not None and dest_rank is not None
                            else stage.get_bwd_send_ops(mb_index)
                        )
                        send_ops.append(schedule._batch_p2p(ops))

                elif comp_type == RECV_F:
                    assert (stage_idx, mb_index) not in fwd_recv_ops, (
                        f"Recv twice for stage_idx={stage_idx} mb_index={mb_index} without executing forward"
                    )
                    with self._rec.record("RECV_F", stage_idx, mb_index):
                        ops = (
                            stage.get_fwd_recv_ops(mb_index, rank=rank, dest_rank=dest_rank)
                            if rank is not None and dest_rank is not None
                            else stage.get_fwd_recv_ops(mb_index)
                        )
                        fwd_recv_ops[(stage_idx, mb_index)] = schedule._batch_p2p(ops)

                elif comp_type == RECV_B:
                    assert (stage_idx, mb_index) not in bwd_recv_ops, (
                        f"Recv twice for stage_idx={stage_idx} mb_index={mb_index} without executing backward"
                    )
                    with self._rec.record("RECV_B", stage_idx, mb_index):
                        ops = (
                            stage.get_bwd_recv_ops(mb_index, rank=rank, dest_rank=dest_rank)
                            if rank is not None and dest_rank is not None
                            else stage.get_bwd_recv_ops(mb_index)
                        )
                        bwd_recv_ops[(stage_idx, mb_index)] = schedule._batch_p2p(ops)
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
                    if stage_uses_fsdp:
                        _assert_unsharded(stage_idx)

                    if (
                        not stage.is_first
                        # no recv op expected for V-schedule special case (see [Note: V-schedule special case])
                        and not is_prev_stage_on_this_rank
                    ):
                        assert (
                            stage_idx,
                            mb_index,
                        ) in fwd_recv_ops, f"Computing {action=} before receiving input"
                        schedule._wait_batch_p2p(fwd_recv_ops.pop((stage_idx, mb_index)))
                        
                    with self._rec.record("FORWARD", stage_idx, mb_index):
                        output = stage.forward_one_chunk(
                            mb_index, arg_mbs[mb_index], kwarg_mbs[mb_index]
                        )
                        
                    self._maybe_compute_loss(stage, output, target_mbs, mb_index)

                    # SEND/RECV op are avoided for special case with 2 adjacent stages on same rank
                    # see [Note: V-schedule special case]
                    if is_next_stage_on_this_rank:
                        stage_index_to_stage[stage_idx + 1].set_local_fwd_input(
                            output, mb_index
                        )

                elif comp_type == FULL_BACKWARD:
                    if stage_uses_fsdp:
                        _assert_unsharded(stage_idx)

                    if (
                        not stage.is_last
                  
                        and not is_next_stage_on_this_rank
                    ):
                        assert (
                            stage_idx,
                            mb_index,
                        ) in bwd_recv_ops, (
                            f"Attempted to run compute {action=} before receiving input"
                        )
                        schedule._wait_batch_p2p(bwd_recv_ops.pop((stage_idx, mb_index)))
                    loss = self._maybe_get_loss(stage, mb_index)
                    backward_counter[stage_idx] += 1
                    last_backward = backward_counter[stage_idx] == self._n_microbatches
                    grad_scale_factor = self._n_microbatches if self.scale_grads else 1
                    with self._rec.record("FULL_BACKWARD", stage_idx, mb_index):
                        stage.backward_one_chunk(
                            mb_index,
                            loss=loss,
                            full_backward=True,
                            last_backward=last_backward,
                        )
                    if last_backward:
                        stage.scale_grads(grad_scale_factor)
 
                    if is_prev_stage_on_this_rank:
                        stage_index_to_stage[stage_idx - 1].set_local_bwd_input(
                            stage.get_local_bwd_output(mb_index), mb_index
                        )
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
                    with self._rec.record("BACKWARD_INPUT", stage_idx, mb_index):
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
                    
                    with self._rec.record("BACKWARD_WEIGHT", stage_idx, mb_index):
                        stage.backward_weight_one_chunk(
                            mb_index,
                            last_backward=backward_counter[stage_idx]
                            == self._n_microbatches,
                        )
                        
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
                        self.pipeline_order_with_comms, 
                        error_step_number=time_step,
                    )
                )
                raise e
       
        while len(send_ops):
            schedule._wait_batch_p2p(send_ops.pop())

        assert len(unshard_ops) == 0, "Unused unshard operations"


        self._update_losses(self._stages, losses)
        
        if (
            record_this_batch
            and self._profile_batch is not None
            and not self._timeline_saved
        ):
            fname_prefix = f"timeline_batch{self._profile_batch}"
            self.save_timeline(fname_prefix)
            self._timeline_saved = True
            self._rec.set_enabled(False)   

    def save_timeline(self, fname_prefix="timeline"):
        """
        每个 rank dump 一份 json；如果想集中到 rank0，
        可用 dist.gather_object 把 self._rec.events 收到 0 号后再统一写。
        """
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
