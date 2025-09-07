# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
from abc import ABC, abstractmethod
from typing import Any, Callable, cast, Optional, Union, List
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.fx.node import Argument, map_aggregate
from torch.nn.parallel import DistributedDataParallel
from torch.utils._pytree import tree_map_only

from pipelining_source_code.stage import PipelineStage, InputInfo, _RecvInfo, _RootArgPlaceholder, _normalize_model_output_as_tuple
from pipelining_source_code._debug import map_debug_info
from pipelining_source_code._utils import flatten_args
from pipelining_source_code._backward import stage_backward, stage_backward_input, stage_backward_weight
logger = logging.getLogger(__name__)

# ===== TAG-ADD: tag 构造工具 =====
_DIR_SHIFT, _MB_SHIFT, _SLOT_SHIFT, _SPLIT_SHIFT = 27, 17, 8, 0
_MB_BITS, _SLOT_BITS, _SPLIT_BITS = 10, 9, 8
_MB_MASK, _SLOT_MASK, _SPLIT_MASK = (1<<_MB_BITS)-1, (1<<_SLOT_BITS)-1, (1<<_SPLIT_BITS)-1

def _mk_tag(direction: int, microbatch_id: int, slot_idx: int, split_idx: int) -> int:
    """31-bit tag。若越界则回退到一致哈希（两端独立可复现）。"""
    need_fallback = (
        microbatch_id > _MB_MASK or
        slot_idx      > _SLOT_MASK or
        split_idx     > _SPLIT_MASK
    )
    if not need_fallback:
        return ((direction & 1) << _DIR_SHIFT) | \
               ((microbatch_id & _MB_MASK) << _MB_SHIFT) | \
               ((slot_idx      & _SLOT_MASK) << _SLOT_SHIFT) | \
               ((split_idx     & _SPLIT_MASK) << _SPLIT_SHIFT)

    # fallback: 简单一致哈希压 31 bit
    v = (direction & 1) << 61
    v ^= (int(microbatch_id) & 0xFFFFFFFF) << 30
    v ^= (int(slot_idx)      & 0x3FFFFFFF) << 10
    v ^= (int(split_idx)     & 0x3FF)
    v ^= (v >> 33); v *= 0xff51afd7ed558ccd; v &= (1<<64)-1; v ^= (v >> 33)
    return int(v & 0x7fffffff)


class PipelineStage_with_mutiple_ranks(PipelineStage):
    def __init__(
        self,
        submodule: nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        input_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        output_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        group: Optional[dist.ProcessGroup] = None,
        dw_builder: Optional[Callable[[], Callable[..., None]]] = None,
        prev_group: list[int] = None, 
        this_group: list[int] = None, 
        next_group: list[int] = None
    ):
        self.prev_group = prev_group          # list[int] | None
        self.this_group = this_group          # list[int]
        self.next_group = next_group          # list[int] | None
        
        self.dp_group = dist.new_group(ranks=this_group)
        self.leader   = min(this_group)

        self.is_leader = dist.get_rank() == self.leader
        self._fwd_post_recv: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._last_comm_plan = {}   # key: (kind, mb_index) -> List[int] (每分块的 op 数量)
        
        self.is_print = 0
        
        super().__init__(submodule, stage_index, num_stages, device, input_args, output_args, group, dw_builder)
        
        
        
    @property
    def is_first(self):
        """
        Returns true if this stage is the first stage in the pipeline.
        """
        return self.prev_group is None

    @property
    def is_last(self):
        """
        Returns true if this stage is the last stage in the pipeline.
        """
        return self.next_group is None
    
    def _shape_inference(self, args, kwargs=None):
        if kwargs is None:
            kwargs = {}
        assert args is not None

        # --- Common ---
        dp_size = dist.get_world_size(self.dp_group)
        is_multi_dp = dp_size > 1

        # === 1) Handle inputs ===
        if self.prev_group is None:
            # First stage (pure DP or "first stage + DP")
            if is_multi_dp:
                if self.is_leader:
                    meta_args = tree_map_only(torch.Tensor, lambda x: x.to("meta"), args)
                    # Leader must also participate in the same broadcast
                    dist.broadcast_object_list([meta_args], src=self.leader, group=self.dp_group)
                    args = meta_args
                else:
                    buf = [None]
                    dist.broadcast_object_list(buf, src=self.leader, group=self.dp_group)
                    args = buf[0]
            else:
                # dp_size == 1, no collectives needed
                args = tree_map_only(torch.Tensor, lambda x: x.to("meta"), args)
        else:
            # Non-first stage: receive from previous stage's leader, then DP-broadcast locally
            prev_leader = min(self.prev_group)
            if self.is_leader:
                buf = [None]
                dist.recv_object_list(buf, src=prev_leader, group=dist.group.WORLD)  # world group
                args = buf[0]
                # Safety: ensure meta in case upstream sent real tensors by mistake
                args = tree_map_only(torch.Tensor, lambda x: x.to("meta"), args)
                if is_multi_dp:
                    dist.broadcast_object_list([args], src=self.leader, group=self.dp_group)
            else:
                if is_multi_dp:
                    buf = [None]
                    dist.broadcast_object_list(buf, src=self.leader, group=self.dp_group)
                    args = buf[0]
                else:
                    # dp_size == 1 and not leader cannot happen
                    raise RuntimeError("Unexpected non-leader with dp_size==1")

        # Basic sanity
        assert isinstance(args, tuple), f"shape-infer args must be tuple, got {type(args)}"

        # === 2) Cache inputs & run a dummy forward on zeros to compute outputs' shapes ===
        self.inputs_meta = args
        args = tree_map_only(torch.Tensor, lambda x: torch.zeros_like(x, device=self.device), args)

        with torch.no_grad():
            outputs = self.submod(*args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        outputs_meta = tuple(tree_map_only(torch.Tensor, lambda x: x.to("meta"), outputs))
        self._configure_outputs_meta(outputs_meta)

        # === 3) Send to next stage (only leader), and DP-broadcast locally for consistency ===
        if self.next_group is not None:
            next_leader = min(self.next_group)
            if self.is_leader:
                dist.send_object_list([outputs_meta], dst=next_leader, group=dist.group.WORLD)  # world group

        if is_multi_dp:
            dist.broadcast_object_list([outputs_meta], src=self.leader, group=self.dp_group)

        return outputs_meta



    
    def _get_recv_ops(
        self,
        recv_infos: tuple[InputInfo, ...],
        rank: int, 
        dest_rank: int
    ) -> list[dist.P2POp]:
        """
        Helper function shared by `get_fwd_recv_ops` and `get_bwd_recv_ops`.
        Returns a list of ops that correspond to the recv infos.
        """
        ops: list[dist.P2POp] = []
        for info in recv_infos:
            if not isinstance(info, _RecvInfo):
                continue

            peer_rank = dest_rank
            peer_global_rank = (
                peer_rank
                if self.group is None
                else dist.get_global_rank(self.group, peer_rank)
            )
            ops.append(
                dist.P2POp(dist.irecv, info.buffer, peer_global_rank, self.group)
            )

        return ops
    
    def _compute_1d_slices(self, numel: int, num_splits: int) -> list[tuple[int, int]]:
        """
        在 1D 长度为 numel 的序列上等量切分成 <= num_splits 份。
        如果 num_splits > numel，则最多只切 numel 份（每份至少 1 个元素）。
        返回 [(start, length), ...]，按元素序覆盖整个向量。
        """
        if num_splits <= 1 or numel <= 0:
            return [(0, numel)]
        k = min(num_splits, numel)
        base = numel // k
        rem = numel % k
        slices = []
        start = 0
        for i in range(k):
            length = base + (1 if i < rem else 0)
            slices.append((start, length))
            start += length
        return slices

    def get_fwd_send_ops(self, fwd_chunk_id: int, rank: int, dest_rank: int, num_splits: int = 1) -> list[dist.P2POp]:
        """
        Forward activations: 1D-flat chunking.
        生成顺序：外层 split_idx，内层轮询所有张量；同时统计每个分块的 P2POp 数量。
        """
        output_tuple, _ = self.fwd_cache[fwd_chunk_id]
        ops: list[dist.P2POp] = []
        ops_per_chunk: list[int] = [0 for _ in range(max(1, num_splits))]

        peer_rank = dest_rank
        peer_global_rank = (peer_rank if self.group is None else dist.get_global_rank(self.group, peer_rank))

        # 预计算每个张量的 1D 视图与切片表
        # …前置不变…
        plans = []  # [(slot_idx, flat, slices, dst_stages)]
        slot_ctr = 0  # ===== TAG-ADD =====
        for idx, out in enumerate(output_tuple):
            dst_stages = self.act_send_info[idx]
            if dst_stages is None:
                continue
            if not isinstance(out, torch.Tensor):
                continue
            flat = out.contiguous().view(-1)
            slices = self._compute_1d_slices(flat.numel(), num_splits)
            plans.append((slot_ctr, flat, slices, dst_stages))  # ===== TAG-ADD =====
            slot_ctr += 1                                       # ===== TAG-ADD =====

        for split_idx in range(max(1, num_splits)):
            for slot_idx, flat, slices, dst_stages in plans:    # ===== TAG-ADD (只改解包变量名) =====
                if split_idx >= len(slices):
                    continue
                off, ln = slices[split_idx]
                chunk_view = flat.narrow(0, off, ln)
                for _dst in dst_stages:
                    if _dst is None:
                        continue
                    # 只新增 tag，不改变你的发送逻辑
                    tag = _mk_tag(0, fwd_chunk_id, slot_idx, split_idx)  # ===== TAG-ADD =====
                    ops.append(dist.P2POp(dist.isend, chunk_view, peer_global_rank, self.group, tag=tag))  # ===== TAG-ADD =====
                    ops_per_chunk[split_idx] += 1


        # 记录“每分块 op 数”
        self._last_comm_plan[("SEND_F", fwd_chunk_id)] = ops_per_chunk
        return ops
            
    def get_fwd_recv_ops(self, fwd_chunk_id: int, rank: int, dest_rank: int, num_splits: int = 1) -> list[dist.P2POp]:
        """
        Forward activations: 1D-flat irecv into final buffers.
        生成顺序：外层 split_idx，内层轮询所有目标缓冲；同时统计每个分块的 P2POp 数量。
        """
        recv_infos: tuple[InputInfo, ...] = self.args_recv_info[fwd_chunk_id]
        ops: list[dist.P2POp] = []
        ops_per_chunk: list[int] = [0 for _ in range(max(1, num_splits))]

        peer_rank = dest_rank
        peer_global_rank = (peer_rank if self.group is None else dist.get_global_rank(self.group, peer_rank))

        plans = []  # [(slot_idx, buf_flat, slices)]
        slot_ctr = 0  # ===== TAG-ADD =====
        for info in recv_infos:  # 保持你原来的写法，不读 arg_idx
            if not isinstance(info, _RecvInfo):
                continue
            buf = info.buffer
            if not isinstance(buf, torch.Tensor):
                continue
            buf_flat = buf.contiguous().view(-1)
            slices = self._compute_1d_slices(buf_flat.numel(), num_splits)
            plans.append((slot_ctr, buf_flat, slices))  # ===== TAG-ADD =====
            slot_ctr += 1                               # ===== TAG-ADD =====

        for split_idx in range(max(1, num_splits)):
            for slot_idx, buf_flat, slices in plans:    # ===== TAG-ADD (只改解包变量名) =====
                if split_idx >= len(slices):
                    continue
                off, ln = slices[split_idx]
                view = buf_flat.narrow(0, off, ln)
                tag = _mk_tag(0, fwd_chunk_id, slot_idx, split_idx)  # ===== TAG-ADD =====
                ops.append(dist.P2POp(dist.irecv, view, peer_global_rank, self.group, tag=tag))  # ===== TAG-ADD =====
                ops_per_chunk[split_idx] += 1


        self._last_comm_plan[("RECV_F", fwd_chunk_id)] = ops_per_chunk
        return ops


    
    def finish_fwd_recv(self, fwd_chunk_id: int) -> None:
        """
        完成接收后的“粘合”：把每个 tmp copy_ 到最终的 target_view。
        """
        post_list = self._fwd_post_recv.pop(fwd_chunk_id, None)
        if not post_list:
            return
        for tmp, view in post_list:
            # 这里用 copy_，目标 view 可能是非连续的窄视图，但 copy_ 支持
            with torch.no_grad():
                view.copy_(tmp)

    def get_bwd_send_ops(self, bwd_chunk_id: int, rank: int, dest_rank: int, num_splits: int = 1) -> list[dist.P2POp]:
        """
        Backward grads: 1D-flat chunking.
        生成顺序：外层 split_idx，内层轮询所有梯度张量；同时统计每个分块的 P2POp 数量。
        """
        self._check_chunk_id(bwd_chunk_id)
        if not self.has_backward or self.is_first:
            return []

        if self.grad_send_info is None:
            self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])

        grads_input = self.bwd_cache.pop(bwd_chunk_id)
        self.fwd_cache.pop(bwd_chunk_id, None)

        peer_rank = dest_rank
        peer_global_rank = (peer_rank if self.group is None else dist.get_global_rank(self.group, peer_rank))

        # 预计算每个梯度的 1D 视图与切片表（只发浮点/复数梯度，保持你原逻辑）
        # …前置不变…
        plans = []  # [(slot_idx, flat, slices)]
        slot_ctr = 0  # ===== TAG-ADD =====
        for grad, grad_recv_stage in zip(grads_input, self.grad_send_info):
            if grad_recv_stage is None:
                continue
            if not isinstance(grad, torch.Tensor):
                if grad is not None:
                    raise RuntimeError(
                        f"[{self.stage_index}] expecting a gradient tensor for an input "
                        f"coming from stage {grad_recv_stage}, but got {type(grad)}"
                    )
                continue
            if not (grad.is_floating_point() or torch.is_complex(grad)):
                continue
            flat = grad.contiguous().view(-1)
            slices = self._compute_1d_slices(flat.numel(), num_splits)
            plans.append((slot_ctr, flat, slices))  # ===== TAG-ADD =====
            slot_ctr += 1                           # ===== TAG-ADD =====

        ops: list[dist.P2POp] = []
        ops_per_chunk: list[int] = [0 for _ in range(max(1, num_splits))]
        for split_idx in range(max(1, num_splits)):
            for slot_idx, flat, slices in plans:    # ===== TAG-ADD (只改解包变量名) =====
                if split_idx >= len(slices):
                    continue
                off, ln = slices[split_idx]
                chunk_view = flat.narrow(0, off, ln)
                tag = _mk_tag(1, bwd_chunk_id, slot_idx, split_idx)  # ===== TAG-ADD =====
                ops.append(dist.P2POp(dist.isend, chunk_view, peer_global_rank, self.group, tag=tag))  # ===== TAG-ADD =====
                ops_per_chunk[split_idx] += 1


        self._last_comm_plan[("SEND_B", bwd_chunk_id)] = ops_per_chunk
        return ops


    def get_bwd_recv_ops(self, bwd_chunk_id: int, rank: int, dest_rank: int, num_splits: int = 1) -> list[dist.P2POp]:
        """
        Backward grads: 1D-flat irecv into final buffers.
        生成顺序：外层 split_idx，内层轮询所有目标缓冲；同时统计每个分块的 P2POp 数量。
        """
        if not self.has_backward or self.is_last:
            return []

        recv_infos = self.grad_recv_info[bwd_chunk_id]
        ops: list[dist.P2POp] = []
        ops_per_chunk: list[int] = [0 for _ in range(max(1, num_splits))]

        peer_rank = dest_rank
        peer_global_rank = (peer_rank if self.group is None else dist.get_global_rank(self.group, peer_rank))

        plans = []  # [(slot_idx, buf_flat, slices)]
        slot_ctr = 0  # ===== TAG-ADD =====
        for info in recv_infos:  # 保持你原来的写法，不读 arg_idx
            if isinstance(info, _RecvInfo) and isinstance(info.buffer, torch.Tensor):
                buf_flat = info.buffer.contiguous().view(-1)
                slices = self._compute_1d_slices(buf_flat.numel(), num_splits)
                plans.append((slot_ctr, buf_flat, slices))  # ===== TAG-ADD =====
                slot_ctr += 1                               # ===== TAG-ADD =====

        ops: list[dist.P2POp] = []
        ops_per_chunk: list[int] = [0 for _ in range(max(1, num_splits))]
        for split_idx in range(max(1, num_splits)):
            for slot_idx, buf_flat, slices in plans:        # ===== TAG-ADD (只改解包变量名) =====
                if split_idx >= len(slices):
                    continue
                off, ln = slices[split_idx]
                view = buf_flat.narrow(0, off, ln)
                tag = _mk_tag(1, bwd_chunk_id, slot_idx, split_idx)  # ===== TAG-ADD =====
                ops.append(dist.P2POp(dist.irecv, view, peer_global_rank, self.group, tag=tag))  # ===== TAG-ADD =====
                ops_per_chunk[split_idx] += 1


        self._last_comm_plan[("RECV_B", bwd_chunk_id)] = ops_per_chunk
        return ops

    
    
    # def get_bwd_send_ops(self, bwd_chunk_id: int, rank: int, dest_rank: int) -> list[dist.P2POp]:
    #     """
    #     Get the gradient send ops for current stage's backward.
    #     """
    #     self._check_chunk_id(bwd_chunk_id)

    #     if not self.has_backward or self.is_first:
    #         return []

    #     if self.grad_send_info is None:
    #         self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])

    #     ops: list[dist.P2POp] = []
    #     grads_input = self.bwd_cache.pop(bwd_chunk_id)
    #     self.fwd_cache.pop(bwd_chunk_id, None)
        
    #     for grad, grad_recv_stage in zip(grads_input, self.grad_send_info):
    #         # 跳过 None 的梯度接收阶段（对应于整数类型的tensor）
    #         if grad_recv_stage is None:
    #             continue
                
    #         if isinstance(grad, torch.Tensor):
    #             # 额外检查：只发送浮点类型的梯度
    #             if not (grad.is_floating_point() or torch.is_complex(grad)):
    #                 #print(f"[{dist.get_rank()}] Skipping non-floating grad send: dtype={grad.dtype}")
    #                 continue
                    
    #             logger.debug(
    #                 "%s Sending gradient to Stage %s: %s",
    #                 self.log_prefix,
    #                 grad_recv_stage,
    #                 grad.size(),
    #             )
    #             peer_rank = dest_rank
    #             peer_global_rank = (
    #                 peer_rank
    #                 if self.group is None
    #                 else dist.get_global_rank(self.group, peer_rank)
    #             )
    #             ops.append(dist.P2POp(dist.isend, grad, peer_global_rank, self.group))
    #         elif grad is not None:
    #             raise RuntimeError(
    #                 f"[{self.stage_index}] expecting a gradient tensor for an input "
    #                 f"coming from stage {grad_recv_stage}, but got {type(grad)}"
    #             )
        
    #     return ops
    
    def _execute_allreduce(self):
        """
        Execute allreduce for gradients across the data parallel group.
        This should be called after all microbatches have been processed.
        """
        if not hasattr(self, 'dp_group') or self.dp_group is None:
            return
            
        # Check if the submodule is wrapped with DDP
        if isinstance(self.submod, DistributedDataParallel):
            # If using DDP, we need to manually trigger allreduce
            # since we've been using no_sync() during backward
            self._trigger_ddp_allreduce()
        else:
            # If not using DDP, manually allreduce all gradients
            self._manual_allreduce_gradients()
    
    def _trigger_ddp_allreduce(self):
        """
        Trigger DDP's allreduce mechanism manually.
        This is used when we've been accumulating gradients with no_sync().
        """
        if not isinstance(self.submod, DistributedDataParallel):
            return
            
        # Method 1: Use prepare_for_backward and trigger with a dummy backward
        # This is the approach used in the original paste.txt
        
        # Collect all output tensors from forward passes
        # Since we don't have access to the actual outputs here, 
        # we create a dummy tensor that depends on all parameters with gradients
        dummy_tensors = []
        for param in self.submod.parameters():
            if param.requires_grad and param.grad is not None:
                # Create a dummy tensor that depends on this parameter
                # Using a very small value to not affect actual gradients
                dummy = param.sum() * 0.0
                if dummy.requires_grad:
                    dummy_tensors.append(dummy)
        
        if not dummy_tensors:
            return
            
        # Prepare reducer for backward
        self.submod.reducer.prepare_for_backward(dummy_tensors)
        
        # Temporarily enable gradient synchronization
        original_sync = self.submod.require_backward_grad_sync
        self.submod.require_backward_grad_sync = True
        
        # Trigger backward to activate allreduce
        if dummy_tensors:
            # Sum all dummy tensors and backward
            total_dummy = sum(dummy_tensors)
            total_dummy.backward()
        
        # Restore original sync setting
        self.submod.require_backward_grad_sync = original_sync
    
    def _manual_allreduce_gradients(self):
        """
        Manually perform allreduce on all gradients when not using DDP.
        """
        if self.dp_group is None:
            return
            
        world_size = dist.get_world_size(self.dp_group)
        
        # Collect all parameters with gradients
        params_with_grad = []
        for param in self.submod.parameters():
            if param.grad is not None:
                params_with_grad.append(param)
        
        if not params_with_grad:
            return
        
        # Perform allreduce on each gradient
        # Can be optimized by batching small gradients together
        for param in params_with_grad:
            # Allreduce the gradient
            dist.all_reduce(
                param.grad.data,
                op=dist.ReduceOp.SUM,
                group=self.dp_group
            )
            # Average the gradient
            param.grad.data /= world_size
    
    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
        pack_size: int = 1
    ):
        """
        Perform forward pass on the stage with one microbatch.
        `args` and `kwargs` are the inputs from *external* to this stage.
        As of Sept 2024:
        - `args` applies to the first stage only, other stages receives args
          through activation transmission.
        - `kwargs` can be passed to all stages via respective `step` calls.
        """


        def _is_first_stage_dp(self):
            try:
                dp_size = dist.get_world_size(self.dp_group)
            except Exception:
                dp_size = 1
            return (self.prev_group is None) and (dp_size > 1)

        need_rt_bcast = _is_first_stage_dp(self)

        composite_args = None

        if need_rt_bcast:
            if self.is_leader:
                if not args or len(args) == 0:
                    raise RuntimeError(
                        f"[rank{dist.get_rank()}] First-stage leader got empty args at "
                        f"fwd_chunk_id={fwd_chunk_id}. Scheduler must pass root inputs to leader."
                    )
                dist.broadcast_object_list([args], src=self.leader, group=self.dp_group)
                composite_args = args
            else:
                buf = [None]
                dist.broadcast_object_list(buf, src=self.leader, group=self.dp_group)
                composite_args = buf[0]

        else:
            if args:
                composite_args = args
            else:
                composite_args = self._retrieve_recv_activations(fwd_chunk_id)

        if composite_args is None or len(composite_args) == 0:
            raise RuntimeError(
                f"[rank{dist.get_rank()}] Empty composite_args after dispatch at "
                f"stage={self.stage_index}, fwd_chunk_id={fwd_chunk_id}, "
                f"is_first={self.prev_group is None}."
            )


        composite_kwargs = kwargs or {}

        if (
            pack_size > 1
            and composite_args
            and isinstance(composite_args[0], torch.Tensor)
        ):
            mb_bs = composite_args[0].shape[0] // pack_size
            args_for_val = tuple(
                (t[:mb_bs] if isinstance(t, torch.Tensor) else t)
                for t in composite_args
            )
            kwargs_for_val = {
                k: (v[:mb_bs] if isinstance(v, torch.Tensor) else v)
                for k, v in composite_kwargs.items()
            }
        else:
            args_for_val = composite_args
            kwargs_for_val = composite_kwargs

        self._validate_fwd_input(args_for_val, kwargs_for_val)

        # Compute forward
        try:
            output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)

        except Exception as e:
            exc_msg = f"""
            {self.log_prefix} failed to run forward:
            args: {map_debug_info(composite_args)}
            kwargs: {map_debug_info(composite_kwargs)}
            """
            raise RuntimeError(exc_msg) from e

        # See [Note: pipeline model output type]
        output_tuple = _normalize_model_output_as_tuple(output)

        # Prepare for final output merge or reduction
        # Output chunks is only used for the last stage since we only merge the output of the last stage
        if self.is_last:
            self.output_chunks.append(output)

        # Save activations and inputs for backward
        flat_args = flatten_args(composite_args)
        flat_kwargs = flatten_args(composite_kwargs)
        flatten_input_tensors = flat_args + flat_kwargs
        self.fwd_cache[fwd_chunk_id] = (
            output_tuple,  # stage_output
            flatten_input_tensors,  # input_values
        )

        logger.debug(
            "%s Forwarded chunk %s, outputs: %s",
            self.log_prefix,
            fwd_chunk_id,
            map_debug_info(output),
        )
        if pack_size > 1 and isinstance(output_tuple, tuple) and len(output_tuple):
            outputs_for_val = tuple(
                t[:mb_bs] if isinstance(t, torch.Tensor) else t
                for t in output_tuple
            )
        elif pack_size > 1 and isinstance(output_tuple, torch.Tensor):
            outputs_for_val = output_tuple[:mb_bs]
        else:
            outputs_for_val = output_tuple

        self._validate_fwd_outputs(outputs_for_val)

        # We return the original user-provied output, not normalized to tuple.
        # See [Note: pipeline model output type]
        return output

    def backward_maybe_with_nosync(
        self,
        backward_type,
        bwd_kwargs: dict,
        last_backward: bool = False,
        retain_graph_for_packed_mbs: bool = False,
    ) -> tuple[tuple[Optional[torch.Tensor], ...], Optional[list[dict[str, Any]]]]:
        """
        Whether using PP with FSDP or DDP, there are some runtime differences between the last backward step and the
        other steps.  Namely, we need to accumulate gradients on previous steps and reduce them on the last step, but
        there are additional state-variables and performance considerations depending on the data parallelism used.
        This helper should adapt any pipeline parallel schedule to work with common/supported data parallel libraries.
        """

        def perform_backward(
            backward_type,
        ) -> Callable[
            [],
            tuple[tuple[Optional[torch.Tensor], ...], Optional[list[dict[str, Any]]]],
        ]:
            if backward_type == "full":
                return lambda: (
                    stage_backward(
                        bwd_kwargs["stage_output"],
                        bwd_kwargs["output_grads"],
                        bwd_kwargs["input_values"],
                        retain_graph_for_packed_mbs=retain_graph_for_packed_mbs
                    ),
                    None,
                )
            elif backward_type == "input":
                return lambda: stage_backward_input(
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                    bwd_kwargs["input_values"],
                    self.submod.parameters(),
                )
            elif backward_type == "weight":
                return lambda: (
                    stage_backward_weight(
                        self.submod.parameters(), bwd_kwargs["param_groups"]
                    ),
                    None,
                )
            else:
                raise RuntimeError(f"Unknown backward type: {backward_type}")

        # If submod is wrapped by DDP
        if isinstance(self.submod, DistributedDataParallel):
            if last_backward:
                # Last chunk, prepare for gradient reduction
                # HACK: reaching into DDP implementation details here. Is there a better way?
                self.submod.reducer.prepare_for_backward(  # type: ignore[union-attr, operator]
                    list(
                        torch.nn.parallel.distributed._find_tensors(  # type: ignore[attr-defined]
                            bwd_kwargs["stage_output"]
                        )
                    )
                )
                result = perform_backward(backward_type)()
            else:
                with self.submod.no_sync():  # type: ignore[operator]
                    result = perform_backward(backward_type)()
        # If submod is a FSDP module
        elif isinstance(self.submod, FSDPModule):
            self.submod.set_is_last_backward(False)
            self.submod.set_reshard_after_backward(False)
            self.submod.set_requires_gradient_sync(False)
            result = perform_backward(backward_type)()
            if last_backward:
                # Manually call post backward for FSDP
                def run_post_backward(fsdp_module: FSDPModule) -> None:
                    fsdp_module.set_is_last_backward(True)
                    fsdp_module.set_reshard_after_backward(True)
                    fsdp_module.set_requires_gradient_sync(True)
                    fsdp_state = fully_shard.state(fsdp_module)  # type: ignore[attr-defined]
                    for state in fsdp_state._state_ctx.all_states:
                        if state._fsdp_param_group:
                            state._fsdp_param_group.post_backward()

                    # it would be much better if pipelining backward invoked .backward so autograd hooks
                    # worked and modules like DDP/FSDP behaved as expected.  Working around this for the time being,
                    # we need to call this too to ensure FSDP syncs its grad reduction ops back to the default stream.
                    fsdp_state._root_post_backward_final_callback()

                run_post_backward(self.submod)

        else:
            # Non-DP submodule, regular backward
            result = perform_backward(backward_type)()

        grads, param_groups = result
        return grads, param_groups
    
    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward=False,
        retain_graph_for_packed_mbs: bool = False,
    ):
        """
        Perform backward pass on the module.
        This should only be called once per microbatch.

        If full_backward is True (the default), the full backward pass including weight and input gradients will be run,
        and it is an error to call `backward_weight_one_chunk` for this bwd_chunk_id.

        If full_backward is False, it is optional that `dw_runner` was provided to the PipelineStage at __init__ time,
        and a subsequent call to `backward_weight_one_chunk` is required to invoke dw_runner and complete the backward.

        last_backward is controlled by the schedule and signals synchronization of gradients across DP groups
        after the last backward.
        """
        self._check_chunk_id(bwd_chunk_id)

        (
            stage_output,
            input_values,
        ) = self.fwd_cache.pop(bwd_chunk_id)

        # Compute backward
        if self.is_last:
            # Last stage computes gradients from loss and has no gradients from
            # next stage
            bwd_kwargs = {
                "stage_output": loss,
                "output_grads": None,
                "input_values": input_values,
            }
        else:
             # Otherwise, receive gradients from next stage
            grads_output = self._retrieve_recv_grads(bwd_chunk_id)

            # === NEW (Plan B): average upstream grads across the next stage's DP replicas ===
            # If the next stage has N data-parallel members (len(self.next_group) == N),
            # each sender contributed a local gradient. To keep "mean loss" semantics,
            # divide by N on the receiver before autograd.backward.
            grads_output = self._avg_next_stage_grads(grads_output)
            # === END NEW ===

            # If an input to the pipeline requires gradient,
            # `torch.autograd.backward` will accumulate the gradient into the
            # `.grad` field of such input
            bwd_kwargs = {
                "stage_output": stage_output,
                "output_grads": grads_output,
                "input_values": input_values,
            }

        grads_input: tuple[Optional[torch.Tensor], ...] = ()

        # Custom backward function
        if self.dw_builder:
            # TODO: We may want to change our semantics so we are allowed to ignore
            # the 'dw_builder' and call full_backward directly when it is a full_backward op.
            grads_input, _ = self.backward_maybe_with_nosync(
                "full",
                bwd_kwargs,
                last_backward=last_backward,
                retain_graph_for_packed_mbs = retain_graph_for_packed_mbs
            )
            if full_backward:
                self.dw_builder()()
            else:
                self.dw_runner[bwd_chunk_id] = self.dw_builder()
        else:
            if full_backward:
                grads_input, _ = self.backward_maybe_with_nosync(
                    "full", bwd_kwargs, last_backward=last_backward,
                    retain_graph_for_packed_mbs = retain_graph_for_packed_mbs
                )
            else:
                param_groups: list[dict[str, Any]] | None = None
                # Skip the backward for the first stage since we will perform the weight update with
                # autograd.backward in backward_weight_one_chunk
                if not self.is_first:
                    if isinstance(bwd_kwargs["stage_output"], torch.Tensor):
                        bwd_kwargs["stage_output"] = (bwd_kwargs["stage_output"],)

                    # perform the partial backwards for the inputs with a custom backward function
                    # when the "stage_ouput" is a loss, then it is a tensor, otherwise it is a tuple of tensors
                    grads_input, param_groups = self.backward_maybe_with_nosync(
                        "input", bwd_kwargs, last_backward=last_backward
                    )

                # TODO: we dont need to save this, add to dw_runner?
                self.backward_state[bwd_chunk_id] = (
                    bwd_kwargs["input_values"],
                    param_groups,
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                )
                # Save a placeholder for the dw_runner
                self.dw_runner[bwd_chunk_id] = lambda: None

        if self.grad_send_info is None:
            self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])

        grads_input = tuple(
            (g if dst is not None and isinstance(g, torch.Tensor) else None)
            for g, dst in zip(grads_input, self.grad_send_info)
        )
        
        self.bwd_cache[bwd_chunk_id] = grads_input

        if self.is_last and not self.is_first:
            # Autograd dependencies:
            #    rest_of_autograd_graph -> stage_output -> loss
            # stage_output is no longer used in the last stage for backward and only needed
            # to return to the user in merge_output_chunks, therefore
            # this should be detached to release autograd graph context and free memory earlier
            for t in stage_output:
                if not t._is_view():  # views are not detachable in-place
                    t.detach_()

        logger.debug("%s Backwarded chunk %s", self.log_prefix, bwd_chunk_id)
        
    
    def _avg_next_stage_grads(self, grads):
        """
        Average upstream gradients received from the *next* stage across its DP replicas.

        Why:
        - If the next stage is data-parallel over N ranks (len(self.next_group) == N),
            each rank sends its *local* gradient dL/d(out). Summing those locally on this stage
            would implement "sum loss" semantics. To keep the common "mean loss" semantics while
            keeping LR unchanged, we should divide the received gradients by N before calling
            autograd.backward on this stage.

        Behavior:
        - No-op when there is no next stage (last stage) or N <= 1.
        - Supports Tensor / list / tuple / None (None is passed through).
        - In-place scaling is used to avoid extra allocations.
        """
        import torch
        import torch.distributed as dist

        fan_in = len(self.next_group) if self.next_group is not None else 1
        if grads is None or fan_in <= 1:
            return grads

        # Helper to scale a single item
        def _scale_one(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                # in-place division; safe for gradient tensors
                return x.div_(fan_in)
            return x  # for unexpected types we leave as-is

        # Compute a small debug metric before scaling (leader only) 
        pre_max = None
        try:
            if self.is_leader:
                if isinstance(grads, (list, tuple)) and len(grads) > 0 and isinstance(grads[0], torch.Tensor):
                    pre_max = grads[0].abs().max().item()
                elif isinstance(grads, torch.Tensor):
                    pre_max = grads.abs().max().item()
        except Exception:
            pre_max = None

        # Apply scaling to common containers
        if isinstance(grads, list):
            out = [_scale_one(g) for g in grads]
        elif isinstance(grads, tuple):
            out = tuple(_scale_one(g) for g in grads)
        else:
            out = _scale_one(grads)

        # Debug print after scaling (leader only)
        if self.is_leader and pre_max is not None:
            try:
                post_max = (
                    (out[0].abs().max().item() if isinstance(out, (list, tuple)) else out.abs().max().item())
                )
                # print(
                #     f"[{dist.get_rank()}] AVG upstream grads by 1/{fan_in} "
                #     f"(stage_idx={self.stage_index}) -> max|g| {pre_max:.6f} -> {post_max:.6f}"
                # )
            except Exception:
                pass

        return out
    
    def _create_grad_recv_info(
        self,
        act_send_info: dict,
    ) -> tuple:
        grad_recv_info_list = []
        outputs_meta = self.get_outputs_meta()

        if not self.is_last:
            for idx, dst_list in act_send_info.items():
                dst = dst_list[0]
                meta = outputs_meta[idx] if idx < len(outputs_meta) else None

                if meta is None or not torch.is_tensor(meta):
                    grad_recv_info_list.append(None)
                    continue
                
                if not (meta.is_floating_point() or torch.is_complex(meta)):
                    grad_recv_info_list.append(None)
                    continue

                buffer = _make_tensor_from_meta(meta, self.device)
                grad_recv_info_list.append(
                    _RecvInfo(
                        f"recv_grad_for_{self.stage_index}_from_{dst}",
                        dst,
                        buffer,
                    )
                )

        return tuple(grad_recv_info_list)
    
    def _create_grad_send_info(
        self,
        args_recv_info: tuple,
    ) -> list[Optional[int]]:
        grad_send_info: list[Optional[int]] = []

        def map_recv_to_send(a):
            if isinstance(a, _RecvInfo) and getattr(a, "buffer", None) is not None:
                if isinstance(a.buffer, torch.Tensor) and (
                    a.buffer.is_floating_point() or torch.is_complex(a.buffer)
                ):
                    grad_send_info.append(a.source)
                    return a.source
            grad_send_info.append(None)
            return None

        map_aggregate(args_recv_info, map_recv_to_send)
        return grad_send_info
    
    def _map_tensor_from_recv_info(
        self,
        recv_infos: tuple[InputInfo, ...],
    ):
        """
        Map tensors from recv infos to a list.
        """

        def get_recv_tensor(info):
            if info is None:
                return None
            elif isinstance(info, _RecvInfo):
                return info.buffer
            else:
                raise AssertionError(f"Expected _RecvInfo or None but got {type(info)}")

        return map_aggregate(cast(Argument, recv_infos), get_recv_tensor)

def _make_tensor_from_meta(
    example: Union[torch.Tensor, FakeTensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Create a real tensor from a tensor.
    """
    return torch.empty(
        example.size(),
        dtype=example.dtype,
        layout=example.layout,
        device=device,
    )

class PipelineStage_Multimodality_Head(PipelineStage_with_mutiple_ranks):
    def __init__(
        self,
        submodule: nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        input_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        output_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        group: Optional[dist.ProcessGroup] = None,
        dw_builder: Optional[Callable[[], Callable[..., None]]] = None,
        prev_group: list[int] = None, 
        this_group: list[int] = None, 
        next_group: list[int] = None,
        modal_type: str = None,
        fuse_stage: bool = False,
        fuse_function: Optional[Callable[..., Any]] = None,
        split_function: Optional[Callable[..., Any]] = None,
    ):  
        self.modal_type = modal_type
        self.fuse_stage = fuse_stage
        if fuse_function is not None and not callable(fuse_function):
            raise TypeError("fuse_function must be a callable object or pass None.")
        if split_function is not None and not callable(split_function):
            raise TypeError("fuse_function must be a callable object or pass None.")
        self.fuse_function = fuse_function
        self.split_function = split_function
        
        # forward 接收信息（按 mb、模态 → tuple[_RecvInfo,...]）
        self.mm_args_recv_info: dict[int, dict[str, tuple[_RecvInfo, ...]]] = defaultdict(dict)
        # backward 接收信息（按 mb、模态 → tuple[_RecvInfo,...]）
        self.mm_grad_recv_info: dict[int, dict[str, tuple[_RecvInfo, ...]]] = defaultdict(dict)

        # forward 未融合缓存（按 mb、模态 → tuple[Tensor,...] 或者 dict）
        self.mm_fwd_cache: dict[int, dict[str, tuple[torch.Tensor, ...]]] = defaultdict(dict)
        # backward 拆分后的梯度缓存（按 mb、模态 → tuple[Tensor,...]）
        self.mm_bwd_cache: dict[int, dict[str, tuple[torch.Tensor, ...]]] = defaultdict(dict)

        # 发送路由（可选：若你不想每次调用传 routes，可以预注册，这里给出占位）
        # 例如：{"vision": stage_id_3, "audio": stage_id_4}
        self.mm_act_send_routes: dict[str, list[int]] = defaultdict(list)
        self.mm_grad_send_routes: dict[str, list[int]] = defaultdict(list)
        
        super().__init__(submodule, stage_index, num_stages, device, input_args, output_args, group, dw_builder, prev_group,  this_group, next_group)
        
    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
        pack_size: int = 1
    ):
        """
        Perform forward pass on the stage with one microbatch.
        `args` and `kwargs` are the inputs from *external* to this stage.
        As of Sept 2024:
        - `args` applies to the first stage only, other stages receives args
        through activation transmission.
        - `kwargs` can be passed to all stages via respective `step` calls.
        """

        def _is_first_stage_dp(self):
            try:
                dp_size = dist.get_world_size(self.dp_group)
            except Exception:
                dp_size = 1
            return (self.prev_group is None) and (dp_size > 1)

        need_rt_bcast = _is_first_stage_dp(self)

        composite_args = None

        if need_rt_bcast:
            if self.is_leader:
                if not args or len(args) == 0:
                    raise RuntimeError(
                        f"[rank{dist.get_rank()}] First-stage leader got empty args at "
                        f"fwd_chunk_id={fwd_chunk_id}. Scheduler must pass root inputs to leader."
                    )
                dist.broadcast_object_list([args], src=self.leader, group=self.dp_group)
                composite_args = args
            else:
                buf = [None]
                dist.broadcast_object_list(buf, src=self.leader, group=self.dp_group)
                composite_args = buf[0]
        else:
            if args:
                composite_args = args
            else:
                composite_args = self._retrieve_recv_activations(fwd_chunk_id)

        if composite_args is None or len(composite_args) == 0:
            raise RuntimeError(
                f"[rank{dist.get_rank()}] Empty composite_args after dispatch at "
                f"stage={self.stage_index}, fwd_chunk_id={fwd_chunk_id}, "
                f"is_first={self.prev_group is None}."
            )

        composite_kwargs = kwargs or {}

        if (
            pack_size > 1
            and composite_args
            and isinstance(composite_args[0], torch.Tensor)
        ):
            mb_bs = composite_args[0].shape[0] // pack_size
            args_for_val = tuple(
                (t[:mb_bs] if isinstance(t, torch.Tensor) else t)
                for t in composite_args
            )
            kwargs_for_val = {
                k: (v[:mb_bs] if isinstance(v, torch.Tensor) else v)
                for k, v in composite_kwargs.items()
            }
        else:
            args_for_val = composite_args
            kwargs_for_val = composite_kwargs

        self._validate_fwd_input(args_for_val, kwargs_for_val)

        # 1) 本地前向
        try:
            output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)
        except Exception as e:
            exc_msg = f"""
            {self.log_prefix} failed to run forward:
            args: {map_debug_info(composite_args)}
            kwargs: {map_debug_info(composite_kwargs)}
            """
            raise RuntimeError(exc_msg) from e

        # 2) 末端融合（可选）
        # >>>> 在这里插入融合：只依赖 stage 与当前 output，避免无关依赖
        if self.fuse_stage and (self.fuse_function is not None):
            try:
                fused_output = self.fuse_function(self, output)
            except Exception as e:
                exc_msg = f"""
                {self.log_prefix} fuse_function failed at forward:
                raw_output: {map_debug_info(output)}
                """
                raise RuntimeError(exc_msg) from e

            if fused_output is None:
                raise RuntimeError(f"{self.log_prefix} fuse_function returned None.")
            output = fused_output  # 用融合后的结果替换原输出，保留计算图

        # 3) 规范化输出，供缓存与校验（以融合后的 output 为准）
        output_tuple = _normalize_model_output_as_tuple(output)

        # 4) 仅最后一段会合并最终输出：应收集融合后的结果
        if self.is_last:
            self.output_chunks.append(output)

        # 5) 保存前向缓存（用于 backward）
        flat_args = flatten_args(composite_args)
        flat_kwargs = flatten_args(composite_kwargs)
        flatten_input_tensors = flat_args + flat_kwargs
        self.fwd_cache[fwd_chunk_id] = (
            output_tuple,            # stage_output（已融合，并规范为 tuple）
            flatten_input_tensors,   # input_values
        )

        logger.debug(
            "%s Forwarded chunk %s, outputs: %s",
            self.log_prefix,
            fwd_chunk_id,
            map_debug_info(output),
        )

        # 6) 抽样校验（依旧基于融合后的 output_tuple）
        if pack_size > 1 and isinstance(output_tuple, tuple) and len(output_tuple):
            outputs_for_val = tuple(
                t[:mb_bs] if isinstance(t, torch.Tensor) else t
                for t in output_tuple
            )
        elif pack_size > 1 and isinstance(output_tuple, torch.Tensor):
            outputs_for_val = output_tuple[:mb_bs]
        else:
            outputs_for_val = output_tuple

        self._validate_fwd_outputs(outputs_for_val)

        # 7) 返回融合后的原始对象，而非 tuple
        return output

    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward=False,
        retain_graph_for_packed_mbs: bool = False,
    ):
        self._check_chunk_id(bwd_chunk_id)

        (stage_output, input_values,) = self.fwd_cache.pop(bwd_chunk_id)

        # Compute backward
        if self.is_last:
            bwd_kwargs = {
                "stage_output": loss,
                "output_grads": None,
                "input_values": input_values,
            }
            split_out = None  # >>> NEW: 最后一个 stage 一般不需要拆分
        else:
            # Otherwise, receive gradients from next stage
            grads_output = self._retrieve_recv_grads(bwd_chunk_id)

            # Average upstream grads across the next stage's DP replicas
            grads_output = self._avg_next_stage_grads(grads_output)

            # >>> NEW: 在这里调用 split_function（仅当是融合阶段且用户提供了函数）
            split_out = None
            if getattr(self, "fuse_stage", False) and (getattr(self, "split_function", None) is not None):
                try:
                    split_out = self.split_function(self, stage_output, grads_output)
                except Exception as e:
                    raise RuntimeError(
                        f"{self.log_prefix} split_function failed at backward (chunk {bwd_chunk_id})."
                    ) from e

                if split_out is None or ("grads_output_local" not in split_out):
                    raise RuntimeError(f"{self.log_prefix} split_function must return 'grads_output_local'.")

                # 用用户提供的 grads_output_local 驱动本地 autograd
                grads_output = split_out["grads_output_local"]

                # 若用户要求保留图，转给后面的 backward 路径
                if bool(split_out.get("retain_graph", False)):
                    retain_graph_for_packed_mbs = True

            # 继续按原逻辑组装 bwd_kwargs
            bwd_kwargs = {
                "stage_output": stage_output,
                "output_grads": grads_output,
                "input_values": input_values,
            }

        grads_input: tuple[Optional[torch.Tensor], ...] = ()

        # Custom backward function（保持不变）
        if self.dw_builder:
            grads_input, _ = self.backward_maybe_with_nosync(
                "full",
                bwd_kwargs,
                last_backward=last_backward,
                retain_graph_for_packed_mbs=retain_graph_for_packed_mbs
            )
            if full_backward:
                self.dw_builder()()
            else:
                self.dw_runner[bwd_chunk_id] = self.dw_builder()
        else:
            if full_backward:
                grads_input, _ = self.backward_maybe_with_nosync(
                    "full", bwd_kwargs, last_backward=last_backward,
                    retain_graph_for_packed_mbs=retain_graph_for_packed_mbs
                )
            else:
                param_groups: list[dict[str, Any]] | None = None
                if not self.is_first:
                    if isinstance(bwd_kwargs["stage_output"], torch.Tensor):
                        bwd_kwargs["stage_output"] = (bwd_kwargs["stage_output"],)
                    grads_input, param_groups = self.backward_maybe_with_nosync(
                        "input", bwd_kwargs, last_backward=last_backward
                    )
                self.backward_state[bwd_chunk_id] = (
                    bwd_kwargs["input_values"],
                    param_groups,
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                )
                self.dw_runner[bwd_chunk_id] = lambda: None

        # >>> NEW: 若 split_function 给了 grads_input_overrides，则对 autograd 产出的 grads_input 做按位覆盖
        #          这样“回给上游的各模态梯度”就能复用你现有的发送通道（args_recv_info/grad_send_info）而无需改通信层。
        if split_out is not None:
            overrides = split_out.get("grads_input_overrides", None)
            if overrides is not None:
                if not isinstance(overrides, (list, tuple)):
                    raise RuntimeError(f"{self.log_prefix} grads_input_overrides must be a list/tuple.")
                if len(overrides) != len(grads_input):
                    raise RuntimeError(
                        f"{self.log_prefix} grads_input_overrides length {len(overrides)} "
                        f"!= grads_input length {len(grads_input)}."
                    )
                grads_input = tuple(
                    (overrides[i] if overrides[i] is not None else grads_input[i])
                    for i in range(len(grads_input))
                )

        # 原逻辑：为发送到 prev stage 做掩码与占位
        if self.grad_send_info is None:
            self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])

        grads_input = tuple(
            (g if dst is not None and isinstance(g, torch.Tensor) else None)
            for g, dst in zip(grads_input, self.grad_send_info)
        )

        self.bwd_cache[bwd_chunk_id] = grads_input

        if self.is_last and not self.is_first:
            for t in stage_output:
                if not t._is_view():
                    t.detach_()

        logger.debug("%s Backwarded chunk %s", self.log_prefix, bwd_chunk_id)
        
    def _to_global(self, peer_rank: int) -> int:
        if self.group is None:
            return peer_rank
        return dist.get_global_rank(self.group, peer_rank)

    def _append_ops(self, ops: list[dist.P2POp], tensors: tuple[torch.Tensor, ...], peer_rank: int, is_send: bool):
        g = self._to_global(peer_rank)
        for t in tensors:
            if is_send:
                ops.append(dist.P2POp(dist.isend, t, g, self.group))
            else:
                ops.append(dist.P2POp(dist.irecv, t, g, self.group))
                
    def get_fwd_recv_ops_modal(
        self,
        fwd_chunk_id: int,
        routes: dict[str, int]  # {"vision": src_rank_local, "audio": src_rank_local, ...}
    ) -> list[dist.P2POp]:
        """
        为同一 microbatch 的多个模态生成 irecv ops。
        routes 中的 rank 是 group 内 rank；函数内部会转 global rank。
        """
        ops: list[dist.P2POp] = []

        modal_infos = self.mm_args_recv_info.get(fwd_chunk_id, {})
        for modal, src_rank in routes.items():
            infos = modal_infos.get(modal, ())
            for info in infos:
                if not isinstance(info, _RecvInfo):
                    continue
                peer_global_rank = self._to_global(src_rank)
                ops.append(dist.P2POp(dist.irecv, info.buffer, peer_global_rank, self.group))
        return ops
    
    def get_fwd_send_ops_modal(
        self,
        fwd_chunk_id: int,
        routes: dict[str, int]  # {"vision": dst_rank_local, ...}
    ) -> list[dist.P2POp]:
        """
        发送尚未融合的模态 payload（很少用；通常融合后才走标准 act 通道）。
        若需要把某模态先转发到另一 rank 做预处理，可以用这个。
        """
        ops: list[dist.P2POp] = []
        modal_cache = self.mm_fwd_cache.get(fwd_chunk_id, {})
        for modal, dst_rank in routes.items():
            payload = modal_cache.get(modal, None)
            if payload is None:
                continue
            g = self._to_global(dst_rank)
            for t in payload:
                ops.append(dist.P2POp(dist.isend, t, g, self.group))
        return ops

    def set_modal_grad_recv_infos(self, bwd_chunk_id: int, modal: str, recv_infos: tuple[_RecvInfo, ...]) -> None:
        self.mm_grad_recv_info[bwd_chunk_id][modal] = recv_infos

    def stash_modal_bwd_grads(self, bwd_chunk_id: int, modal: str, grads: tuple[torch.Tensor, ...]) -> None:
        # 例如 grads = (grad_vision_tokens, grad_vision_lens_optional, ...)
        self.mm_bwd_cache[bwd_chunk_id][modal] = grads
        
    def get_bwd_recv_ops_modal(
        self,
        bwd_chunk_id: int,
        routes: dict[str, int]  # {"vision": src_rank_local, ...}
    ) -> list[dist.P2POp]:
        if not self.has_backward or self.is_last:
            return []
        ops: list[dist.P2POp] = []
        modal_infos = self.mm_grad_recv_info.get(bwd_chunk_id, {})
        for modal, src_rank in routes.items():
            infos = modal_infos.get(modal, ())
            for info in infos:
                if not isinstance(info, _RecvInfo):
                    continue
                ops.append(dist.P2POp(dist.irecv, info.buffer, self._to_global(src_rank), self.group))
        return ops
    
    def get_bwd_send_ops_modal(
        self,
        bwd_chunk_id: int,
        routes: dict[str, int]  # {"vision": dst_rank_local, "audio": dst_rank_local, ...}
    ) -> list[dist.P2POp]:
        """
        发送拆分后的各模态梯度。与现有 get_bwd_send_ops 并存：
        - get_bwd_send_ops：负责“主通道”的 grads_input（对应融合后三元组上一段）
        - 本函数：负责“模态通道”的 grads（对应各 encoder 上一段）
        """
        if not self.has_backward or self.is_first:
            return []

        ops: list[dist.P2POp] = []
        modal_cache = self.mm_bwd_cache.pop(bwd_chunk_id, {})  # 用完即清
        for modal, dst_rank in routes.items():
            grads_pack = modal_cache.get(modal, None)
            if grads_pack is None:
                continue
            for g in grads_pack:
                if isinstance(g, torch.Tensor):
                    # 仅发送浮点/复数梯度，跳过整型长度等
                    if not (g.is_floating_point() or torch.is_complex(g)):
                        # 允许存在非浮点（如 lens）；忽略即可
                        continue
                    ops.append(dist.P2POp(dist.isend, g, self._to_global(dst_rank), self.group))
                else:
                    raise RuntimeError(
                        f"[{self.stage_index}] modal={modal} expects tensor grad, got {type(g)}"
                    )
        return ops
    
    
    #下一步处理forwad和backward中cache或mmcache的管理





