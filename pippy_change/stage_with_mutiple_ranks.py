# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
from abc import ABC, abstractmethod
from typing import Any, Callable, cast, Optional, Union

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.fx.node import Argument, map_aggregate
from torch.nn.parallel import DistributedDataParallel
from torch.utils._pytree import tree_map_only

from torch.distributed.pipelining.stage import PipelineStage, InputInfo, _RecvInfo

logger = logging.getLogger(__name__)

class PipelineStage_with_mutiple_ranks(PipelineStage):
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
    
    def get_fwd_recv_ops(self, fwd_chunk_id: int, rank: int, dest_rank: int) -> list[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the input arguments
        for this stage.
        """
        recv_infos: tuple[InputInfo, ...] = self.args_recv_info[fwd_chunk_id]

        return self._get_recv_ops(recv_infos, rank, dest_rank)

    def get_bwd_recv_ops(self, bwd_chunk_id: int, rank: int, dest_rank: int) -> list[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the gradients
        for this stage.
        """
        if not self.has_backward or self.is_last:
            return []

        recv_infos = self.grad_recv_info[bwd_chunk_id]
        return self._get_recv_ops(recv_infos, rank, dest_rank)

    def get_fwd_send_ops(self, fwd_chunk_id: int, rank: int, dest_rank: int) -> list[dist.P2POp]:
        """
        Get the activation send ops for current stage's forward.
        """
        output_tuple, _ = self.fwd_cache[fwd_chunk_id]

        ops: list[dist.P2POp] = []

        for idx, out in enumerate(output_tuple):
            dst_stages = self.act_send_info[idx]
            for dst in dst_stages:
                if dst is None:
                    continue
                logger.debug(
                    "%s Sending tensor to Stage %s: %s",
                    self.log_prefix,
                    dst,
                    out.size(),
                )
                peer_rank = dest_rank
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )
                ops.append(dist.P2POp(dist.isend, out, peer_global_rank, self.group))

        return ops

    def get_bwd_send_ops(self, bwd_chunk_id: int, rank: int, dest_rank: int) -> list[dist.P2POp]:
        """
        Get the gradient send ops for current stage's backward.
        """
        self._check_chunk_id(bwd_chunk_id)

        if not self.has_backward or self.is_first:
            return []

        # Create bwd send infra lazily
        if self.grad_send_info is None:
            # Send info for input grads during backward:
            # List of destinations corresponding to input grads
            # Can be None if an input has no grad
            # `grad_send_info` is a mirror of `args_recv_info`
            self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])

        ops: list[dist.P2POp] = []
        grads_input = self.bwd_cache.pop(bwd_chunk_id)
        for grad, grad_recv_stage in zip(grads_input, self.grad_send_info):
            if isinstance(grad, torch.Tensor) and grad_recv_stage is not None:
                logger.debug(
                    "%s Sending gradient to Stage %s: %s",
                    self.log_prefix,
                    grad_recv_stage,
                    grad.size(),
                )
                peer_rank = dest_rank
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )
                ops.append(dist.P2POp(dist.isend, grad, peer_global_rank, self.group))
            else:
                if not (grad is None and grad_recv_stage is None):
                    raise RuntimeError(
                        f"[{self.stage_index}] for chunk {bwd_chunk_id} has gradients {grad} "
                        f"and is expecting to send gradients to stage {grad_recv_stage}"
                    )
        return ops