"""Temporary DP transport rows, never extra rollout or PPO training samples."""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Temporary DP transport rows, never extra rollout or PPO training samples.
# Extension point: PPOTrainer step, advantage, actor-update, and FSDP reduction hooks

# DYAD-AGENT-STEPS: legacy DataProto transport, distinct from statistical occurrences.
import os
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from verl import DataProto
from verl.utils.seqlen_balancing import calculate_workload

REAL_ROW_MASK = "dyad_real_row_mask"


def exact_batch_padding_enabled() -> bool:
    """The public trainer validates the supported configuration before dispatch."""
    return os.environ.get("DYAD_EXACT_BATCH_PADDING", "0") == "1"


def transport_divisor(dp_size: int, micro_batch_size: int = 1) -> int:
    """Fixed microbatches require complete microbatches on every DP rank."""
    if dp_size <= 0 or micro_batch_size <= 0:
        raise ValueError("DP size and transport microbatch size must be positive")
    return dp_size * micro_batch_size


def _copy_rows(batch: DataProto, indices: list[int], real_rows: list[bool]) -> DataProto:
    # DataProto indexing copies tensors but shares metadata and Python objects.
    # Copy those explicitly: conversion/worker processing must not mutate the
    # canonical batch, nor share mutable context between a dummy and a real row.
    result = batch.select_idxs(indices)
    result.meta_info = deepcopy(batch.meta_info)
    if "global_token_num" in result.meta_info:
        # MFU describes actual forward work, including valid dummy contexts.
        result.meta_info["global_token_num"] = result.batch["attention_mask"].flatten(1).sum(-1).tolist()
    for key, values in result.non_tensor_batch.items():
        if values.dtype.hasobject:
            copied = np.empty(values.shape, dtype=object)
            for index in np.ndindex(values.shape):
                copied[index] = deepcopy(values[index])
            result.non_tensor_batch[key] = copied
        else:
            result.non_tensor_batch[key] = values.copy()
    device = result.batch["attention_mask"].device
    real_mask = torch.tensor(real_rows, dtype=torch.bool, device=device)
    result.batch[REAL_ROW_MASK] = real_mask
    if "seq_mask" not in result.batch:
        result.batch["seq_mask"] = torch.ones_like(result.batch["response_mask"], dtype=torch.bool)
    # Keep attention/input/candidate/encoder context valid for the forward pass.
    # Only objective masks and targets are zeroed; loss normalization is unchanged.
    for key in ("response_mask", "seq_mask", "advantages", "returns"):
        if key in result.batch:
            result.batch[key][~real_mask] = 0
    return result


@dataclass(frozen=True)
class InferenceTransport:
    batch: DataProto
    real_size: int

    def restore(self, output: DataProto) -> DataProto:
        """Remove temporary rows before the result can be merged into rollouts."""
        if len(output) != len(self.batch):
            raise ValueError("Inference output row count does not match transport batch")
        if len(output) == self.real_size:
            return output
        return output.select_idxs(list(range(self.real_size)))


def pad_inference_batch(batch: DataProto, dp_size: int, micro_batch_size: int = 1) -> InferenceTransport:
    """Append valid masked copies and preserve canonical order for round trips."""
    divisor = transport_divisor(dp_size, micro_batch_size)
    size = len(batch)
    if size == 0:
        raise ValueError("Cannot transport an empty batch")
    padding = (-size) % divisor
    if not padding:
        return InferenceTransport(batch, size)
    lengths = batch.batch["attention_mask"].flatten(1).sum(-1)
    source = int(lengths.argmin().item())
    padded = _copy_rows(batch, list(range(size)) + [source] * padding, [True] * size + [False] * padding)
    return InferenceTransport(padded, size)


@dataclass(frozen=True)
class ActorTransport:
    batch: DataProto
    logical_mini_batch_size: int
    physical_mini_batch_size: int


def pad_actor_batch(
    batch: DataProto, logical_mini_batch_size: int, dp_size: int, micro_batch_size: int = 1
) -> ActorTransport:
    """Lay out [rank][logical minibatch][real rows, dummy rows] for dispatch.

    Equal contiguous DP dispatch and the unshuffled TrainingWorker iterator then
    produce exactly the original logical minibatches, including their boundaries.
    """
    divisor = transport_divisor(dp_size, micro_batch_size)
    if logical_mini_batch_size <= 0 or not len(batch) or len(batch) % logical_mini_batch_size:
        raise ValueError("Actor transport requires complete nonempty logical PPO minibatches")
    physical_size = ((logical_mini_batch_size + divisor - 1) // divisor) * divisor
    if physical_size == logical_mini_batch_size:
        return ActorTransport(batch, logical_mini_batch_size, physical_size)

    local_size = physical_size // dp_size
    base, remainder = divmod(logical_mini_batch_size, dp_size)
    quotas = [base + (rank < remainder) for rank in range(dp_size)]
    lengths = batch.batch["attention_mask"].flatten(1).sum(-1)
    workloads = calculate_workload(lengths).tolist()
    rank_indices = [[] for _ in range(dp_size)]
    rank_masks = [[] for _ in range(dp_size)]
    for start in range(0, len(batch), logical_mini_batch_size):
        logical_indices = list(range(start, start + logical_mini_batch_size))
        partitions = [[] for _ in range(dp_size)]
        loads = [0] * dp_size
        # Cardinality-constrained longest-first placement balances token work
        # deterministically without exchanging samples between logical updates.
        for index in sorted(logical_indices, key=lambda index: (-workloads[index], index)):
            eligible = [rank for rank in range(dp_size) if len(partitions[rank]) < quotas[rank]]
            rank = min(eligible, key=lambda rank: (loads[rank], rank))
            partitions[rank].append(index)
            loads[rank] += workloads[index]
        source = min(logical_indices, key=lambda index: (workloads[index], index))
        for rank, partition in enumerate(partitions):
            partition.sort()
            padding = local_size - len(partition)
            rank_indices[rank].extend(partition + [source] * padding)
            rank_masks[rank].extend([True] * len(partition) + [False] * padding)
    indices = [index for partition in rank_indices for index in partition]
    masks = [real for partition in rank_masks for real in partition]
    return ActorTransport(_copy_rows(batch, indices, masks), logical_mini_batch_size, physical_size)
