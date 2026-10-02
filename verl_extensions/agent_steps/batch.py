"""Reference statistical occurrences, separate from TransferQueue transport padding."""

# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Reference statistical occurrences, separate from TransferQueue transport padding.
# Extension point: PPOTrainer step, advantage, actor-update, and FSDP reduction hooks
# DYAD-AGENT-STEPS: occurrence preparation belongs to the shared training pipeline.
from __future__ import annotations

import copy
import math
import uuid

import numpy as np

from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from verl.utils.transferqueue_utils import KVBatchMeta, tq


def reference_batch_multiple(config, dp_size: int) -> int:
    """Match the reference text actor/ref/rollout microbatch divisibility."""
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    reference = config.actor_rollout_ref.ref
    sizes = [actor.get("ppo_micro_batch_size_per_gpu"), rollout.get("log_prob_micro_batch_size_per_gpu")]
    if actor.get("use_kl_loss", False) or config.algorithm.get("use_kl_in_reward", False):
        sizes.append(reference.get("log_prob_micro_batch_size_per_gpu"))
    # Dynamic batching has no fixed microbatch divisor in the newer engine.
    # It still needs equal row counts for DP dispatch.
    multiples = [dp_size]
    for size in sizes:
        if size is not None:
            if isinstance(size, bool) or int(size) != size or size <= 0:
                raise ValueError("Reference step resampling requires positive microbatch sizes")
            multiples.append(int(size) * dp_size)
    return math.lcm(*multiples)


def prepare_step_occurrences(batch, config, dp_size: int, seed: int):
    """Copy real rows without pretending the copies are extra environment decisions.

    Advantage code reconstructs raw complete trajectories from source keys before
    applying this occurrence selection, so copies never enter discounted returns.
    Stable occurrence order survives subsequent DP balancing and similarity grouping.
    """
    settings = config.algorithm.get("step_rollout", {})
    mode = settings.get("resampling", "reference_copy")
    if mode not in ("reference_copy", "none"):
        raise ValueError(f"Unsupported step resampling: {mode}")
    if not len(batch) or any(tag.get("is_padding", False) for tag in batch.tags):
        raise ValueError("Step resampling requires nonempty original rollout rows before transport padding")
    if len(set(batch.keys)) != len(batch):
        raise ValueError("Original step transport keys must be unique")
    # The reference flattens attempts, source prompts, sessions, then decisions.
    # Async completion order must not change copies or optimizer partitions.
    group_order = {}
    order_groups = {}
    ordering = []
    for index, (key, tag) in enumerate(zip(batch.keys, batch.tags, strict=True)):
        parts = key.rsplit("_", 2)
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
            raise ValueError(f"Invalid original step key: {key}")
        if settings.get("enabled", False):
            prompt_order = tag.get("step_prompt_order")
            attempt = tag.get("dynamic_sampling_attempt", 0)
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (prompt_order, attempt)
            ):
                raise ValueError(f"Shared step rows require a nonnegative source prompt order: {key}")
            order = (attempt, prompt_order)
            if group_order.setdefault(parts[0], order) != order:
                raise ValueError(f"Inconsistent source prompt order within group: {parts[0]}")
            if order_groups.setdefault(order, parts[0]) != parts[0]:
                raise ValueError(f"Duplicate source prompt order across groups: {order}")
        else:
            order = group_order.setdefault(parts[0], (0, len(group_order)))
        ordering.append((*order, int(parts[1]), int(parts[2]), index))
    batch = batch.select_keys([batch.keys[item[-1]] for item in sorted(ordering)])
    tags = copy.deepcopy(batch.tags)
    for ordinal, (key, tag) in enumerate(zip(batch.keys, tags, strict=True)):
        tag.update(step_source_key=key, step_occurrence=ordinal, is_statistical_copy=False)
    # KVBatchMeta tags are local snapshots; later fields-only puts reload TQ tags.
    # Persist originals even when no statistical copies are needed.
    tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, tags=tags)
    count = len(batch)
    multiple = reference_batch_multiple(config, dp_size) if mode == "reference_copy" else 1
    to_add = (-count) % multiple
    copy_keys, copy_tags = [], []
    if to_add:
        from verl_extensions.agent_steps.advantages import reference_sample_indices

        indices = reference_sample_indices(count, multiple, np.random.RandomState(seed))[count:].tolist()
        source = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id)
        rows = []
        for ordinal, index in enumerate(indices, start=count):
            key = batch.keys[index]
            uid, _, step = key.rsplit("_", 2)
            copy_keys.append(f"{uid}_copy{uuid.uuid4().hex}_{step}")
            row = copy.deepcopy(source[index])
            rows.append(row)
            tag = copy.deepcopy(tags[index])
            tag.update(step_occurrence=ordinal, is_statistical_copy=True)
            copy_tags.append(tag)
        tq.kv_batch_put(
            keys=copy_keys, partition_id=batch.partition_id, fields=list_of_dict_to_tensordict(rows), tags=copy_tags
        )
    result = KVBatchMeta(
        keys=batch.keys + copy_keys,
        tags=tags + copy_tags,
        partition_id=batch.partition_id,
        fields=batch.fields,
        extra_info=batch.extra_info,
    )
    return result, {
        "step/original_decisions": count,
        "step/statistical_copies": to_add,
        "step/training_occurrences": count + to_add,
    }
