"""Real step mini-batches with separate zero-loss DP transport padding."""

# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Real step mini-batches with separate zero-loss DP transport padding.
# Extension point: PPOTrainer step, advantage, actor-update, and FSDP reduction hooks
# DYAD-AGENT-STEPS: common step dispatch for text and expanded-action policies.
from __future__ import annotations

from contextlib import ExitStack, contextmanager

import torch
from tensordict import TensorDict

from verl.trainer.ppo.padding_utils import upsample_batch_to_divisible_size
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions
from verl.utils.transferqueue_utils import KVBatchMeta, tq


def _normalize_padding_fields(batch, padding_keys):
    # Upstream pads before advantage/log-prob computation. Here those fields
    # already exist, so its copied tensors need the new one-response-token shape.
    fields = [
        "old_log_probs",
        "ref_log_prob",
        "advantages",
        "returns",
        "values",
        "entropy",
        "token_level_scores",
        "token_level_rewards",
        "rollout_is_weights",
    ]
    source = tq.kv_batch_get(keys=batch.keys[:1], partition_id=batch.partition_id, select_fields=fields)
    zeros = {}
    for field in fields:
        if field in source:
            value = source[field][0]
            zeros[field] = torch.nested.as_nested_tensor(
                [torch.zeros(1, dtype=value.dtype, device=value.device) for _ in padding_keys],
                layout=torch.jagged,
            )
    if zeros:
        tq.kv_batch_put(
            keys=padding_keys, partition_id=batch.partition_id, fields=TensorDict(zeros, batch_size=len(padding_keys))
        )


@contextmanager
def _padded_training_minibatch(batch, dp_size, eos_token_id):
    """Own only this mini's temporary padding, including on worker failures."""
    real_keys = set(batch.keys)
    padded = upsample_batch_to_divisible_size(batch, dp_size, eos_token_id)
    padding_keys = [key for key in padded.keys if key not in real_keys]
    try:
        if padding_keys:
            _normalize_padding_fields(batch, padding_keys)
        yield padded
    finally:
        # The outer rollout cleanup does not know these newly allocated keys.
        # Never clear real rows or the rollout's pre-log-prob padding here.
        if padding_keys:
            tq.kv_clear(keys=padding_keys, partition_id=batch.partition_id)


@contextmanager
def _actor_update_residency(trainer, update_count):
    """DYAD-PERF: offload once per update phase, including failed RPCs.

    Mini-batch membership and optimizer/cache/scheduler boundaries stay unchanged.
    No rollout or reference inference runs inside this controller-owned scope.
    """
    actor = trainer.config.actor_rollout_ref.actor
    engine = actor.get("fsdp_config", {})
    enabled = (
        update_count > 1
        and actor.get("strategy") in {"fsdp", "fsdp2"}
        and (engine.get("param_offload", False) or engine.get("optimizer_offload", False))
    )
    if not enabled:
        yield False
        return
    try:
        # Include loading in try: some ranks may load before another rank fails.
        trainer.actor_rollout_wg.set_actor_update_residency(True)
        yield True
    except BaseException as error:
        # DYAD-CLEANUP: an OOM may leave FSDP unable to offload its shard. Still
        # attempt cleanup, but do not replace the actionable training failure
        # with a secondary shard-pointer assertion from that cleanup.
        try:
            trainer.actor_rollout_wg.set_actor_update_residency(False)
        except Exception as cleanup_error:
            error.add_note(f"Actor residency cleanup also failed: {cleanup_error!r}")
        raise
    else:
        trainer.actor_rollout_wg.set_actor_update_residency(False)


def update_step_minibatches(trainer, batch, extra_info):
    """Update all real rows against one rollout anchor, then tick the scheduler once."""
    rows = [(key, tag) for key, tag in zip(batch.keys, batch.tags, strict=True) if not tag.get("is_padding", False)]
    # Recover the complete statistical occurrence stream before reproducing the
    # reference's full-batch rank assignment. Log-prob transport order is separate.
    if rows and all("step_occurrence" in tag for _, tag in rows):
        rows.sort(key=lambda row: row[1]["step_occurrence"])
    real_keys = [key for key, _ in rows]
    if not real_keys:
        raise ValueError("Step mini-batches require at least one real rollout row")
    mini_size = extra_info["mini_batch_size"]
    epochs = extra_info["epochs"]
    if mini_size <= 0 or epochs <= 0:
        raise ValueError("Step mini-batch size and PPO epochs must be positive")
    if not trainer.config.get("algorithm", {}).get("step_rollout", {}).get("enabled", False):
        raise ValueError("Step mini-batches require shared step protocol v2")
    dp_size = trainer._get_actor_dp_size()
    actor = trainer.config.actor_rollout_ref.actor
    if dp_size < 1 or len(real_keys) % dp_size or mini_size % dp_size:
        raise ValueError("Reference step batches and mini-batches must be divisible by actor DP size")
    if extra_info["dataloader_kwargs"].get("shuffle", False):
        raise ValueError("Reference step mini-batches require shuffle=False")
    local_mini_size = mini_size // dp_size
    padding_multiple = 1
    if not actor.get("use_dynamic_bsz", True):
        padding_multiple = actor.ppo_micro_batch_size_per_gpu
        if padding_multiple is None or padding_multiple <= 0 or local_mini_size % padding_multiple:
            raise ValueError("Reference rank-local mini-batches must be divisible by the micro-batch size")
    rank_size = len(real_keys) // dp_size
    if trainer.config.trainer.get("balance_batch", True):
        # The reference balances raw full sequence lengths, not estimated FLOPs.
        partitions = get_seqlen_balanced_partitions(
            [tag["seq_len"] for _, tag in rows],
            k_partitions=dp_size,
            equal_size=True,
        )
    else:
        partitions = [list(range(start, start + rank_size)) for start in range(0, len(real_keys), rank_size)]
    rank_keys = [[real_keys[index] for index in partition] for partition in partitions]
    metrics = {}
    optimizer_updates = 0
    padding_rows = 0
    update_count = epochs * ((rank_size + local_mini_size - 1) // local_mini_size)
    with _actor_update_residency(trainer, update_count) as resident:
        for epoch in range(epochs):
            for start in range(0, rank_size, local_mini_size):
                # Dispatch chunks rank-major, exactly as whole-batch dispatch followed
                # by rank-local TensorDict.split in the reference actor. Never rebalance
                # these minis: token-mean micro losses depend on their membership.
                with ExitStack() as stack:
                    rank_minis = [
                        stack.enter_context(
                            _padded_training_minibatch(
                                batch.select_keys(keys[start : start + local_mini_size]),
                                padding_multiple,
                                trainer.tokenizer.eos_token_id,
                            )
                        )
                        for keys in rank_keys
                    ]
                    real_size = min(local_mini_size, rank_size - start) * dp_size
                    mini = KVBatchMeta.concat(rank_minis)
                    mini.extra_info = dict(extra_info)
                    mini.extra_info.update(
                        global_batch_size=real_size,
                        mini_batch_size=len(mini),
                        epochs=1,
                        dataloader_kwargs={"shuffle": False},
                        update_lr_scheduler=epoch == epochs - 1 and start + local_mini_size >= rank_size,
                        update_dyad_cache=epoch == epochs - 1 and start + local_mini_size >= rank_size,
                    )
                    if resident:
                        mini.extra_info["disable_auto_offload"] = True
                    output = trainer.actor_rollout_wg.update_actor(mini)
                    optimizer_updates += 1
                    padding_rows += len(mini) - real_size
                    for name, value in output["metrics"].items():
                        metrics.setdefault(name, []).extend(value if isinstance(value, list) else [value])
    metrics.update(real_step_rows=len(real_keys), optimizer_updates=optimizer_updates, step_padding_rows=padding_rows)
    return metrics
