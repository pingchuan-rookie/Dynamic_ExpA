"""Reference microbatch PPO reduction on the current engine interface.

Fixed partitions divide microbatch losses by the configured accumulation count.
Dynamic partitions weight those losses by real rows over the configured mini.
Both retain the nominal denominator on tails, as the reference text actor does.
Transport-only micros have zero weight; genuine empty decisions remain real rows
with zero gradient.
"""

# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Reference microbatch PPO reduction on the current engine interface.
# Extension point: PPOTrainer step, advantage, actor-update, and FSDP reduction hooks
# DYAD-AGENT-STEPS: text and Dyad PPO use the same training reduction contract.
from __future__ import annotations

import copy
from numbers import Integral

import torch

from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric

REDUCTION_KEY = "environment_step_loss_reduction"
PADDING_KEY = "environment_step_is_padding"
WEIGHT_KEY = "environment_step_micro_weight"
ROWS_KEY = "environment_step_micro_rows"
RESPONSE_LENGTH_KEY = "environment_step_response_length"
NOMINAL_MINI_KEY = "environment_step_nominal_mini_batch_size"


def uses_reference_reduction(data):
    mode = tu.get_non_tensor_data(data, REDUCTION_KEY, "global")
    if mode not in ("global", "reference_microbatch"):
        raise ValueError(f"Unsupported environment-step loss reduction: {mode}")
    return mode == "reference_microbatch"


def real_row_count(data):
    if PADDING_KEY not in data:
        raise ValueError("Reference microbatch reduction requires explicit environment_step_is_padding")
    padding = data[PADDING_KEY]
    if padding.is_nested or padding.numel() != len(data) or padding.dtype != torch.bool:
        raise ValueError("environment_step_is_padding must be one boolean per row")
    return int((~padding).sum().item())


def _integer(value, name, *, minimum=1):
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def microbatch_weights(row_counts, *, dynamic, dp_size, nominal_mini_size, micro_batch_size=None):
    """Local backward weights; DP averaging cancels ``dp_size``, not the tail."""
    if not isinstance(dynamic, bool):
        raise ValueError("use_dynamic_bsz must be boolean")
    dp_size = _integer(dp_size, "dp_size")
    nominal_mini_size = _integer(nominal_mini_size, NOMINAL_MINI_KEY)
    counts = [_integer(count, "real micro rows", minimum=0) for count in row_counts]
    if nominal_mini_size % dp_size:
        raise ValueError("Nominal mini-batch size must be divisible by DP size")
    if not dynamic:
        micro_batch_size = _integer(micro_batch_size, "micro_batch_size_per_gpu")
        if nominal_mini_size % (dp_size * micro_batch_size):
            raise ValueError("Nominal rank-local mini must be divisible by the fixed micro size")
        if any(count > micro_batch_size for count in counts):
            raise ValueError("Real micro rows exceed the configured fixed micro size")
    if sum(counts) > nominal_mini_size // dp_size:
        raise ValueError("Real rank-local rows exceed the nominal mini size")
    return [dp_size * (count if dynamic else micro_batch_size * int(count > 0)) / nominal_mini_size for count in counts]


def prepare_reference_microbatches(data, micro_batches, *, dp_group=None, device=None):
    """Attach nominal-budget reduction metadata after the engine fixes the partition."""
    if not uses_reference_reduction(data):
        return
    # DYAD-STEP: actual global_batch_size and transport mini size shrink on tails.
    # Neither may stand in for the configured reference actor's mini budget.
    nominal = tu.get_non_tensor_data(data, NOMINAL_MINI_KEY, None)
    dynamic = tu.get_non_tensor_data(data, "use_dynamic_bsz", None)
    dp_size = _integer(tu.get_non_tensor_data(data, "dp_size", None), "dp_size")
    micro_size = tu.get_non_tensor_data(data, "micro_batch_size_per_gpu", None)
    counts = [real_row_count(micro) for micro in micro_batches]
    weights = microbatch_weights(
        counts, dynamic=dynamic, dp_size=dp_size, nominal_mini_size=nominal, micro_batch_size=micro_size
    )
    if sum(len(micro) for micro in micro_batches) != len(data) or sum(counts) != real_row_count(data):
        raise ValueError("Reference microbatch partition does not preserve row counts")
    if not dynamic and any(len(micro) > _integer(micro_size, "micro_batch_size_per_gpu") for micro in micro_batches):
        raise ValueError("Physical micro rows exceed the configured fixed micro size")
    if torch.distributed.is_initialized():
        if torch.distributed.get_world_size(group=dp_group) != dp_size:
            raise ValueError("Reference reduction DP size disagrees with the process group")
    elif dp_size != 1:
        raise RuntimeError("Distributed reference reduction requires an initialized process group")
    for micro, count, weight in zip(micro_batches, counts, weights, strict=True):
        tu.assign_non_tensor(micro, **{ROWS_KEY: count, WEIGHT_KEY: weight})


def reference_loss_config(config, data):
    """Return an isolated loss config, or None for the unchanged global path.

    All Dyad surrogates retain the shared response-token denominator, including
    the encoder's action-only numerator. Mask-specific normalization is not an
    incidental side effect of switching the accumulation protocol.
    """
    if not uses_reference_reduction(data):
        return None
    if WEIGHT_KEY not in data or ROWS_KEY not in data:
        raise ValueError("Reference reduction metadata must be attached before loss computation")
    local = copy.deepcopy(config)
    rows = int(tu.get_non_tensor_data(data, ROWS_KEY, None))
    weight = float(tu.get_non_tensor_data(data, WEIGHT_KEY, None))
    mask = data["response_mask"]
    tokens = int(mask.sum().item())
    loss_scale_factor = config.loss_scale_factor
    if config.loss_agg_mode == "seq-mean-token-sum-norm" and loss_scale_factor is None:
        # The reference divides by the rollout's fixed padded response width,
        # not a microbatch-local length reconstructed after TQ removes padding.
        loss_scale_factor = tu.get_non_tensor_data(data, RESPONSE_LENGTH_KEY, None)
        if loss_scale_factor is None and not mask.is_nested:
            loss_scale_factor = mask.shape[-1]
        if isinstance(loss_scale_factor, bool) or not isinstance(loss_scale_factor, Integral) or loss_scale_factor <= 0:
            raise ValueError(
                f"Reference seq-mean-token-sum-norm requires positive integer {RESPONSE_LENGTH_KEY} "
                "or an explicit loss_scale_factor; nested local widths are not a rollout horizon"
            )
    # agg_loss's dp_size is a multiplicative coefficient. Here DDP correction
    # and accumulation weight are already combined by the engine adapter.
    local.global_batch_info.clear()
    local.global_batch_info.update(
        {
            "dp_size": weight,
            "batch_num_tokens": max(tokens, 1),
            "global_batch_size": 1 if config.loss_agg_mode == "seq-mean-token-sum-norm" else max(rows, 1),
            "loss_scale_factor": loss_scale_factor,
        }
    )
    return local


def reference_policy_metrics(values, data):
    weight = float(tu.get_non_tensor_data(data, WEIGHT_KEY, None))
    return {name: Metric(value=value * weight, aggregation=AggregationType.SUM) for name, value in values.items()}
