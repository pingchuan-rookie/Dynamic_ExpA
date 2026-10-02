"""Dyad's PPO loss on verl 0.9's injectable-loss interface.

Signature and normalization follow `verl.workers.utils.losses.ppo_loss` exactly; this is a
deliberate near-copy rather than a wrapper, because the two differences below sit in the middle of
that function and reaching them through it would mean either mutating the shared `ActorConfig` or
depending on upstream internals that the engine rewrite has already moved once.

The two differences, and why each exists:

  1. `response_mask &= seq_mask`.
     `seq_mask=False` marks positions that are in the response span but are not policy decisions:
     force-written template tokens, environment observations, padding. They have no log-prob to
     optimize (the engine zeroes them). They are excluded from the numerator; the shared
     response-token denominator is intentionally preserved for the policy and encoder lines.

  2. KL uses `vocab_log_probs`, masked to non-action positions.
     The reference actor is a plain vocabulary model with no action head, so at an action-decision
     position there is no reference distribution to be KL-divergent from. Taking KL between the
     Dyad policy log-prob and a vocab-only reference at those positions would compare two different
     sample spaces and produce a number that looks like a KL but is not one.
"""
# DYAD-ADD(module): Objective adapter at the official actor loss-function boundary.
# Derived function: verl/workers/utils/losses.py::ppo_loss, verl v0.9.0.
# Source commit: 483b8a009ba3a97563edee3a19887e4862b8094a
# Changed objective steps retain the corresponding upstream code below.

import os

import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig
from verl.workers.utils.padding import no_padding_2_padding

from agent_system.utils.diagnostics import OFF, diag_level, log_event


_MASKED_POLICY_METRICS = frozenset({"actor/pg_clipfrac", "actor/ppo_kl", "actor/pg_clipfrac_lower"})


def _policy_metrics(values, mask, data, *, encoder=False):
    """Transport-invariant masked means, without changing the shared Metric reducer.

    Metric.SUM adds microbatches but averages DP ranks, so each contribution is
    its masked numerator times dp_size divided by the global decision count.
    These counts are attached once per minibatch before splitting, separately
    from batch_num_tokens: that existing objective denominator includes response
    positions excluded by seq_mask and must not change here.
    Keep one scalar per microbatch, including zero for dummy-only microbatches,
    since Metric.aggregate_dp requires equal list lengths on every rank.
    """
    from verl_extensions.agent_steps.loss_reduction import reference_policy_metrics, uses_reference_reduction

    if uses_reference_reduction(data):
        return reference_policy_metrics(values, data)
    metrics = Metric.from_dict(values, aggregation=AggregationType.MEAN)
    if os.environ.get("DYAD_EXACT_BATCH_PADDING") != "1":
        return metrics

    key = "dyad_metric_num_action_tokens" if encoder else "dyad_metric_num_tokens"
    if key not in data.keys():
        raise RuntimeError(
            f"DYAD_EXACT_BATCH_PADDING=1 requires global minibatch metric count {key!r}; "
            "attach it before microbatch splitting, without changing batch_num_tokens"
        )
    count = int(mask.sum().detach().item())
    global_count = int(data[key])
    if global_count < count:
        raise RuntimeError(f"invalid {key}={global_count}: local masked count is {count}")
    # Undo masked_mean's epsilon before applying the global mean denominator.
    # An empty mask contributes zero, not a microbatch observation or NaN * 0.
    weight = (count + 1e-8) * data["dp_size"] / (global_count + 1e-8) if count else 0.0
    for name in _MASKED_POLICY_METRICS.intersection(values):
        value = values[name] * weight if count else 0.0
        metrics[name] = Metric(value=value, aggregation=AggregationType.SUM)
    return metrics


def dyad_ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """PPO loss over Dyad's split vocabulary/action policy.

    `model_output` comes from `DyadFSDPEngineWithLMHead.prepare_model_outputs`, so `log_probs` is
    already the split-head log-prob: the base vocabulary head at vocabulary decisions, the action
    head restricted to that step's admissible action set at action decisions.
    """
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor
    # >>> DYAD-ADD(dyad_ppo_loss)
    # Use the shared step reduction contract before assembling policy and encoder objectives.
    from verl_extensions.agent_steps.loss_reduction import reference_loss_config

    reference_config = reference_loss_config(config, data)
    if reference_config is not None:
        config = reference_config
    # <<< DYAD-ADD(dyad_ppo_loss)

    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    fields = ["response_mask", "old_log_probs", "advantages"]
    # >>> DYAD-REPLACE(dyad_ppo_loss)
    # Retain seq_mask/tool_mask and apply the exact sampled-decision mask without changing stored advantages.
    # Original upstream verl/workers/utils/losses.py L87-L97 (before replacement):
    # if "rollout_is_weights" in data:
    #     fields.append("rollout_is_weights")
    # if "ref_log_prob" in data:
    #     fields.append("ref_log_prob")
    # data = data.select(*fields).to_padded_tensor()
    # response_mask = data["response_mask"].to(bool)
    # old_log_prob = data["old_log_probs"]
    # advantages = data["advantages"]
    # rollout_is_weights = data.get("rollout_is_weights", None)
    for optional in ("rollout_is_weights", "ref_log_prob", "seq_mask", "tool_mask"):
        if optional in data.keys():
            fields.append(optional)
    selected = data.select(*fields).to_padded_tensor()

    response_mask = selected["response_mask"].to(bool)

    # Difference 1: non-decision positions are not part of the objective.
    if "seq_mask" in selected.keys():
        response_mask = torch.logical_and(response_mask, selected["seq_mask"].to(bool))

    old_log_prob = selected["old_log_probs"]
    advantages = selected["advantages"]
    rollout_is_weights = selected.get("rollout_is_weights", None)
    # <<< DYAD-REPLACE(dyad_ppo_loss)

    loss_agg_mode = config.loss_agg_mode
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # >>> DYAD-REPLACE(dyad_ppo_loss)
    # Aggregate clipped-PPO diagnostics using the shared step/DP metric weights.
    # Original upstream verl/workers/utils/losses.py L116-L116 (before replacement):
    # pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)
    pg_metrics = _policy_metrics(pg_metrics, response_mask, data)
    # <<< DYAD-REPLACE(dyad_ppo_loss)
    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # Difference 3: the action encoder's own surrogate, when the engine emitted one.
    #
    # Same advantages, same clipping, same `policy_loss_fn` -- what differs is the mask and the
    # graph. The encoder numerator contains expanded-action decisions alone, where `w_a`
    # appears. Both lines share the response-token denominator under token-mean; this
    # deliberately retains the established policy/encoder relative gradient scaling.
    #
    # The graph is the encoder line: `sg[x_t] . w_a`. Adding the two losses is exact rather than
    # approximate here -- the policy line has no gradient to phi and the encoder line none to theta,
    # so `d(L_policy + L_AE)/d(theta) = d(L_policy)/d(theta)` and likewise for phi. A single backward
    # therefore does what two separate ones would, without FSDP having to survive two of them.
    #
    # Reference microbatch mode changes accumulation across micros, not the relative
    # policy/encoder denominator inside one micro. Sequence means use each line's
    # per-row token mask but retain their shared outer sequence denominator.
    # >>> DYAD-ADD(dyad_ppo_loss)
    # Add the action-encoder surrogate through the same registered PPO loss, with policy states detached.
    encoder_log_prob = model_output.get("encoder_log_probs", None)
    if encoder_log_prob is not None:
        if "tool_mask" not in selected.keys():
            raise RuntimeError(
                "the engine produced an encoder gradient line but the batch carries no tool_mask, "
                "so there is no way to restrict the encoder's loss to expanded-action decisions. "
                "Without the restriction the projector would be normalised over vocabulary "
                "positions it does not appear in."
            )
        encoder_log_prob = no_padding_2_padding(encoder_log_prob, data)
        encoder_mask = torch.logical_and(response_mask, selected["tool_mask"].to(bool))
        ae_loss, ae_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=encoder_log_prob,
            advantages=advantages,
            response_mask=encoder_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
        )
        policy_loss = policy_loss + ae_loss
        metrics["dyad/ae_loss"] = Metric(value=ae_loss, aggregation=metric_aggregation)
        # SUM metrics still average DP ranks; compensate counts, not loss weights.
        metrics["dyad/ae_positions"] = Metric(
            value=encoder_mask.sum() * data["dp_size"], aggregation=AggregationType.SUM
        )
        for key, value in _policy_metrics(ae_metrics, encoder_mask, data, encoder=True).items():
            metrics[f"dyad/ae_{key.split('/')[-1]}"] = value
    # <<< DYAD-ADD(dyad_ppo_loss)

    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        # Keep entropy metrics at zero coefficient without scheduling its backward.
        # Fused vocabulary kernels otherwise allocate a full chunk for zero gradients.
        # >>> DYAD-REPLACE(dyad_ppo_loss)
        # Avoid retaining a zero-weight entropy backward graph while preserving the metric.
        # Original upstream verl/workers/utils/losses.py L127-L128 (before replacement):
        # entropy_coeff = config.entropy_coeff
        # policy_loss -= entropy_coeff * entropy_loss
        if config.entropy_coeff != 0:
            policy_loss -= config.entropy_coeff * entropy_loss
        # <<< DYAD-REPLACE(dyad_ppo_loss)
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    if config.use_kl_loss:
        # >>> DYAD-REPLACE(dyad_ppo_loss)
        # Restrict reference KL to vocabulary decisions because the reference policy has no action head.
        # Original upstream verl/workers/utils/losses.py L133-L138 (before replacement):
        # ref_log_prob = data["ref_log_prob"]
        # kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        # kl_loss = agg_loss(
        #     loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        # )
        ref_log_prob = selected["ref_log_prob"]

        # Difference 2: vocabulary-only log-prob, and only at vocabulary positions.
        kl_log_prob = model_output.get("vocab_log_probs", None)
        if kl_log_prob is None:
            kl_log_prob = log_prob
        else:
            kl_log_prob = no_padding_2_padding(kl_log_prob, data)

        kl_mask = response_mask
        if "tool_mask" in selected.keys():
            kl_mask = torch.logical_and(kl_mask, ~selected["tool_mask"].to(bool))

        kld = kl_penalty(logprob=kl_log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=kl_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )
        # <<< DYAD-REPLACE(dyad_ppo_loss)

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    # Dyad-specific observability: how the batch splits between the two heads. A run whose
    # action-position count collapses to zero is training the base-token policy under an Dyad name,
    # and nothing else in the metric set would say so.
    # >>> DYAD-ADD(dyad_ppo_loss)
    # Report decision counts and sampled-policy diagnostics after constructing the objective.
    tool_mask = selected["tool_mask"].to(bool) if "tool_mask" in selected.keys() else None
    if tool_mask is not None:
        action_positions = torch.logical_and(response_mask, tool_mask).sum()
        vocab_positions = torch.logical_and(response_mask, ~tool_mask).sum()
        metrics["dyad/action_positions"] = Metric(
            value=action_positions * data["dp_size"], aggregation=AggregationType.SUM
        )
        metrics["dyad/vocab_positions"] = Metric(
            value=vocab_positions * data["dp_size"], aggregation=AggregationType.SUM
        )

    _log_micro_batch_events(
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        response_mask=response_mask,
        tool_mask=tool_mask,
        pg_loss=pg_loss,
        policy_loss=policy_loss,
    )
    # <<< DYAD-ADD(dyad_ppo_loss)

    return policy_loss, metrics


def _log_micro_batch_events(
    *, log_prob, old_log_prob, advantages, response_mask, tool_mask, pg_loss, policy_loss
) -> None:
    """Emit the two per-micro-batch diagnostic events the analysis tooling reads.

    `experiments/shared/analysis/run/verify_dyad.py` keys on the source `dyad_actor` and on these two event
    names. They were emitted from `DyadActor.update_policy` before the 0.9 port moved the loss out
    of the policy LLM backbone class; keeping the source and names means the analysis scripts and
    `diagnostics_contract.md` need no change. Losing them would not raise anywhere -- verify_dyad
    would simply report no events, which reads as "the run did not train" rather than "the
    diagnostics moved".

    The legacy "backward" event records the assembled loss before the engine calls
    backward. It is not evidence that backward completed; use optimizer-step events
    and a successful batch return to certify completed updates.
    """
    # Avoid device reductions, indexing and host synchronizations for discarded logs.
    if diag_level() == OFF:
        return
    inactive_mask = ~response_mask
    alignment_stats = {
        "active_positions": int(response_mask.sum().detach().item()),
        "inactive_positions": int(inactive_mask.sum().detach().item()),
        "old_inactive_abs_max": float(old_log_prob[inactive_mask].detach().abs().max().item())
        if inactive_mask.any()
        else 0.0,
        "new_inactive_abs_max": float(log_prob[inactive_mask].detach().abs().max().item())
        if inactive_mask.any()
        else 0.0,
    }
    if tool_mask is not None:
        alignment_stats["tool_active_positions"] = int(
            torch.logical_and(response_mask, tool_mask).sum().detach().item()
        )
        alignment_stats["vocab_active_positions"] = int(
            torch.logical_and(response_mask, ~tool_mask).sum().detach().item()
        )

    log_event(
        "dyad_actor",
        "policy_micro_batch_forward",
        response_mask=response_mask,
        old_log_prob=old_log_prob,
        new_log_prob=log_prob,
        advantages=advantages,
        ratio=torch.exp(torch.clamp(log_prob.detach() - old_log_prob.detach(), -20.0, 20.0)),
        alignment_stats=alignment_stats,
        has_dyad_fields=tool_mask is not None,
    )
    log_event(
        "dyad_actor",
        "policy_micro_batch_backward",
        loss=policy_loss.detach(),
        pg_loss=pg_loss.detach(),
    )
