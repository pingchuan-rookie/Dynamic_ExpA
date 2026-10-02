"""Compute vocabulary and expanded-action policy statistics with separate softmaxes."""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Compute token or admissible-action probabilities at each sampled decision.
# Extension point: DyadFSDPEngineWithLMHead.prepare_model_outputs

import torch

from agent_system.policies.dyad.algorithms.vocab_statistics import compute_vocab_statistics


def _entropy_from_masked_logits(masked_logits: torch.Tensor, logits_mask: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    probs = torch.softmax(masked_logits, dim=-1)
    log_probs = torch.where(logits_mask, log_probs, torch.zeros_like(log_probs))
    probs = torch.where(logits_mask, probs, torch.zeros_like(probs))
    return -(probs * log_probs).sum(dim=-1)


def _sum_pi_squared_from_masked_logits(masked_logits: torch.Tensor, logits_mask: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(masked_logits, dim=-1)
    probs = torch.where(logits_mask, probs, torch.zeros_like(probs))
    return (probs * probs).sum(dim=-1)


def compute_split_policy_outputs(
    base_logits: torch.Tensor | None,
    action_logits: torch.Tensor,
    labels: torch.Tensor,
    tool_mask: torch.Tensor,
    dyad_action_mask: torch.Tensor,
    temperature: float | torch.Tensor,
    calculate_entropy: bool,
    calculate_sum_pi_squared: bool,
    vocab_kl_labels: torch.Tensor | None = None,
    *,
    precomputed_vocab: dict[str, torch.Tensor] | None = None,
    vocab_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """Return log-probabilities and optional entropy / squared-probability sums.

    Logits have shape [..., vocabulary_size] and [..., action_size]; labels and
    tool_mask have shape [...]. A True tool_mask selects the action distribution.
    dyad_action_mask is [..., action_size], with True marking an allowed action.
    Action labels use extended IDs offset by vocabulary_size. Action normalization
    and returned statistics use FP32; invalid or masked-out labels raise.
    """
    if (base_logits is None) != (precomputed_vocab is not None):
        raise ValueError("Supply either vocabulary logits or fused vocabulary statistics, not both.")
    if precomputed_vocab is not None:
        if not isinstance(vocab_size, int) or vocab_size < 1:
            raise ValueError("Fused vocabulary statistics require the model's vocabulary size.")
        org_vocab_size = vocab_size
        required = ["log_probs", "labels"]
        if calculate_entropy:
            required.append("entropy")
        if calculate_sum_pi_squared:
            required.append("sum_pi_squared")
        for key in required:
            if key not in precomputed_vocab or precomputed_vocab[key].shape != labels.shape:
                raise ValueError(f"Missing or misaligned fused vocabulary statistic: {key}")
        vocab_reference = precomputed_vocab["log_probs"]
    else:
        org_vocab_size = base_logits.size(-1)
        vocab_reference = base_logits
    action_size = dyad_action_mask.size(-1)
    if ((base_logits is not None and base_logits.shape[:-1] != labels.shape)
            or action_logits.shape[:-1] != labels.shape):
        raise RuntimeError(
            "Dyad split policy shape mismatch: "
            f"base_logits={None if base_logits is None else tuple(base_logits.shape)}, action_logits={tuple(action_logits.shape)}, "
            f"labels={tuple(labels.shape)}"
        )
    if action_logits.size(-1) != action_size:
        raise RuntimeError(
            f"Action head output size mismatch: action_logits.size(-1)={action_logits.size(-1)}, "
            f"dyad_action_mask.size(-1)={action_size}"
        )

    # Scale vocabulary rows inside the bounded-memory statistics boundary, not here.
    # Keep the small action distribution in FP32, independently of the vocabulary dtype.
    action_temperature = temperature
    if isinstance(temperature, torch.Tensor):
        action_temperature = temperature.to(device=vocab_reference.device, dtype=torch.float32)
        if action_temperature.numel() != 1:
            action_temperature = action_temperature.reshape(*labels.shape, 1)
    action_logits = action_logits.to(device=vocab_reference.device, dtype=torch.float32) / action_temperature
    tool_mask = tool_mask.to(dtype=torch.bool)
    dyad_action_mask = dyad_action_mask.to(dtype=torch.bool)

    flat_base_logits = None if base_logits is None else base_logits.reshape(-1, org_vocab_size)
    flat_action_logits = action_logits.reshape(-1, action_size)
    flat_labels = labels.reshape(-1)
    flat_tool_mask = tool_mask.reshape(-1)
    flat_action_mask = dyad_action_mask.reshape(-1, action_size)

    flat_log_probs = torch.empty_like(flat_labels, dtype=torch.float32)
    flat_entropy = torch.zeros_like(flat_log_probs) if calculate_entropy else None
    flat_sum_pi_squared = torch.zeros_like(flat_log_probs) if calculate_sum_pi_squared else None

    vocab_positions = ~flat_tool_mask
    if precomputed_vocab is not None:
        fused_labels = precomputed_vocab["labels"].reshape(-1)
        if torch.any(fused_labels[vocab_positions] != flat_labels[vocab_positions]):
            raise ValueError("Fused vocabulary labels disagree with Dyad replay labels.")
        if vocab_kl_labels is not None and torch.any(fused_labels != vocab_kl_labels.reshape(-1)):
            raise ValueError("Fused vocabulary KL labels disagree with their scored labels.")
    if vocab_positions.any():
        vocab_labels = flat_labels[vocab_positions]
        if (vocab_labels < 0).any() or (vocab_labels >= org_vocab_size).any():
            raise RuntimeError("Dyad vocab position contains an expanded action label.")
    vocab_outputs = None
    if vocab_positions.any() or vocab_kl_labels is not None:
        if precomputed_vocab is None:
            vocab_outputs = compute_vocab_statistics(
                flat_base_logits, flat_labels, vocab_positions, temperature,
                calculate_entropy, calculate_sum_pi_squared,
                None if vocab_kl_labels is None else vocab_kl_labels.reshape(-1),
            )
        else:
            # Fused LM kernels already normalized over the complete vocabulary.
            # The engine checks their base-token labels against replay labels.
            vocabulary_log_probs = precomputed_vocab["log_probs"].reshape(-1).float()
            vocab_outputs = (
                vocabulary_log_probs.masked_fill(~vocab_positions, 0.0),
                precomputed_vocab["entropy"].reshape(-1).float().masked_fill(~vocab_positions, 0.0)
                if calculate_entropy else None,
                precomputed_vocab["sum_pi_squared"].reshape(-1).float().masked_fill(~vocab_positions, 0.0)
                if calculate_sum_pi_squared else None,
                vocabulary_log_probs,
            )
        flat_log_probs = vocab_outputs[0]
        if calculate_entropy:
            flat_entropy = vocab_outputs[1]
        if calculate_sum_pi_squared:
            flat_sum_pi_squared = vocab_outputs[2]
        # Custom Function outputs are views; action scatter must not modify them in place.
        flat_log_probs = flat_log_probs.clone()
        if calculate_entropy:
            flat_entropy = flat_entropy.clone()
        if calculate_sum_pi_squared:
            flat_sum_pi_squared = flat_sum_pi_squared.clone()

    if flat_tool_mask.any():
        action_labels = flat_labels[flat_tool_mask] - org_vocab_size
        action_masks = flat_action_mask[flat_tool_mask]
        if (action_labels < 0).any() or (action_labels >= action_size).any():
            raise RuntimeError("Dyad tool position contains a label outside the action-head range.")
        if (~action_masks.any(dim=-1)).any():
            # This message used to be the bare sentence, which is why the codegym blocker (B1, filed
            # 2026-07-28) sat undiagnosed: it says a position had an empty admissible action set but
            # not which one, so there is no way to tell an off-by-one in the replay from a phase that
            # genuinely admits nothing.
            #
            # Everything below works on the *flattened* tensors. `labels` is 2-D
            # ([batch, seqlen]) on the padded path but 1-D on the remove-padding path, so indexing
            # it as [row, col] crashes with "too many indices for tensor of dimension 1" -- and it
            # crashes while reporting another error, which hides the original one entirely.
            bad = (~action_masks.any(dim=-1)).nonzero(as_tuple=True)[0]
            tool_positions = flat_tool_mask.nonzero(as_tuple=True)[0]
            flat_positions = tool_positions[bad]
            total = flat_labels.numel()
            detail = []
            for k in range(min(5, int(bad.numel()))):
                flat_pos = int(flat_positions[k])
                # The neighbouring decisions distinguish "the replay drifted" (the positions around
                # it are fine and this one is off by one action) from "this phase admits nothing
                # by design" (a whole run of positions is empty).
                lo, hi = max(0, flat_pos - 3), min(total, flat_pos + 4)
                detail.append(
                    f"flat_pos={flat_pos} label={int(flat_labels[flat_pos])} "
                    f"(action_id={int(flat_labels[flat_pos]) - org_vocab_size}) "
                    f"neighbours[{lo}:{hi}] tool_mask={flat_tool_mask[lo:hi].tolist()} "
                    f"n_allowed={flat_action_mask[lo:hi].sum(dim=-1).tolist()}"
                )
            raise RuntimeError(
                "Dyad tool position has no allowed actions. "
                f"{int(bad.numel())} of {int(flat_tool_mask.sum())} tool positions have an all-zero "
                f"dyad_action_mask (action_size={action_size}, labels.shape={tuple(labels.shape)}). "
                "The admissible action set applied while sampling and the one replayed for training must be "
                "the same one (AGENTS.md section 1), so an empty set here means the replay and the "
                "rollout disagree about what phase this position is in. "
                + " | ".join(detail)
            )
        selected_actions_allowed = action_masks.gather(dim=-1, index=action_labels.unsqueeze(-1)).squeeze(-1)
        if (~selected_actions_allowed).any():
            raise RuntimeError("Dyad selected action label is not enabled by dyad_action_mask.")

        masked_action_logits = flat_action_logits[flat_tool_mask].masked_fill(~action_masks, float("-inf"))
        # The action head is small: stable log_softmax avoids subtracting a large common
        # score from its logsumexp and matches the rollout sampler's FP32 normalization.
        flat_log_probs[flat_tool_mask] = torch.log_softmax(
            masked_action_logits, dim=-1, dtype=torch.float32
        ).gather(dim=-1, index=action_labels.unsqueeze(-1)).squeeze(-1)
        if calculate_entropy:
            flat_entropy[flat_tool_mask] = _entropy_from_masked_logits(masked_action_logits, action_masks).to(
                dtype=flat_entropy.dtype
            )
        if calculate_sum_pi_squared:
            flat_sum_pi_squared[flat_tool_mask] = _sum_pi_squared_from_masked_logits(
                masked_action_logits, action_masks
            ).to(dtype=flat_sum_pi_squared.dtype)

    outputs = {"log_probs": flat_log_probs.reshape(labels.shape)}
    if vocab_kl_labels is not None:
        outputs["vocab_log_probs"] = vocab_outputs[3].reshape(labels.shape)
    if calculate_entropy:
        outputs["entropys"] = flat_entropy.reshape(labels.shape)
    if calculate_sum_pi_squared:
        outputs["sum_pi_squared"] = flat_sum_pi_squared.reshape(labels.shape)
    return outputs
