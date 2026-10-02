"""Attach action logits to the policy forward before FSDP resharding.

The policy and encoder gradient paths are computed with opposite factors detached.
"""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Attach differentiable action logits with separate policy and encoder gradient paths.
# Extension point: install_dyad_action_head -> policy forward hook before FSDP resharding

from torch import nn


def attach_dyad_forward_hook(actor_module: nn.Module):
    """Compute action logits before FSDP reshards the policy LLM backbone parameters.

    Two ways to get the head's weight, and which one is used decides whether the encoder's projector
    can learn anything:

      `module.action_head`            A frozen nn.Linear whose weight was written in place by
                                    `_reinit_actor_action_head_from_lm_head`. `copy_` severs the graph
                                    and the parameter is frozen, so nothing upstream of it gets a
                                    gradient -- which is correct when the head is not trained.
      `module.action_head_weight_fn`  installed when the schedule trains the action encoder
                                    (joint / frozen_llm_adaptation). Returns a **differentiable** head built
                                    from encoder representations and the trainable projector, so
                                    d(loss)/d(projector) exists.

    Only the second path can train the projector. Leaving the first one in place for
    `policy_lm_only` is deliberate: that schedule uses the materialized, frozen action rows.

    **Two gradient lines.** An action logit is `x_t . w_a`, and both factors are trainable in
    principle: `x_t` belongs to the policy LLM backbone, `w_a` to the action encoder. One tensor carrying both
    gradients means one loss updating both parameter sets, and then the policy LLM backbone's normalisation
    (over every decision) and the encoder's (over expanded-action decisions only) cannot both hold
    -- the two objectives differ by a factor that changes with every trajectory.

    So the head is evaluated once per line, with the other factor detached:

        policy line     x_t . sg[w_a]      gradient reaches theta only
        encoder line    sg[x_t] . w_a      gradient reaches phi only

    Which lines exist is read off `module.dyad_gradient_lines`, set by `apply_training_schedule`.
    It is an explicit attribute rather than an inspection of `requires_grad` because under
    gradient checkpointing the forward runs twice, once under `no_grad`, and a hook that branched
    on `requires_grad` would emit a different number of tensors on the two passes.

    Absent the attribute this emits the single undetached tensor it always did, so every path that
    does not set it -- the tests, the rollout engine, `TRAIN_TARGET=policy_lm` on the frozen head --
    keeps its previous behaviour exactly.
    """

    def add_action_logits(module: nn.Module, _args, output):
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            return output
        # Only the final layer feeds the action head. Release the output's other
        # layer references before projecting a large tool schema; otherwise they
        # keep activations on GPU even when autograd offloads its saved tensors.
        last_hidden = hidden_states[-1]
        try:
            output["hidden_states"] = None
        except TypeError as exc:
            raise RuntimeError("Dyad policy LM output must be a mutable transformers ModelOutput.") from exc
        del hidden_states

        dynamic_logits = getattr(module, "dyad_task_logits_fn", None)
        if dynamic_logits is not None:
            output["hidden_states"] = dynamic_logits(last_hidden)
            return output
        weight_fn = getattr(module, "action_head_weight_fn", None)
        if weight_fn is None:
            action_logits = (module.action_head(last_hidden),)
        else:
            weight = weight_fn()
            if weight.shape != module.action_head.weight.shape:
                # A shape drift here would otherwise surface as a wrong-sized logits block that the
                # split-policy guards blame on the mask.
                raise RuntimeError(
                    f"action_head_weight_fn returned {tuple(weight.shape)} but the head is "
                    f"{tuple(module.action_head.weight.shape)}; the projector and the head must "
                    "describe the same action list in the same order"
                )
            weight = weight.to(last_hidden.dtype)
            lines = getattr(module, "dyad_gradient_lines", None)
            if lines == "both":
                # The only configuration that pays for two matmuls, and the only one that needs
                # them: with one of the two frozen, one of these tensors carries no gradient at all.
                action_logits = (
                    nn.functional.linear(last_hidden, weight.detach()),
                    nn.functional.linear(last_hidden.detach(), weight),
                )
            elif lines == "policy":
                action_logits = (nn.functional.linear(last_hidden, weight.detach()),)
            elif lines == "encoder":
                action_logits = (nn.functional.linear(last_hidden.detach(), weight),)
            else:
                action_logits = (nn.functional.linear(last_hidden, weight),)

        try:
            # FSDP rebuilds training outputs while installing backward hooks and can
            # drop undeclared ModelOutput keys. Reuse this declared field so the
            # pre-reshard action logits survive the FSDP output traversal.
            output["hidden_states"] = action_logits
        except TypeError as exc:
            raise RuntimeError("Dyad policy LM output must be a mutable transformers ModelOutput.") from exc
        return output

    return actor_module.register_forward_hook(add_action_logits)
