"""Capability gate for exact-budget CodeGym transport padding."""

# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Capability gate for exact-budget CodeGym transport padding.
# Extension point: PPOTrainer step, advantage, actor-update, and FSDP reduction hooks
# DYAD-AGENT-STEPS: retain the legacy controller's explicit padding capability gate.
from __future__ import annotations

import os


def validate_exact_batch_padding(config, *, use_critic=False, env=None):
    """Only the custom Dyad FSDP controller implements padded transport."""
    env = os.environ if env is None else env
    value = env.get("DYAD_EXACT_BATCH_PADDING", "0")
    if value not in ("0", "1"):
        raise ValueError("DYAD_EXACT_BATCH_PADDING must be 0 or 1")
    if value == "0":
        return False
    from omegaconf import OmegaConf

    def get(path, default=None):
        return OmegaConf.select(config, path, default=default)

    failures = []
    if env.get("DYAD_CODEGYM_ALL") != "1":
        failures.append("full CodeGym")
    if get("actor_rollout_ref.actor.strategy") != "dyad":
        failures.append("Dyad actor (FSDP engine)")
    # The custom trainer normalizes all supported FSDP modes to the new engine.
    if get("trainer.use_legacy_worker_impl", "auto") not in ("auto", "enable", "disable"):
        failures.append("valid engine worker selection")
    if use_critic or get("algorithm.adv_estimator") not in ("grpo", "gigpo"):
        failures.append("GRPO or GiGPO without critic")
    if get("actor_rollout_ref.actor.policy_loss.loss_mode", "vanilla") != "vanilla":
        failures.append("vanilla PPO loss")
    if get("actor_rollout_ref.actor.shuffle", False):
        failures.append("actor.shuffle=False")
    if int(get("actor_rollout_ref.actor.ppo_epochs", 1)) != 1:
        failures.append("one PPO epoch")
    if int(get("trainer.nnodes", 1)) != 1:
        failures.append("single-node policy")
    if int(get("actor_rollout_ref.actor.ulysses_sequence_parallel_size", 1)) != 1:
        failures.append("sequence parallel size 1")
    if int(get("actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size", 1)) != 1:
        failures.append("engine sequence parallel size 1")
    if get("actor_rollout_ref.actor.use_prefix_grouper", False):
        failures.append("no prefix grouper")
    for role in ("actor", "ref"):
        path = f"actor_rollout_ref.{role}"
        if role == "ref" and get(path + ".strategy", "fsdp") not in ("dyad", "fsdp", "fsdp2"):
            failures.append("FSDP reference")
    batch = int(get("data.train_batch_size", 0))
    mini = int(get("actor_rollout_ref.actor.ppo_mini_batch_size", 0))
    if batch < 1 or mini < 1 or batch % mini:
        failures.append("whole real optimizer minibatches")
    if failures:
        raise ValueError("DYAD_EXACT_BATCH_PADDING requires " + ", ".join(failures))
    return True
