"""Versioned training semantics, checked before restoring optimizer state."""

# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Versioned training semantics, checked before restoring optimizer state.
# Extension point: PPOTrainer step, advantage, actor-update, and FSDP reduction hooks
# DYAD-AGENT-STEPS: preserve saved training semantics across runtime package moves.
from __future__ import annotations

import json
import math
from pathlib import Path

from omegaconf import OmegaConf

PROTOCOL_VERSION = 2
PARTITION_PROTOCOL = "source_order_then_raw_length_dp_v1"
# DYAD-STEP: apply only with the CPU-float32 advantage and nominal-mini loss fixes.
# Fixed runtime identities, not config overrides: old optimizer state cannot silently migrate.
ADVANTAGE_NUMERICAL_PROTOCOL = "reference_cpu_float32_v1"
LOSS_REDUCTION_NUMERICAL_PROTOCOL = "reference_nominal_mini_v1"
PROFILES = {"alfworld_official_v2", "webshop_official_v2", "dive_native_v2", "codegym_native_v2"}


def shared_step_enabled(config):
    return bool(config.algorithm.get("step_rollout", {}).get("enabled", False))


def validate_step_config(config, *, use_critic=False):
    """Reject incompatible configurations before allocating workers."""
    if not shared_step_enabled(config):
        return
    step = config.algorithm.step_rollout
    actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    # DYAD-SWE: permit the OOD scaffold only in the no-update evaluation path.
    profiles = PROFILES | ({"swebench_verified_native_v2"} if config.trainer.val_only else set())
    if step.get("protocol_version") != PROTOCOL_VERSION or step.get("profile") not in profiles:
        raise ValueError("Unsupported shared step protocol version/profile")
    if config.algorithm.adv_estimator not in {"grpo", "gigpo"}:
        raise ValueError("Shared step training requires GRPO or GiGPO")
    if actor.get("ppo_mini_batch_size_unit") != "step":
        raise ValueError("Shared step training requires ppo_mini_batch_size_unit=step")
    world_size = int(config.trainer.n_gpus_per_node) * int(config.trainer.get("nnodes", 1))
    fsdp = actor.get("fsdp_config") or {}
    sequence_parallel = int(fsdp.get("ulysses_sequence_parallel_size", actor.get("ulysses_sequence_parallel_size", 1)))
    if sequence_parallel < 1 or world_size < 1 or world_size % sequence_parallel:
        raise ValueError("Shared step actor world size must be divisible by sequence parallel size")
    dp_size = world_size // sequence_parallel
    mini_size = actor.ppo_mini_batch_size
    if isinstance(mini_size, bool) or not isinstance(mini_size, int) or mini_size < 1 or mini_size % dp_size:
        raise ValueError("Reference step mini-batches must be divisible by actor DP size")
    if actor.get("shuffle", False):
        raise ValueError("Reference step mini-batches require shuffle=False")
    if not actor.get("use_dynamic_bsz", False):
        micro = actor.get("ppo_micro_batch_size_per_gpu")
        if isinstance(micro, bool) or not isinstance(micro, int) or micro < 1 or (mini_size // dp_size) % micro:
            raise ValueError("Reference rank-local mini-batches must be divisible by the micro-batch size")
    if not config.trainer.get("use_v1", True) or config.trainer.v1.trainer_mode != "sync":
        raise ValueError("Shared step training requires V1 sync mode")
    if config.trainer.v1.get("sync", {}).get("parameter_sync_step", 1) != 1:
        raise ValueError("Shared step training requires parameter_sync_step=1")
    if rollout.agent.default_agent_loop != "environment_step_agent":
        raise ValueError("Shared step training requires environment_step_agent")
    dyad = step.get("action_interface") == "dyad"
    if step.get("action_interface") not in {"text", "dyad"}:
        raise ValueError("Shared step action_interface must be text or dyad")
    if (actor.strategy == "dyad") != dyad or (rollout.name == "dyadvllm") != dyad:
        raise ValueError("Shared step action interface, worker strategy and rollout backend disagree")
    if not dyad and actor.strategy not in {"fsdp", "fsdp2"}:
        raise ValueError("Shared step text policy requires FSDP")
    if use_critic or config.get("distillation", {}).get("enabled", False):
        raise ValueError("Shared GRPO/GiGPO steps do not support critic or distillation")
    if config.get("reward", {}).get("reward_model", {}).get("enable", False):
        raise ValueError("Shared step v2 uses raw environment rewards, not a learned reward model")
    if config.algorithm.use_kl_in_reward or config.algorithm.use_pf_ppo:
        raise ValueError("Shared step protocol requires raw environment rewards and unweighted PPO")
    filtering = config.algorithm.get("filter_groups")
    if filtering and filtering.get("enable", False):
        attempts = filtering.get("max_num_gen_batches", 0)
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ValueError("Shared dynamic sampling requires positive max_num_gen_batches")
        if filtering.get("metric") not in {None, "episode_return"}:
            raise ValueError("Shared dynamic sampling filters raw episode_return only")
        if filtering.get("max_inflight_gen_batches", 1) != 1:
            raise ValueError("Shared dynamic sampling requires max_inflight_gen_batches=1")
        sampler = config.trainer.v1.get("sampler", {})
        custom = sampler.get("custom_sampler") or {}
        if custom.get("path") or custom.get("name") or sampler.get("sync_refill_failed_groups", False):
            raise ValueError("Shared dynamic sampling requires its built-in whole-batch replay buffer")
        if config.data.get("gen_batch_size") not in {None, config.data.train_batch_size}:
            raise ValueError("Shared dynamic sampling requires gen_batch_size=train_batch_size")
    if actor.policy_loss.get("loss_mode", "vanilla") != "vanilla":
        raise ValueError("Shared step protocol requires vanilla PPO loss")
    if config.data.continuous_token.enable or not rollout.multi_turn.enable:
        raise ValueError("Shared steps require independent multi-turn sequences")
    if not config.trainer.val_only and rollout.n < 2:
        raise ValueError("Shared GRPO/GiGPO training requires at least two trajectories per task")
    for field in ("history_length", "max_steps"):
        value = step.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if field == "history_length" else 1):
            raise ValueError(f"Shared step {field} must be a valid integer")
    penalty = float(step.invalid_action_penalty)
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("Shared step invalid_action_penalty must be finite and nonnegative")
    if step.get("resampling") != "reference_copy" or step.get("loss_reduction") != "reference_microbatch":
        raise ValueError("Shared step v2 requires reference_copy and reference_microbatch")
    if "rollout_layout" in config.algorithm.gigpo:
        raise ValueError("algorithm.gigpo.rollout_layout was removed; use shared step protocol v2")
    options = config.algorithm.gigpo
    if config.algorithm.adv_estimator == "gigpo":
        weight, gamma = float(options.step_advantage_w), float(config.algorithm.gamma)
        if options.mode not in {"mean_norm", "mean_std_norm"}:
            raise ValueError("GiGPO mode must be mean_norm or mean_std_norm")
        if not math.isfinite(weight) or weight < 0 or not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("GiGPO requires finite nonnegative weight and gamma in [0,1]")
        if options.get("enable_similarity", False):
            threshold = float(options.get("similarity_thresh", 0.95))
            if not math.isfinite(threshold) or not 0 < threshold < 1:
                raise ValueError("GiGPO similarity_thresh must be in (0,1)")
    correction = config.algorithm.get("rollout_correction")
    if correction and any(correction.get(key) for key in ("rollout_rs", "rollout_is", "bypass_mode")):
        raise ValueError("Shared step v2 requires recomputed old-policy logprobs without rollout IS/RS correction")
    if actor.get("use_rollout_log_probs", False):
        raise ValueError("Shared step v2 requires recomputed old-policy logprobs")


def training_protocol(config):
    """Stable semantic identity, including mean-of-means partition semantics."""
    if not shared_step_enabled(config):
        return None

    def plain(value):
        return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value

    actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    world_size = int(config.trainer.n_gpus_per_node) * int(config.trainer.get("nnodes", 1))
    fsdp = actor.get("fsdp_config") or {}
    sequence_parallel = int(fsdp.get("ulysses_sequence_parallel_size", actor.get("ulysses_sequence_parallel_size", 1)))
    if sequence_parallel < 1 or world_size < 1 or world_size % sequence_parallel:
        raise ValueError("Shared step actor world size must be divisible by sequence parallel size")
    micro_sizes = [actor.get("ppo_micro_batch_size_per_gpu"), rollout.get("log_prob_micro_batch_size_per_gpu")]
    if actor.get("use_kl_loss", False) or config.algorithm.get("use_kl_in_reward", False):
        micro_sizes.append(config.actor_rollout_ref.get("ref", {}).get("log_prob_micro_batch_size_per_gpu"))
    micro_divisors = [1]
    for size in micro_sizes:
        if size is not None:
            if isinstance(size, bool) or int(size) != size or size <= 0:
                raise ValueError("Reference step resampling requires positive microbatch sizes")
            micro_divisors.append(int(size))
    return {
        "version": PROTOCOL_VERSION,
        "partition_protocol": PARTITION_PROTOCOL,
        "numerical_protocol": {
            "advantage": ADVANTAGE_NUMERICAL_PROTOCOL,
            "loss_reduction": LOSS_REDUCTION_NUMERICAL_PROTOCOL,
        },
        "estimator": config.algorithm.adv_estimator,
        "action_interface": config.algorithm.step_rollout.action_interface,
        # DYAD-CODEGYM: do not silently resume an optimizer across repaired-call
        # versus official raw-call/error feedback semantics.
        **(
            {"action_error_protocol": "codegym_official_client_v1"}
            if config.algorithm.step_rollout.profile == "codegym_native_v2"
            else {}
        ),
        "step_rollout": plain(config.algorithm.step_rollout),
        "gamma": config.algorithm.gamma,
        "gigpo": plain(config.algorithm.gigpo) if config.algorithm.adv_estimator == "gigpo" else None,
        "norm_adv_by_std_in_grpo": config.algorithm.norm_adv_by_std_in_grpo,
        "prompt_length": rollout.prompt_length,
        "response_length": rollout.response_length,
        "thinking": plain(config.data.get("apply_chat_template_kwargs", {})),
        "group_n": rollout.n,
        "mini_batch_size": actor.ppo_mini_batch_size,
        "ppo_epochs": actor.ppo_epochs,
        "loss_agg_mode": actor.loss_agg_mode,
        "temperature": rollout.temperature,
        "partition": {
            "actor_world_size": world_size,
            "data_parallel_size": world_size // sequence_parallel,
            "sequence_parallel_size": sequence_parallel,
            "fsdp_size": fsdp.get("fsdp_size", -1),
            "balance_batch": config.trainer.get("balance_batch", True),
        },
        "sampling": {
            "reference_copy_divisor": (world_size // sequence_parallel) * math.lcm(*micro_divisors),
            "train_batch_size": config.data.get("train_batch_size"),
            "generation_batch_size": config.data.get("gen_batch_size"),
            "dynamic_group_sampling": plain(config.algorithm.get("filter_groups")),
            "sampler": plain(config.trainer.get("v1", {}).get("sampler")),
            "top_p": rollout.get("top_p"),
            "top_k": rollout.get("top_k"),
            "do_sample": rollout.get("do_sample"),
            "parameter_sync_step": config.trainer.get("v1", {}).get("sync", {}).get("parameter_sync_step", 1),
            "rollout_micro_batch_size_per_gpu": rollout.get("log_prob_micro_batch_size_per_gpu"),
            "reference_micro_batch_size_per_gpu": config.actor_rollout_ref.get("ref", {}).get(
                "log_prob_micro_batch_size_per_gpu"
            ),
        },
        "optimizer": plain(actor.get("optim")),
        "objective": {
            "policy_loss": plain(actor.get("policy_loss")),
            "rollout_correction": plain(config.algorithm.get("rollout_correction")),
            "use_kl_in_reward": config.algorithm.get("use_kl_in_reward", False),
            "use_pf_ppo": config.algorithm.get("use_pf_ppo", False),
        },
        "optimization": {
            key: plain(actor.get(key))
            for key in (
                "use_dynamic_bsz",
                "ppo_micro_batch_size_per_gpu",
                "ppo_max_token_len_per_gpu",
                "clip_ratio",
                "clip_ratio_low",
                "clip_ratio_high",
                "clip_ratio_c",
                "grad_clip",
                "entropy_coeff",
                "use_kl_loss",
                "kl_loss_coef",
                "kl_loss_type",
                "use_rollout_log_probs",
                "loss_scale_factor",
                "shuffle",
                "data_loader_seed",
            )
        },
    }


def save_training_protocol(config, checkpoint_dir):
    protocol = training_protocol(config)
    if protocol is not None:
        target = Path(checkpoint_dir) / "training_protocol.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")


def validate_training_resume(config, checkpoint_dir, *, weights_only=False):
    """Evaluation/explicit weight import may change protocol, optimizer resume may not."""
    expected = training_protocol(config)
    if expected is None or config.trainer.val_only or weights_only:
        return
    target = Path(checkpoint_dir) / "training_protocol.json"
    if not target.is_file():
        raise ValueError("Legacy checkpoint has no shared training protocol; full-state resume is unsafe")
    saved = json.loads(target.read_text())
    # DYAD-CHECKPOINT: a method-label rename does not change the training protocol.
    from agent_system.policies.dyad.checkpoint_compat import normalize_training_protocol

    saved = normalize_training_protocol(saved)
    if saved != expected:
        changed = sorted(key for key in set(saved) | set(expected) if saved.get(key) != expected.get(key))
        raise ValueError("Checkpoint training protocol mismatch: " + ", ".join(changed))
