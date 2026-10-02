"""Validate V1 step identities and broadcast scalar advantages to policy tokens."""

# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Validate V1 step identities and broadcast scalar advantages to policy tokens.
# Extension point: PPOTrainer step, advantage, actor-update, and FSDP reduction hooks
# DYAD-AGENT-STEPS: shared trainer integration, independent of action policy implementation.
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch


def compute_environment_step_batch(data, batch_keys, estimator, gamma, config, norm_adv_by_std=True):
    from verl_extensions.agent_steps.advantages import _compute_step_components, _finite

    settings = config.get("step_rollout", {})
    if config.get("use_kl_in_reward", False):
        raise ValueError("Environment-step scores cannot silently include token KL rewards")
    if len(batch_keys) != len(data):
        raise ValueError("Environment-step keys must align with batch rows")
    records = data.non_tensor_batch.get("environment_step")
    if records is None:
        raise ValueError("Shared step training requires environment_step metadata")
    padding = data.non_tensor_batch.get("is_padding", np.zeros(len(data), dtype=bool))
    sources = data.non_tensor_batch.get("step_source_key", batch_keys)
    order = data.non_tensor_batch.get("step_occurrence", np.arange(len(data)))
    # Shared V1 balances the complete occurrence batch before computing
    # advantages, as the reference does. Preserve that rank-major order:
    # similarity clustering and first-occurrence episode statistics depend on it.
    # step_occurrence identifies copies/order before balancing, not the order in
    # which these statistics are evaluated. Complete episodes are reconstructed
    # by source identity below and their returns use numeric step indices.
    real = [i for i in range(len(data)) if not padding[i]]
    if len({int(order[i]) for i in real}) != len(real) or any(order[i] < 0 for i in real):
        raise ValueError("Every statistical step occurrence needs a unique nonnegative ordinal")
    required = {
        "group_id",
        "trajectory_id",
        "step_index",
        "env_reward",
        "action_valid",
        "episode_reward",
        "episode_length",
    }
    original, source_indices, sample_indices = [], {}, []
    trajectories = defaultdict(list)
    for row in real:
        record = records[row]
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError("Every environment step must carry complete identity and episode metadata")
        source = sources[row]
        parts = source.rsplit("_", 2)
        if (
            len(parts) != 3
            or not parts[2].isdigit()
            or record["group_id"] != parts[0]
            or record["group_id"] != data.non_tensor_batch["uid"][row]
            or record["trajectory_id"] != f"{parts[0]}_{parts[1]}"
            or record["step_index"] != int(parts[2])
        ):
            raise ValueError("Environment-step identity does not match its source transport key and uid")
        if source not in source_indices:
            source_indices[source] = len(original)
            original.append(record)
            trajectories[record["trajectory_id"]].append(record)
        else:
            previous = original[source_indices[source]]
            for field in required | ({"anchor"} if estimator == "gigpo" else set()):
                if not np.array_equal(record.get(field), previous.get(field)):
                    raise ValueError(f"Statistical copies disagree on source field {field}")
        sample_indices.append(source_indices[source])
    for episode in trajectories.values():
        total = math.fsum(record["env_reward"] for record in episode)
        for record in episode:
            if (
                isinstance(record["episode_length"], bool)
                or record["episode_length"] != len(episode)
                or not math.isclose(record["episode_reward"], total, rel_tol=1e-6, abs_tol=1e-6)
            ):
                raise ValueError("Step metadata must describe complete consistently scored original episodes")
    gigpo = config.get("gigpo", {})
    _, episode_advantages, state_advantages, metrics = _compute_step_components(
        estimator=estimator,
        group_ids=[r["group_id"] for r in original],
        trajectory_ids=[r["trajectory_id"] for r in original],
        step_indices=[r["step_index"] for r in original],
        anchors=[r.get("anchor") for r in original],
        env_rewards=[r["env_reward"] for r in original],
        action_validity=[r["action_valid"] for r in original],
        sample_indices=sample_indices,
        gamma=gamma,
        invalid_action_penalty=settings.get("invalid_action_penalty", 0.1),
        norm_adv_by_std_in_grpo=norm_adv_by_std,
        mode=gigpo.get("mode", "mean_std_norm"),
        step_advantage_w=gigpo.get("step_advantage_w", 1.0),
        enable_similarity=gigpo.get("enable_similarity", False),
        similarity_thresh=gigpo.get("similarity_thresh", 0.95),
        compute_mean_std_cross_steps=settings.get("compute_mean_std_cross_steps", True),
    )
    mask = data.batch["response_mask"]
    if "seq_mask" in data.batch:
        mask = mask * data.batch["seq_mask"].to(mask.dtype)
    # DYAD-REFERENCE-FLOAT32: controller CPU mask each component before combining.
    cpu_mask = _finite(mask.to(device="cpu", dtype=torch.float32), "effective response mask")
    if not torch.all((cpu_mask == 0) | (cpu_mask == 1)):
        raise ValueError("Environment-step effective response mask must be binary")
    token_advantages = episode_advantages[:, None] * cpu_mask[real]
    if estimator == "gigpo":
        masked_state = state_advantages[:, None] * cpu_mask[real]
        weighted_state = _finite(gigpo.get("step_advantage_w", 1.0) * masked_state, "masked weighted state")
        token_advantages = _finite(token_advantages + weighted_state, "masked joint advantages")
    scores = torch.zeros(mask.shape, dtype=torch.float32, device="cpu")
    scores[real] = _finite(token_advantages, "final token advantages")
    data.batch["advantages"] = scores.to(mask.device)
    data.batch["returns"] = data.batch["advantages"].clone()
    episodes = list(trajectories.values())
    metrics["step/episode_reward_mean"] = (
        float(np.mean([ep[0]["episode_reward"] for ep in episodes])) if episodes else 0.0
    )
    metrics["step/episode_length_mean"] = float(np.mean([len(ep) for ep in episodes])) if episodes else 0.0
    data.meta_info["step_metrics"] = metrics
    return data
