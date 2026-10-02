"""Reference GRPO/GiGPO statistics over independent environment decisions.

Original decisions describe complete trajectories. Statistical occurrences refer
back to those decisions, so real-row resampling cannot change discounted returns.
Transport-only padding is not an occurrence and must be excluded by the caller.
"""

# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Reference GRPO/GiGPO statistics over independent environment decisions.
# Extension point: PPOTrainer step, advantage, actor-update, and FSDP reduction hooks
# DYAD-AGENT-STEPS: shared GRPO/GiGPO numerical core, not an Dyad policy component.
from __future__ import annotations

import math
from collections import defaultdict
from difflib import SequenceMatcher
from numbers import Integral, Real

import numpy as np
import torch

_EPSILON = 1e-6


def _number(value, name):
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _column(values, name, count=None, *, objects=False):
    if not isinstance(values, list | tuple | np.ndarray) or (isinstance(values, np.ndarray) and values.ndim != 1):
        raise ValueError(f"{name} must be a one-dimensional column")
    values = list(values)
    if not objects and any(isinstance(value, list | tuple | np.ndarray | dict) for value in values):
        raise ValueError(f"{name} must be a one-dimensional scalar column")
    if count is not None and len(values) != count:
        raise ValueError(f"{name} must contain {count} original decisions")
    return values


def _identity(value, name):
    if isinstance(value, np.generic):
        value = value.item()
    if (
        isinstance(value, bool)
        or not isinstance(value, str | int | float)
        or isinstance(value, float)
        and not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a string or finite numeric identity")
    return value


def anchor_to_hashable(value):
    """Match verl-agent's exact-state conversion, including flattened arrays."""
    if isinstance(value, int | float | str | bool):
        return value
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(value.flatten())
    if isinstance(value, list | tuple):
        return tuple(anchor_to_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, anchor_to_hashable(item)) for key, item in value.items()))
    raise TypeError(f"Unsupported anchor type: {type(value)}")


def reference_sample_indices(size: int, multiple: int, rng=None) -> np.ndarray:
    """Copy real rows to a multiple, without the reference's small-batch failure.

    The normal case samples without replacement, as in ``adjust_batch``.
    When more copies than source rows are needed, replacement is necessary.
    Pass an owned Generator/RandomState to make resampling reproducible.
    """
    for name, value in (("size", size), ("multiple", multiple)):
        if isinstance(value, bool | np.bool_) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    indices = np.arange(size, dtype=np.int64)
    extra = (-size) % multiple
    if not extra:
        return indices
    rng = np.random.default_rng() if rng is None else rng
    return np.concatenate((indices, rng.choice(size, extra, replace=extra > size)))


def _finite(value, name):
    finite = torch.isfinite(value).all().item() if isinstance(value, torch.Tensor) else np.isfinite(value).all()
    if not finite:
        raise ValueError(f"{name} must be finite in reference float32 arithmetic")
    return value


def _normalize(values, groups, *, divide_std, episode, samples=None):
    # DYAD-REFERENCE-FLOAT32: match controller CPU reductions, including [1,N] std.
    result = torch.zeros(len(values), dtype=torch.float32, device="cpu")
    with torch.no_grad():
        for key, rows in groups.items():
            selected = rows if samples is None else samples[key]
            scalars = [values[row] for row in selected]
            if len(selected) == 1:
                mean = (
                    torch.tensor(0.0, dtype=torch.float32, device="cpu")
                    if episode
                    else torch.mean(torch.tensor(scalars, dtype=torch.float32, device="cpu"))
                )
                std = torch.tensor(1.0, dtype=torch.float32, device="cpu")
            else:
                scores = torch.tensor(scalars, dtype=torch.float32, device="cpu")
                # DYAD-NUMERICS: exactly equal scores have exactly zero variance.
                # FP32 reduction of repeated -0.1 can round its mean away from
                # every member; division by epsilon then invents an advantage.
                # Keep singleton semantics and the reference nonconstant path.
                if torch.all(scores == scores[0]):
                    mean = scores[0]
                    std = torch.zeros((), dtype=torch.float32, device="cpu")
                else:
                    mean = torch.mean(scores)
                    std = torch.std(scores.unsqueeze(0))
            _finite(mean, "group mean")
            _finite(std, "group std")
            denominator = _finite(std + _EPSILON, "normalization denominator") if divide_std else None
            for row in rows:
                centered = _finite(values[row] - mean, "centered score")
                result[row] = _finite(centered / denominator, "normalized score") if divide_std else centered
    return result


def _state_groups(groups, anchors, *, enable_similarity, similarity_thresh):
    states = defaultdict(list)
    representatives = defaultdict(list)
    for row, (group, anchor) in enumerate(zip(groups, anchors, strict=True)):
        if enable_similarity:
            if not isinstance(anchor, str):
                raise ValueError("similarity grouping requires text anchors")
            clusters = representatives[group]
            for cluster, representative in enumerate(clusters):
                if SequenceMatcher(None, anchor, representative).ratio() >= similarity_thresh:
                    break
            else:
                cluster = len(clusters)
                clusters.append(anchor)
            key = (group, cluster)
        else:
            key = (group, anchor_to_hashable(anchor))
        try:
            states[key].append(row)
        except TypeError as exc:
            raise ValueError("anchor conversion must produce a hashable state") from exc
    return states


def _compute_step_components(
    *,
    estimator,
    group_ids,
    trajectory_ids,
    step_indices,
    env_rewards,
    action_validity,
    anchors=None,
    sample_indices=None,
    gamma=0.95,
    step_advantage_w=1.0,
    mode="mean_std_norm",
    invalid_action_penalty=0.1,
    norm_adv_by_std_in_grpo=True,
    compute_mean_std_cross_steps=True,
    enable_similarity=False,
    similarity_thresh=0.95,
):
    """Compute CPU float32 components once, in statistical-occurrence order.

    Raw input rows must uniquely identify every zero-based step of each complete
    trajectory, but may be interleaved/shuffled. ``sample_indices`` references
    those rows in training order, with repetitions allowed. Returns and totals
    are computed before selecting occurrences and subtracting invalid penalties.

    Episode statistics count occurrences by default. Optional trajectory dedup
    selects the first *occurrence* per trajectory for statistics, just as the
    reference library does, even when invalid penalties differ across steps.
    Similarity grouping also follows occurrence order, not trajectory order.

    GRPO only needs episode scores: anchors, gamma, mode, step weight and
    similarity settings are unused. No RTG/state computation occurs for GRPO.
    An empty generated response remains an occurrence; token broadcasting later
    gives it no gradient, without dropping it from the group statistics.
    """
    if estimator not in ("grpo", "gigpo"):
        raise ValueError("estimator must be grpo or gigpo")
    for name, value in (
        ("compute_mean_std_cross_steps", compute_mean_std_cross_steps),
        ("norm_adv_by_std_in_grpo", norm_adv_by_std_in_grpo),
    ):
        if not isinstance(value, bool | np.bool_):
            raise ValueError(f"{name} must be boolean")
    penalty = _number(invalid_action_penalty, "invalid_action_penalty")
    if penalty < 0:
        raise ValueError("invalid_action_penalty must be nonnegative")
    if estimator == "gigpo":
        gamma = _number(gamma, "gamma")
        weight = _number(step_advantage_w, "step_advantage_w")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if weight < 0:
            raise ValueError("step_advantage_w must be nonnegative")
        if mode not in ("mean_norm", "mean_std_norm"):
            raise ValueError(f"unsupported mode: {mode}")
        if not isinstance(enable_similarity, bool | np.bool_):
            raise ValueError("enable_similarity must be boolean")
        if enable_similarity:
            similarity_thresh = _number(similarity_thresh, "similarity_thresh")
            if not 0 < similarity_thresh < 1:
                raise ValueError("similarity_thresh must be in (0, 1)")

    groups = _column(group_ids, "group_ids")
    count = len(groups)
    trajectories = _column(trajectory_ids, "trajectory_ids", count)
    steps = _column(step_indices, "step_indices", count)
    rewards = _column(env_rewards, "env_rewards", count)
    validity = _column(action_validity, "action_validity", count)
    if estimator == "gigpo":
        anchors = _column(anchors, "anchors", count, objects=True)
    ordered_trajectories = defaultdict(list)
    trajectory_groups = {}
    for row in range(count):
        groups[row] = _identity(groups[row], "group_ids")
        trajectories[row] = _identity(trajectories[row], "trajectory_ids")
        trajectory = trajectories[row]
        if trajectory in trajectory_groups and trajectory_groups[trajectory] != groups[row]:
            raise ValueError("each trajectory identity must belong to one task group")
        trajectory_groups[trajectory] = groups[row]
        if isinstance(steps[row], bool | np.bool_) or not isinstance(steps[row], Integral) or steps[row] < 0:
            raise ValueError("step_indices must be nonnegative integers")
        if not isinstance(validity[row], Real | bool | np.bool_) or validity[row] not in (0, 1):
            raise ValueError("action_validity must be boolean or numeric 0/1")
        rewards[row] = _number(rewards[row], "env_rewards")
        ordered_trajectories[trajectory].append((int(steps[row]), row))

    totals = np.zeros(count, dtype=np.float32)
    returns = np.zeros(count, dtype=np.float32) if estimator == "gigpo" else None
    with np.errstate(over="ignore", invalid="ignore"):
        float_rewards = _finite(np.asarray(rewards, dtype=np.float32), "reward cast")
        for trajectory in ordered_trajectories.values():
            trajectory.sort()
            if any(step != expected for expected, (step, _) in enumerate(trajectory)):
                raise ValueError("trajectory step indices must be unique and contiguous starting at 0")
            # The reference collector stores each addition back to a float32 array.
            total = np.zeros(1, dtype=np.float32)
            for _, row in trajectory:
                total[:] += np.asarray(rewards[row : row + 1], dtype=np.float64)
                _finite(total, "episode accumulation")
            running = 0  # Preserve upstream NumPy scalar promotion, not Python float64.
            for _, row in reversed(trajectory):
                totals[row] = total[0]
                if returns is not None:
                    discounted = _finite(gamma * running, "discounted future")
                    running = _finite(float_rewards[row] + discounted, "discounted return")
                    returns[row] = running
                    _finite(returns[row], "discounted return cast")

    selected = list(range(count)) if sample_indices is None else _column(sample_indices, "sample_indices")
    for index in selected:
        if isinstance(index, bool | np.bool_) or not isinstance(index, Integral) or not 0 <= index < count:
            raise ValueError("sample_indices must reference original decisions with integer indices")
    selected = np.asarray(selected, dtype=np.int64)
    selected_groups = [groups[row] for row in selected]
    selected_trajectories = [trajectories[row] for row in selected]
    invalid_values = torch.tensor([1.0 - validity[row] for row in selected], dtype=torch.float32, device="cpu")
    penalties = _finite(penalty * invalid_values, "penalty multiplication")
    episode_values = _finite(torch.from_numpy(totals[selected]) - penalties, "penalized episode rewards")
    episode_groups = defaultdict(list)
    episode_samples = defaultdict(list)
    seen = set()
    for row, (group, trajectory) in enumerate(zip(selected_groups, selected_trajectories, strict=True)):
        episode_groups[group].append(row)
        if compute_mean_std_cross_steps or (group, trajectory) not in seen:
            episode_samples[group].append(row)
            seen.add((group, trajectory))
    divide_std = norm_adv_by_std_in_grpo if estimator == "grpo" else mode == "mean_std_norm"
    episode_advantages = _normalize(
        episode_values, episode_groups, divide_std=divide_std, episode=True, samples=episode_samples
    )
    state_advantages = torch.zeros(len(selected), dtype=torch.float32, device="cpu")
    states = {}
    if estimator == "gigpo":
        states = _state_groups(
            selected_groups,
            [anchors[row] for row in selected],
            enable_similarity=enable_similarity,
            similarity_thresh=similarity_thresh,
        )
        state_values = _finite(torch.from_numpy(returns[selected]) - penalties, "penalized discounted returns")
        state_advantages = _normalize(state_values, states, divide_std=divide_std, episode=False)
        weighted_state = _finite(weight * state_advantages, "weighted state advantages")
        joint = _finite(episode_advantages + weighted_state, "normalized advantages")
    else:
        joint = episode_advantages

    size = len(selected)
    invalid = sum(not validity[row] for row in selected)
    matched = sum(len(rows) for rows in states.values() if len(rows) > 1)
    metrics = {
        "original_steps": float(count),
        "steps": float(size),
        "resampled_copies": float(size - len(set(selected))),
        "episode_groups": float(len(episode_groups)),
        "trajectories": float(len(set(selected_trajectories))),
        "invalid_steps": float(invalid),
        "invalid_step_ratio": invalid / size if size else 0.0,
    }
    if estimator == "gigpo":
        metrics.update(
            state_groups=float(len(states)),
            matched_steps=float(matched),
            matched_step_ratio=matched / size if size else 0.0,
            state_group_size_mean=size / len(states) if states else 0.0,
            state_group_size_max=float(max(map(len, states.values()), default=0)),
        )
    with np.errstate(over="ignore", invalid="ignore"):
        components = [("episode_adv", episode_advantages), ("adv", joint)]
        if estimator == "gigpo":
            components.append(("step_adv", state_advantages))
        for name, values in components:
            # Preserve population-std diagnostic semantics; metrics do not feed training.
            diagnostic = values.numpy().astype(np.float64)
            metrics[f"{name}_mean"] = float(diagnostic.mean()) if size else 0.0
            metrics[f"{name}_std"] = float(diagnostic.std()) if size else 0.0
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("advantage statistics must be finite")
    return (
        joint,
        episode_advantages,
        state_advantages,
        {f"{estimator}/{name}": value for name, value in metrics.items()},
    )


def compute_step_advantages(
    *,
    estimator,
    group_ids,
    trajectory_ids,
    step_indices,
    env_rewards,
    action_validity,
    anchors=None,
    sample_indices=None,
    gamma=0.95,
    step_advantage_w=1.0,
    mode="mean_std_norm",
    invalid_action_penalty=0.1,
    norm_adv_by_std_in_grpo=True,
    compute_mean_std_cross_steps=True,
    enable_similarity=False,
    similarity_thresh=0.95,
):
    """Return reference-rounded scalars as float64 NumPy plus existing metrics.

    The public array dtype is retained for caller compatibility; every training
    arithmetic stage is CPU float32. Transport-only padding is excluded upstream.
    """
    joint, _, _, metrics = _compute_step_components(
        estimator=estimator,
        group_ids=group_ids,
        trajectory_ids=trajectory_ids,
        step_indices=step_indices,
        env_rewards=env_rewards,
        action_validity=action_validity,
        anchors=anchors,
        sample_indices=sample_indices,
        gamma=gamma,
        step_advantage_w=step_advantage_w,
        mode=mode,
        invalid_action_penalty=invalid_action_penalty,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        compute_mean_std_cross_steps=compute_mean_std_cross_steps,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
    )
    return joint.numpy().astype(np.float64), metrics
