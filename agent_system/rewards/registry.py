"""Environment reward dispatch used by the verl reward fallback integration."""

SOURCES = frozenset({"alfworld", "ALFWorld", "alfworld_env", "codegym", "codegym_v1", "codegym_gym_v1"})


def supports_environment_reward(data_source: str) -> bool:
    return data_source in SOURCES


def compute_environment_reward(data_source: str, solution_str, ground_truth, extra_info, **kwargs):
    """Preserve each environment's existing reward contract and aliases."""
    if data_source in {"alfworld", "ALFWorld", "alfworld_env"}:
        from agent_system.rewards import alfworld_reward as reward
    elif data_source in {"codegym", "codegym_v1", "codegym_gym_v1"}:
        from agent_system.rewards import codegym_reward as reward
    else:
        raise ValueError(f"Unsupported environment reward source: {data_source}")
    return reward.compute_score(solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info, **kwargs)
