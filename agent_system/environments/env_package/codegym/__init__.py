"""Standalone codegym environment."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .envs import CodeGymEnv


def build_env(**kwargs: Any) -> "CodeGymEnv":
    """Construct one environment without allocating rollout workers."""
    from .envs import CodeGymEnv

    return CodeGymEnv(**kwargs)
