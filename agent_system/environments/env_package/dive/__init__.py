"""Standalone dive environment."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .envs import DiveEnv


def build_env(**kwargs: Any) -> "DiveEnv":
    """Construct one environment without allocating rollout workers."""
    from .envs import DiveEnv

    return DiveEnv(**kwargs)
