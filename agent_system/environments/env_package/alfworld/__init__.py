"""Standalone alfworld environment."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .envs import AlfworldEnv


def build_env(**kwargs: Any) -> "AlfworldEnv":
    """Construct one environment without allocating rollout workers."""
    from .envs import AlfworldEnv

    return AlfworldEnv(**kwargs)
