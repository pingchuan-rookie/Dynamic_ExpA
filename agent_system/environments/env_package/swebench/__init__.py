"""Standalone swebench environment."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .envs import SweBenchEnv


def build_env(**kwargs: Any) -> "SweBenchEnv":
    """Construct one environment without allocating rollout workers."""
    from .envs import SweBenchEnv

    return SweBenchEnv(**kwargs)
