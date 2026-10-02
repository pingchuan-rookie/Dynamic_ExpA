"""Standalone webshop environment."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .envs import WebShopEnv


def build_env(**kwargs: Any) -> "WebShopEnv":
    """Construct one environment without allocating rollout workers."""
    from .envs import WebShopEnv

    return WebShopEnv(**kwargs)
