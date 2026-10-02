# CodeGym local env tool (in-process env pool backend, replacing the three-tier manager+bridge+env_server HTTP stack)
#
# Lifecycle, state handling and metrics live in agent_system/environments/core/local_tool.py; only the CodeGym-specific
# hooks are here. Instead of HTTP to the bridge(:8200) it calls the in-process CodeGymEnvPool
# (external envs always run inside Ray).
# The adapter (build_action / format_tool_text / get_tool_schema) comes from
# agent_system.environments.backends.codegym.adapter.CodeGymAdapter,
# and the dict shape returned by the env pool (observation/reward/available_actions/done/step_count) matches what
# the adapter expects.
#
# The only difference from AlfworldLocalEnvTool: ALFWorld create takes a game_idx (integer index into a fixed list)
# whereas CodeGym create takes
# an env_str (per-sample string from the dataset's tools_kwargs.codegym_call.create_kwargs.create_payload.env_str).

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from agent_system.environments.backends.codegym.pool import CodeGymEnvPool, _default_envs_dir
from agent_system.environments.core.local_tool import BaseLocalEnvTool
from verl.tools.schemas import ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# All CodeGymLocalEnvTool instances in one process share a single env pool (one set of actors), cached by config
# signature.
_POOL_REGISTRY: dict[tuple, CodeGymEnvPool] = {}


def _get_or_create_pool(config: dict) -> CodeGymEnvPool:
    """Get/create the shared env pool by config signature (start() is not called here; it happens on the first
    async create)."""
    envs_dir = config.get("envs_dir") or os.environ.get("CODEGYM_ENVS_DIR") or _default_envs_dir()
    envs_dir = os.path.expanduser(os.path.expandvars(envs_dir))
    configured_size = config.get("pool_size")
    pool_size = None if configured_size is None else int(configured_size)
    num_cpus = float(config.get("num_cpus_per_worker", 0.25))
    # Explicit overrides are per worker; the default follows the actual dispatched shard.
    _ps = os.environ.get("CODEGYM_ENV_POOL_SIZE", "").strip()
    if _ps:
        pool_size = int(_ps)
    health_env_str = config.get("health_env_str")

    timeouts = CodeGymEnvPool.resolve_timeouts(config)
    action_timeout_s = CodeGymEnvPool.resolve_action_timeout(config)
    key = (envs_dir, pool_size, num_cpus, health_env_str, tuple(sorted(timeouts.items())), action_timeout_s)
    pool = _POOL_REGISTRY.get(key)
    if pool is None:
        pool = CodeGymEnvPool(
            envs_dir=envs_dir,
            pool_size=pool_size,
            num_cpus_per_worker=num_cpus,
            health_env_str=health_env_str,
            timeouts=timeouts,
            action_timeout_s=action_timeout_s,
        )
        _POOL_REGISTRY[key] = pool
    return pool


class CodeGymLocalEnvTool(BaseLocalEnvTool):
    """In-process CodeGym env tool. Lifecycle: create -> execute (repeatedly) -> calc_reward -> release."""

    DEFAULT_ENV_TYPE = "codegym"
    fail_fast_on_error = True
    # Penalises bad actions, unlike alfworld -- see the divergence table in base_local_tool.py.
    INVALID_ACTION_REWARD = -0.05
    ENV_ERROR_REWARD = -0.1

    @staticmethod
    def _resolve_env_str(create_kwargs: dict) -> Optional[str]:
        """Take env_str out of the merged create_kwargs (same convention as the bridge's /create:
        create_payload.env_str)."""
        create_payload = create_kwargs.get("create_payload", {}) or {}
        return create_payload.get("env_str") or create_kwargs.get("env_str")

    def _get_pool(self, config: dict) -> CodeGymEnvPool:
        return _get_or_create_pool(config)

    def _build_reset_spec(self, create_kwargs: dict, call_kwargs: dict) -> dict:
        env_str = self._resolve_env_str(create_kwargs)
        if not env_str:
            raise ValueError(
                "CodeGymLocalEnvTool.create is missing env_str (expected at "
                "tools_kwargs.codegym_call.create_kwargs.create_payload.env_str)"
            )
        return {"env_str": str(env_str)}

    def _state_extra(self, reset_spec: dict) -> dict:
        return {"env_str": reset_spec["env_str"]}

    def _extra_step_metrics(self, step_resp: dict) -> dict:
        return {
            key: step_resp[key]
            for key in ("action_failed", "action_timeout", "action_error_type")
            if key in step_resp
        }

    def _on_step_exception(self, instance_id: str, action_repr: str, exc: Exception) -> None:
        state = self._instance_dict.get(instance_id, {})
        raise RuntimeError(
            f"CodeGym actor failed for session={instance_id!r}, "
            f"env_str={state.get('env_str')!r}, action={action_repr!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    def _on_release_error(self, instance_id: str, exc: Exception) -> None:
        raise RuntimeError(f"Failed to close CodeGym session={instance_id!r}: {exc}") from exc

    # Declared here only so rollout_trace_op labels the span CodeGymLocalEnvTool.execute; see base_local_tool.py.
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return await super().execute(instance_id, parameters, **kwargs)
