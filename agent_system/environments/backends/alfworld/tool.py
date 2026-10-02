# ALFWorld local env tool (in-process env pool backend, env runs inside verl's Ray).
#
# Lifecycle, state handling and metrics live in agent_system/environments/core/local_tool.py; only the ALFWorld-specific
# hooks are here. Transport: calls the in-process AlfworldEnvPool (external envs always run inside Ray);
# no external HTTP server.
# The adapter (build_action / format_tool_text / get_tool_schema) lives in
# agent_system/environments/backends/alfworld/adapter.py + registry.py.

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from agent_system.environments.backends.alfworld.pool import AlfworldEnvPool
from agent_system.environments.core.local_tool import BaseLocalEnvTool
from verl.tools.schemas import ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# All AlfworldLocalEnvTool instances in one process share a single env pool (one set of actors), cached by config
# signature.
_POOL_REGISTRY: dict[tuple, AlfworldEnvPool] = {}

# Default config/mappings shipped inside the package (keeps the training side self-contained and
# cluster-portable, with no dependency on the external reference directory).
_DEFAULT_CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs")


def _get_or_create_pool(config: dict) -> AlfworldEnvPool:
    """
    Get/create the shared env pool by config signature (start() is not called here; it happens on the first async
    create).
    """
    config_path = config.get("alfworld_config_path") or os.path.join(_DEFAULT_CFG_DIR, "alfworld_base_config.yaml")
    alfworld_data = config.get("alfworld_data") or os.environ.get("ALFWORLD_DATA") or "~/.cache/alfworld"
    train_mapping = config.get("train_mapping") or os.path.join(_DEFAULT_CFG_DIR, "alfworld_mappings_train.json")
    test_mapping = config.get("test_mapping") or os.path.join(_DEFAULT_CFG_DIR, "alfworld_mappings_test.json")
    # Optional OOD games: defaults to configs/alfworld_mappings_unseen.json (added only if present).
    # They are appended after train+test (index 3692+), so only training/validation that actually uses
    # valid_unseen (OOD) data hits those indices; train/test data never references them, hence loading
    # them has no side effect.
    unseen_mapping = config.get("unseen_mapping") or os.path.join(_DEFAULT_CFG_DIR, "alfworld_mappings_unseen.json")
    if not os.path.exists(unseen_mapping):
        unseen_mapping = None
    configured_size = config.get("pool_size")
    pool_size = None if configured_size is None else int(configured_size)
    num_cpus = float(config.get("num_cpus_per_worker", 0.1))
    # An explicit override is a per-agent-worker capacity contract, not a global budget.
    _ps = os.environ.get("ALFWORLD_ENV_POOL_SIZE", "").strip()
    if _ps:
        pool_size = int(_ps)
    timeouts = AlfworldEnvPool.resolve_timeouts(config)
    key = (config_path, os.path.expanduser(os.path.expandvars(alfworld_data)), train_mapping,
           test_mapping, unseen_mapping, pool_size, num_cpus, tuple(sorted(timeouts.items())))
    pool = _POOL_REGISTRY.get(key)
    if pool is None:
        pool = AlfworldEnvPool(
            config_path=config_path,
            alfworld_data=alfworld_data,
            train_mapping=train_mapping,
            test_mapping=test_mapping,
            pool_size=pool_size,
            num_cpus_per_worker=num_cpus,
            unseen_mapping=unseen_mapping,
            timeouts={f"{method}_timeout_s": value for method, value in timeouts.items()},
        )
        _POOL_REGISTRY[key] = pool
    return pool


class AlfworldLocalEnvTool(BaseLocalEnvTool):
    """In-process ALFWorld env tool. Lifecycle: create -> execute (repeatedly) -> calc_reward -> release."""

    DEFAULT_ENV_TYPE = "alfworld"
    fail_fast_on_error = True
    # Invalid actions retain zero shaping; infrastructure faults raise instead of becoming rewards.
    INVALID_ACTION_REWARD = 0.0
    ENV_ERROR_REWARD = 0.0

    def _get_pool(self, config: dict) -> AlfworldEnvPool:
        return _get_or_create_pool(config)

    def _build_reset_spec(self, create_kwargs: dict, call_kwargs: dict) -> dict:
        # Resolve the game / world_type used by reset (same convention as the old HTTP tool).
        reset_payload = dict(create_kwargs.get("reset_payload", {}) or {})
        reset_payload.update(
            {k: v for k, v in create_kwargs.items() if k not in {"create_payload", "reset_payload", "auto_reset"}}
        )
        return {
            "game_idx": int(reset_payload.get("game", 0)),
            "world_type": str(reset_payload.get("world_type", "Text")),
        }

    def _state_extra(self, reset_spec: dict) -> dict:
        return {"game": reset_spec["game_idx"]}

    def _validate_action(self, state: dict[str, Any], action_repr: str) -> Optional[tuple[str, dict]]:
        if not (self.strict_action_validation and state["available_actions"]):
            return None
        available = [str(x) for x in state["available_actions"]]
        if action_repr in available:
            return None
        text = (
            f"Action not in available_actions: {action_repr}\n"
            "available_actions:\n" + "\n".join(f"- {a}" for a in available)
        )
        return text, {"strict_action_validation": True}

    def _on_release_error(self, instance_id: str, exc: Exception) -> None:
        raise RuntimeError(f"ALFWorld failed to close session {instance_id}") from exc

    def _on_step_exception(self, instance_id: str, action_repr: str, exc: Exception) -> None:
        raise RuntimeError(f"ALFWorld environment RPC failed for session {instance_id}") from exc

    # Declared here only so rollout_trace_op labels the span AlfworldLocalEnvTool.execute; see base_local_tool.py.
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return await super().execute(instance_id, parameters, **kwargs)
