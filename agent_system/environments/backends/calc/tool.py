# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""GSM8K calculator local env tool (in-process Ray-actor env pool backend).

Lifecycle, state handling and metrics live in agent_system/environments/core/local_tool.py; only the calc-specific
hooks are here. Transport: calls the in-process CalcEnvPool (external envs always run inside Ray);
there is no external HTTP server.
The adapter (build_action / format_tool_text / get_tool_schema) comes from agent_system.environments.backends.calc.adapter.CalcAdapter.

Each sample's ground_truth is forwarded from the dataset's tools_kwargs.calculator.create_kwargs.ground_truth.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from verl.utils.rollout_trace import rollout_trace_op
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from agent_system.environments.core.local_tool import BaseLocalEnvTool
from agent_system.environments.backends.calc.pool import CalcEnvPool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# All CalcLocalEnvTool instances in one process share a single env pool (one set of actors), cached by config signature.
_POOL_REGISTRY: dict[tuple, CalcEnvPool] = {}


def _get_or_create_pool(config: dict) -> CalcEnvPool:
    num_cpus = float(config.get("num_cpus_per_worker", 0.1))
    max_turns = int(config.get("max_turns", 20))
    configured_pool_size = os.environ.get("CALC_ENV_POOL_SIZE", str(config.get("pool_size", "auto"))).strip()
    if configured_pool_size.lower() == "auto":
        pool_size = None
    else:
        pool_size = int(configured_pool_size)
        if pool_size <= 0:
            raise ValueError(f"CALC_ENV_POOL_SIZE/pool_size must be 'auto' or positive, got {configured_pool_size!r}")
    key = (pool_size, num_cpus, max_turns)
    pool = _POOL_REGISTRY.get(key)
    if pool is None:
        pool = CalcEnvPool(
            pool_size=pool_size,
            num_cpus_per_worker=num_cpus,
            max_turns=max_turns,
        )
        _POOL_REGISTRY[key] = pool
    return pool


class CalcLocalEnvTool(BaseLocalEnvTool):
    """In-process GSM8K calculator env tool. Lifecycle: create -> execute (repeatedly) -> calc_reward -> release."""

    DEFAULT_ENV_TYPE = "gsm8k_calc"
    # Penalises bad actions, unlike alfworld -- see the divergence table in base_local_tool.py.
    INVALID_ACTION_REWARD = -0.05
    ENV_ERROR_REWARD = -0.1

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        super().__init__(config, tool_schema)
        self.fail_fast_on_error = bool(self.config.get("fail_fast_on_error", True))
        # Read back through getattr() by dyad_tool_agent_loop to arm a soft timeout; calc only.
        self.execution_timeout_ms = float(self.config.get("execution_timeout_ms", 0) or 0)

    def _get_pool(self, config: dict) -> CalcEnvPool:
        return _get_or_create_pool(config)

    def _build_reset_spec(self, create_kwargs: dict, call_kwargs: dict) -> dict:
        # ground_truth precedence: explicit argument > create_kwargs.ground_truth.
        ground_truth = call_kwargs.get("ground_truth")
        if ground_truth is None:
            ground_truth = create_kwargs.get("ground_truth")
        gt = None if ground_truth is None else str(ground_truth)
        return {"ground_truth": gt, "max_turns": self.max_turns}

    def _state_extra(self, reset_spec: dict) -> dict:
        return {"ground_truth": reset_spec["ground_truth"]}

    def _on_step_exception(self, instance_id: str, action_repr: str, exc: Exception) -> None:
        if self.fail_fast_on_error:
            raise RuntimeError(
                f"Calc external env actor failed for instance_id={instance_id!r}, action={action_repr!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _env_reports_max_turns(self, step_resp: dict) -> bool:
        return bool(step_resp.get("max_turns_reached"))

    def _extra_step_metrics(self, step_resp: dict) -> dict:
        return {"action_name": step_resp.get("action_name")}

    def _on_release_error(self, instance_id: str, exc: Exception) -> None:
        if self.fail_fast_on_error:
            raise RuntimeError(f"Failed to close calc external env session {instance_id!r}: {exc}") from exc
        logger.warning("Failed to close session %s: %s", instance_id, exc)

    # Declared here only so rollout_trace_op labels the span CalcLocalEnvTool.execute; see base_local_tool.py.
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return await super().execute(instance_id, parameters, **kwargs)
