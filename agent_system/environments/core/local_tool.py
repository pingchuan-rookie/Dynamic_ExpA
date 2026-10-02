# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Shared base class for the in-process local env tools (Calc / ALFWorld / CodeGym).

The three tools present one lifecycle to agent_loop / GRPO / Dyad -- create -> execute* ->
calc_reward -> release -- over an in-process `BaseEnvPool` (external envs always run inside Ray,
there is no HTTP layer). Everything except the env-specific hooks below was verbatim-identical
across the three files before this base existed.

**The exception-path rewards are NOT uniform, and unifying them would silently rewrite a baseline
reward function.** Measured from the code, not from prose:

    | path                        | calc  | alfworld | codegym |
    |-----------------------------|-------|----------|---------|
    | build_action raised         | -0.05 | 0.0      | -0.05   |
    | strict validation rejected  | n/a   | 0.0      | n/a     |
    | pool.step_session raised    | raise | raise    | raise   |

The RPC row describes default fail-fast hooks; calc can explicitly disable its hook.
Action penalties are class attributes so each subclass keeps its own historical values; alfworld's 0.0 comes
with an explicit "pure env outcome reward, no negative shaping" rationale, the other two predate it.
Deciding whether they *should* agree is a research question, not a refactor -- see
../dynamic-expa-design/progress/evidence/w2_env_base_classes.md.

Subclasses override the following (everything else is inherited):
  - DEFAULT_ENV_TYPE          : config's env_type fallback
  - INVALID_ACTION_REWARD / ENV_ERROR_REWARD : the divergence above
  - _get_pool(config)         : the module-level pool registry, keyed by that env's config signature
  - _build_reset_spec(...)    : calc -> ground_truth, alfworld -> game_idx, codegym -> env_str
  - _state_extra(reset_spec)  : the per-env key mirrored into instance state
  - _validate_action(...)     : alfworld's strict available-actions check
  - _on_step_exception(...)   : backend infrastructure failure propagation
  - _env_reports_max_turns(...) : calc also honours the env's own max_turns_reached flag
  - _extra_step_metrics(...)  : calc reports action_name
  - _on_release_error(...)    : backend cleanup failure propagation

`execute` is deliberately left undecorated here and re-declared with `@rollout_trace_op` in each
subclass: the decorator names its trace span `func.__qualname__`, so decorating it once on the base
would relabel all three envs' spans to `BaseLocalEnvTool.execute`. Tracing is off in the debug smoke
runs, so nothing in the acceptance suite would have caught that.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from agent_system.utils.logging import get_dyad_logger
from agent_system.environments.registry import build_adapter

dyad_logger = get_dyad_logger()


class BaseLocalEnvTool(BaseTool):
    """In-process env tool. Lifecycle: create -> execute (repeatedly) -> calc_reward -> release."""

    # ---- specified by subclasses ----
    DEFAULT_ENV_TYPE: str = ""
    # See the divergence table in the module docstring. Do not "tidy" these into agreement.
    INVALID_ACTION_REWARD: float = 0.0
    ENV_ERROR_REWARD: float = 0.0

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        self.config = config
        self.env_type = self.config.get("env_type", self.DEFAULT_ENV_TYPE)
        self.adapter = build_adapter(self.env_type, self.config)
        resolved_schema = tool_schema or self.adapter.get_tool_schema()
        super().__init__(config, resolved_schema)

        self.pool = self._get_pool(self.config)

        self.strict_action_validation = bool(self.config.get("strict_action_validation", False))
        # reward_mode: last = single-step reward, total = accumulated.
        self.reward_mode = self.config.get("reward_mode", "last")
        max_turns = self.config.get("max_turns")
        self.max_turns = int(max_turns) if max_turns is not None else None
        if self.max_turns is not None and self.max_turns <= 0:
            raise ValueError(f"max_turns must be positive, got {self.max_turns}")

        self._default_create_kwargs = dict(self.config.get("create_kwargs", {}) or {})
        # instance_id -> local state
        self._instance_dict: dict[str, dict[str, Any]] = {}

    # =====================================================================
    # Env-specific hooks (overridden by subclasses)
    # =====================================================================
    def _get_pool(self, config: dict) -> Any:
        """Return the process-shared env pool for this config (each env keys its registry differently)."""
        raise NotImplementedError

    def _build_reset_spec(self, create_kwargs: dict, call_kwargs: dict) -> dict:
        """Build the pool's reset spec. `call_kwargs` are create()'s own keyword arguments."""
        raise NotImplementedError

    def _state_extra(self, reset_spec: dict) -> dict:
        """Env-specific key mirrored into instance state (ground_truth / game / env_str)."""
        return {}

    def _validate_action(self, state: dict[str, Any], action_repr: str) -> Optional[tuple[str, dict]]:
        """Reject an action before stepping. Return (text, extra_metrics) to reject, None to proceed."""
        return None

    def _on_step_exception(self, instance_id: str, action_repr: str, exc: Exception) -> None:
        """Called before the env-error reward path; raise here to fail fast instead of returning a reward."""
        return None

    def _env_reports_max_turns(self, step_resp: dict) -> bool:
        """Whether the env itself signalled the turn cap, ORed with this tool's own turn accounting."""
        return False

    def _extra_step_metrics(self, step_resp: dict) -> dict:
        """Extra per-step metric fields."""
        return {}

    def _on_release_error(self, instance_id: str, exc: Exception) -> None:
        """Handle a failure to close the session. State is dropped either way."""
        return None

    # =====================================================================
    # Shared lifecycle
    # =====================================================================
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = dict(self._default_create_kwargs)
        create_kwargs.update(kwargs.get("create_kwargs", {}) or {})
        reset_spec = self._build_reset_spec(create_kwargs, kwargs)

        # Lease a worker + reset (the pool auto-starts on first use, including health self-check + diagnostics).
        reset_resp = await self.pool.create_session(instance_id, reset_spec)

        try:
            state = {
                "observation": "",
                "available_actions": [],
                "done": False,
                "last_reward": 0.0,
                "total_reward": 0.0,
                "step_count": 0,
                "turn_count": 0,
                **self._state_extra(reset_spec),
                "last_action_repr": "<create>",
            }
            self._update_state_from_response(state, reset_resp)

            create_text = self.adapter.format_tool_text(
                observation=state["observation"],
                available_actions=state["available_actions"],
            )
            response = ToolResponse(text=create_text)
        except BaseException:
            # Only a successful pool create grants ownership. A rejected duplicate
            # must never close another tool's lease. Finish cleanup on cancellation.
            await asyncio.shield(self.pool.close_session(instance_id))
            raise
        self._instance_dict[instance_id] = state
        return instance_id, response

    def get_observation(self, instance_id: str) -> str:
        """Return only the raw environment observation, never private reset/task state."""
        observation = self._instance_dict[instance_id]["observation"]
        if not isinstance(observation, str):
            raise ValueError("Environment observation must be textual")
        return observation

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        if instance_id not in self._instance_dict:
            raise KeyError(f"Unknown instance_id={instance_id!r}")
        state = self._instance_dict[instance_id]
        # Raw transition observations are independent of the consuming training algorithm.
        observation_before = self.get_observation(instance_id)

        def observation_metrics(advanced=False):
            return {"observation_before": observation_before,
                    "observation_after": self.get_observation(instance_id),
                    "environment_advanced": advanced}

        if state["done"]:
            text = self.adapter.format_tool_text(
                observation=state["observation"], available_actions=state["available_actions"]
            )
            return ToolResponse(text=text), 0.0, {
                "already_done": True, **self._turn_limit_metrics(state), **observation_metrics(),
            }

        # Every execute counts as one turn (including parse failures / invalid actions).
        state["turn_count"] += 1

        try:
            action = self.adapter.build_action(parameters)
        except Exception as e:
            self._apply_turn_limit(state)
            return ToolResponse(text=f"Invalid action parameters: {e}"), self.INVALID_ACTION_REWARD, {
                "invalid_action": True, "error": str(e), **self._turn_limit_metrics(state), **observation_metrics(),
            }

        action_repr = str(action)
        dyad_logger.debug("model's parsed action_repr: %s", action_repr)

        rejection = self._validate_action(state, action_repr)
        if rejection is not None:
            text, extra_metrics = rejection
            self._apply_turn_limit(state)
            return ToolResponse(text=text), self.INVALID_ACTION_REWARD, {
                "invalid_action": True, **extra_metrics, **self._turn_limit_metrics(state), **observation_metrics(),
            }

        # In-process step (no HTTP any more; env pool faults raise env_actor_error and only affect this one trajectory).
        try:
            step_resp = await self.pool.step_session(instance_id, action)
        except Exception as e:
            self._apply_turn_limit(state)
            error_type = type(e).__name__
            self._on_step_exception(instance_id, action_repr, e)
            text = f"Env step failed for action={action_repr!r}: {error_type}: {e!r}"
            return ToolResponse(text=text), self.ENV_ERROR_REWARD, {
                "env_actor_error": True, "error": str(e), "error_type": error_type,
                "step_count": state["step_count"], **self._turn_limit_metrics(state), **observation_metrics(),
            }

        state["last_action_repr"] = action_repr
        self._update_state_from_response(state, step_resp)
        state["step_count"] += 1
        max_turns_reached = self._apply_turn_limit(state) or self._env_reports_max_turns(step_resp)

        text = self.adapter.format_tool_text(
            observation=state["observation"], available_actions=state["available_actions"]
        )
        if max_turns_reached:
            text = f"{text}\nMaximum environment turns reached ({self.max_turns})."

        tool_reward = float(state["total_reward"]) if self.reward_mode == "total" else float(state["last_reward"])
        metrics = {
            "done": state["done"],
            "won": bool(step_resp.get("won", False)),
            "step_count": state["step_count"],
            "last_reward": state["last_reward"],
            "total_reward": state["total_reward"],
            "last_action_repr": state["last_action_repr"],
            **self._extra_step_metrics(step_resp),
            "available_actions": list(state["available_actions"]),
            "http_failure": False,  # in-process backend: HTTP can never fail (key kept so diagnostics stay aligned with the old schema)
            **self._turn_limit_metrics(state),
            **observation_metrics(advanced=True),
        }
        return ToolResponse(text=text), tool_reward, metrics

    def _apply_turn_limit(self, state: dict[str, Any]) -> bool:
        reached = self.max_turns is not None and state["turn_count"] >= self.max_turns
        if reached:
            state["done"] = True
        return reached

    def _turn_limit_metrics(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "done": bool(state["done"]),
            "turn_count": int(state["turn_count"]),
            "max_turns": self.max_turns,
            "max_turns_reached": bool(self.max_turns is not None and state["turn_count"] >= self.max_turns),
        }

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        state = self._instance_dict.get(instance_id)
        return float(state["total_reward"]) if state is not None else 0.0

    async def release(self, instance_id: str, **kwargs) -> bool:
        if instance_id not in self._instance_dict:
            return True
        try:
            await self.pool.close_session(instance_id)
        except Exception as e:  # noqa: BLE001
            self._on_release_error(instance_id, e)
            # Preserve non-fatal cleanup policy, but expose failures to evaluation.
            return False
        finally:
            del self._instance_dict[instance_id]
        return True

    async def abort(self, instance_id: str, reason: str) -> None:
        """Discard a failed/timed-out session: kill the worker whose state is unknown."""
        self._instance_dict.pop(instance_id, None)
        await self.pool.abort_session(instance_id, reason)

    def _update_state_from_response(self, state: dict[str, Any], response: dict[str, Any]) -> None:
        reward = self.adapter.extract_reward(response)
        done = self.adapter.extract_done(response)
        observation = self.adapter.extract_observation(response)
        available_actions = self.adapter.extract_available_actions(response)
        info = self.adapter.extract_info(response)
        state["last_reward"] = float(reward)
        state["total_reward"] += float(reward)
        state["done"] = bool(done)
        state["observation"] = observation
        state["available_actions"] = available_actions
        state["last_info"] = info
