"""WebShop native tool lifecycle with isolated session leases and official scoring."""
from __future__ import annotations

import json
import os
from typing import Any

from verl.utils.rollout_trace import rollout_trace_op

from agent_system.environments.core.local_tool import BaseLocalEnvTool
from agent_system.environments.backends.webshop.config import resolve_webshop_config
from agent_system.environments.backends.webshop.pool import WebShopEnvPool


_POOL_REGISTRY: dict[str, WebShopEnvPool] = {}


def _get_or_create_pool(config):
    key = json.dumps(resolve_webshop_config(config), sort_keys=True, allow_nan=False)
    pool = _POOL_REGISTRY.get(key)
    if pool is None:
        pool = WebShopEnvPool(config)
        _POOL_REGISTRY[key] = pool
    return pool


class WebShopLocalEnvTool(BaseLocalEnvTool):
    DEFAULT_ENV_TYPE = "webshop"
    # Both real rollout loops otherwise turn backend/cleanup exceptions into
    # ordinary zero-reward tool results. Infrastructure failure is not a score.
    fail_fast_on_error = True
    INVALID_ACTION_REWARD = 0.0
    ENV_ERROR_REWARD = 0.0

    def __init__(self, config, tool_schema=None):
        config = dict(config)
        config["max_turns"] = int(os.environ.get("WEBSHOP_MAX_STEPS") or config.get("max_turns", 100))
        if config.get("reward_mode", "last") != "last":
            raise ValueError("WebShop requires reward_mode=last (official outcome score)")
        super().__init__(config, tool_schema)
        self._creating: set[str] = set()

    def _get_pool(self, config):
        return _get_or_create_pool(config)

    def _build_reset_spec(self, create_kwargs: dict, call_kwargs: dict) -> dict:
        payload = dict(create_kwargs.get("reset_payload", {}) or {})
        payload.update({k: v for k, v in create_kwargs.items() if k not in {"reset_payload", "auto_reset", "create_payload"}})
        if "task_id" not in payload:
            raise ValueError("WebShop create_kwargs requires the global human-goal task_id")
        task_id = payload["task_id"]
        if isinstance(task_id, bool) or str(task_id) != str(int(task_id)) or int(task_id) < 0:
            raise ValueError("WebShop task_id must be a nonnegative integer")
        task_id = int(task_id)
        split = payload.get("split")
        if split is not None:
            bounds = {"test": (0, 500), "dev": (500, 1500), "train": (1500, None)}
            if split not in bounds:
                raise ValueError("WebShop split must be test, dev or train")
            low, high = bounds[split]
            if task_id < low or (high is not None and task_id >= high):
                raise ValueError(f"WebShop task_id={task_id} does not belong to split={split}")
        result = {"task_id": task_id}
        if split is not None:
            result["split"] = split
        if "initial_observation" in payload:
            if not isinstance(payload["initial_observation"], str):
                raise ValueError("initial_observation must be a string")
            result["initial_observation"] = payload["initial_observation"]
        return result

    def _state_extra(self, reset_spec):
        return {"task_id": reset_spec["task_id"], "task_score": 0.0, "won": False}

    def _validate_action(self, state: dict[str, Any], action_repr: str):
        if action_repr.startswith("search["):
            # Searches are open text, not a finite candidate list. The native
            # environment handles search availability, including no-op pages.
            return None
        # Native WebShop lowercases action arguments before clickable lookup.
        target = action_repr[len("click["):-1].lower()
        available = state["available_actions"]
        if target != "search" and f"click[{target}]" in available:
            return None
        return "Invalid WebShop click target: choose a visible clickable.", {"strict_action_validation": True}

    def _update_state_from_response(self, state, response):
        super()._update_state_from_response(state, response)
        score = float(response.get("task_score", state["last_reward"]))
        # Official terminal outcome is a score, not sum of repeated score reports.
        if not 0.0 <= score <= 1.0:
            raise ValueError("WebShop task_score must be in [0, 1]")
        state["task_score"] = score
        state["total_reward"] = score
        state["won"] = bool(response.get("done")) and score == 1.0
        if bool(response.get("won", False)) != state["won"]:
            raise ValueError("WebShop won must mean terminal full success, not positive partial reward")

    def _extra_step_metrics(self, step_resp):
        return {"task_score": float(step_resp.get("task_score", step_resp.get("reward", 0.0)))}

    def _on_step_exception(self, instance_id, action_repr, exc):
        # Backend failures are not legitimate failed purchases or zero-score data.
        raise RuntimeError(f"WebShop session {instance_id!r} backend failed: {exc}") from exc

    def _on_release_error(self, instance_id, exc):
        raise RuntimeError(f"WebShop session {instance_id!r} cleanup failed: {exc}") from exc

    async def create(self, instance_id=None, **kwargs):
        # Track ownership explicitly: a duplicate rejected by a process-shared
        # pool must not cause this tool to release another tool's existing lease.
        import asyncio
        from uuid import uuid4
        from verl.tools.schemas import ToolResponse
        instance_id = instance_id or str(uuid4())
        if instance_id in self._instance_dict or instance_id in self._creating:
            raise ValueError(f"duplicate active instance_id={instance_id!r}")
        self._creating.add(instance_id)
        acquired = False
        try:
            create_kwargs = dict(self._default_create_kwargs)
            create_kwargs.update(kwargs.get("create_kwargs", {}) or {})
            spec = self._build_reset_spec(create_kwargs, kwargs)
            response = await self.pool.create_session(instance_id, spec)
            acquired = True
            state = {
                "observation": "", "available_actions": [], "done": False,
                "last_reward": 0.0, "total_reward": 0.0,
                "step_count": 0, "turn_count": 0, "last_action_repr": "<create>",
                **self._state_extra(spec),
            }
            self._update_state_from_response(state, response)
            text = self.adapter.format_tool_text(
                observation=state["observation"], available_actions=state["available_actions"],
            )
            self._instance_dict[instance_id] = state
            return instance_id, ToolResponse(text=text)
        except BaseException:
            if acquired:
                await asyncio.shield(self.pool.close_session(instance_id))
            raise
        finally:
            self._creating.discard(instance_id)

    @rollout_trace_op
    async def execute(self, instance_id, parameters, **kwargs):
        return await super().execute(instance_id, parameters, **kwargs)

    async def abort(self, instance_id, reason):
        if instance_id not in self._instance_dict and instance_id not in self._creating:
            return
        await super().abort(instance_id, reason)

    async def calc_reward(self, instance_id, **kwargs):
        return float(self._instance_dict.get(instance_id, {}).get("task_score", 0.0))
