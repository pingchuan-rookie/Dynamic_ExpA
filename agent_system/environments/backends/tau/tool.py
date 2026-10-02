"""Runtime-session tool transport, separate from legacy observation/reward tools."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

from agent_system.environments.backends.tau.adapter import TauAdapter
from agent_system.environments.backends.tau.pool import TauEnvPool

_POOL_REGISTRY = {}


def _get_pool(config):
    # Only a digest is retained as the registry key; config may contain credentials.
    signature = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    key = (asyncio.get_running_loop(), signature)
    if key not in _POOL_REGISTRY:
        _POOL_REGISTRY[key] = TauEnvPool(config)
    return _POOL_REGISTRY[key]


class TauLocalEnvTool(BaseTool):
    runtime_session = True
    protocol = 't2bench'

    def __init__(self, config, tool_schema=None):
        self.adapter = TauAdapter(config)
        super().__init__(config, tool_schema or self.adapter.get_tool_schema())
        self._pool = None
        self._contexts = {}
        self._specs = {}
        self._owned_sessions = set()

    @staticmethod
    def _context(response, previous=None):
        context = deepcopy(previous or {})
        context.update(deepcopy(response))
        if response.get("result") is not None:
            context["episode_result"] = deepcopy(response["result"])
        elif response.get("episode_result") is not None:
            context["result"] = deepcopy(response["episode_result"])
        return context

    def _failure(self, instance_id, exc, phase):
        spec = self._specs[instance_id]
        result = {k: spec[k] for k in ("benchmark", "domain", "split", "task_id", "trial", "seed", "episode_id") if k in spec}
        result.update(status="infra_error", official_scored=False, metric_valid=False,
                      official_reward=None, attempt_reward=None, termination_reason=phase,
                      error={"phase": phase, "type": type(exc).__name__})
        response = {"done": True, "result": result, "messages_delta": []}
        context = self._context(response, self._contexts.get(instance_id))
        context.setdefault("initial_messages", [])
        context.setdefault("action_tools", [])
        context.setdefault("schema_hash", None)
        self._contexts[instance_id] = context
        return deepcopy(context)

    async def bootstrap(self, instance_id=None, *, create_kwargs=None, **kwargs):
        instance_id = instance_id or str(uuid4())
        if instance_id in self._specs:
            raise ValueError("Duplicate tau tool instance")
        params = dict(create_kwargs or {})
        params.update(kwargs)
        spec = deepcopy(params["create_payload"] if "create_payload" in params else params)
        if not isinstance(spec, dict):
            raise TypeError("Tau create_payload must be a JSON object")
        self._specs[instance_id] = spec
        try:
            if self._pool is None:
                self._pool = _get_pool(self.config)
            response = await self._pool.create_session(instance_id, spec)
            self._owned_sessions.add(instance_id)
            self._contexts[instance_id] = self._context(response)
        except asyncio.CancelledError:
            await self.abort(instance_id, "reset_cancelled")
            raise
        except Exception as exc:
            self._failure(instance_id, exc, "reset")
        return instance_id, self.get_session_context(instance_id)

    async def create(self, instance_id=None, **kwargs):
        instance_id, _ = await self.bootstrap(instance_id, **kwargs)
        return instance_id, ToolResponse()

    def get_session_context(self, instance_id):
        return deepcopy(self._contexts[instance_id])

    async def step(self, instance_id, action):
        context = self._contexts[instance_id]
        if context.get("done"):
            raise RuntimeError("Cannot step a terminal tau session")
        action = self.adapter.build_action(action)
        try:
            response = await self._pool.step_session(instance_id, action)
            self._contexts[instance_id] = self._context(response, context)
        except asyncio.CancelledError:
            await self.abort(instance_id, "step_cancelled")
            raise
        except Exception as exc:
            self._failure(instance_id, exc, "step")
        return self.get_session_context(instance_id)

    async def execute(self, instance_id, parameters, **kwargs):
        context = await self.step(instance_id, parameters)
        # Legacy loop API only. This zero is transport padding, never official reward.
        return ToolResponse(), 0.0, {"done": bool(context.get("done")),
                                     "episode_result": context.get("episode_result")}

    async def finalize(self, instance_id, reason="rollout_terminated"):
        context = self._contexts[instance_id]
        if context.get("result") is not None:
            return deepcopy(context["result"])
        try:
            response = await self._pool.finalize_session(instance_id, reason)
            # Accept the session response envelope, not an implicit step reward.
            self._contexts[instance_id] = self._context(response, context)
            result = self._contexts[instance_id].get("result")
            if result is None:
                raise ValueError("Tau finalize returned no terminal result")
        except asyncio.CancelledError:
            await self.abort(instance_id, "finalize_cancelled")
            raise
        except Exception as exc:
            self._failure(instance_id, exc, "finalize")
        return deepcopy(self._contexts[instance_id]["result"])

    async def calc_reward(self, instance_id, **kwargs):
        # An unscored episode stays null. Do not call this through legacy score sums.
        return (self._contexts[instance_id].get("result") or {}).get("official_reward")

    async def release(self, instance_id, **kwargs):
        try:
            if self._pool is not None and instance_id in self._owned_sessions:
                await self._pool.close_session(instance_id)
        finally:
            self._contexts.pop(instance_id, None)
            self._specs.pop(instance_id, None)
            self._owned_sessions.discard(instance_id)

    async def abort(self, instance_id, reason="cancelled"):
        try:
            if self._pool is not None and instance_id in self._owned_sessions:
                await self._pool.abort_session(instance_id, reason)
        finally:
            self._contexts.pop(instance_id, None)
            self._specs.pop(instance_id, None)
            self._owned_sessions.discard(instance_id)
