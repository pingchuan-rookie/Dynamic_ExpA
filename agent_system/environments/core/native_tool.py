"""Shared runtime-session transport with fail-closed pool lifecycle."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

_POOLS = {}


class NativeToolLocalEnvTool(BaseTool):
    runtime_session = True
    protocol = "native_tools"
    reward_mode = "last"
    environment = "native_tools"
    tool_name = "native_session"
    pool_class = None

    def __init__(self, config, tool_schema=None):
        schema = tool_schema or OpenAIFunctionToolSchema.model_validate({
            "type": "function", "function": {"name": self.tool_name, "description": "Environment session transport",
            "parameters": {"type": "object", "properties": {}, "required": []}}})
        super().__init__(config, schema)
        self._pool = None
        self._contexts = {}

    @property
    def pool(self):
        if self._pool is None:
            signature = hashlib.sha256(json.dumps(self.config, sort_keys=True, allow_nan=False).encode()).hexdigest()
            key = (asyncio.get_running_loop(), self.pool_class, signature)
            if key not in _POOLS:
                _POOLS[key] = self.pool_class(self.config)
            self._pool = _POOLS[key]
        return self._pool

    @staticmethod
    def _context(response, previous=None):
        context = deepcopy(previous or {})
        context.update(deepcopy(response))
        result = response.get("result") or response.get("episode_result")
        if result is not None:
            context["result"] = deepcopy(result)
            context["episode_result"] = deepcopy(result)
        return context

    async def bootstrap(self, instance_id=None, *, create_kwargs=None, **kwargs):
        instance_id = instance_id or str(uuid4())
        if instance_id in self._contexts:
            raise ValueError("Duplicate environment session instance")
        params = dict(create_kwargs or {})
        params.update(kwargs)
        spec = deepcopy(params.get("create_payload", params))
        # The pool cleans up failed/cancelled creates. A rejected duplicate belongs
        # to another caller, so this tool must not abort that caller's lease.
        response = await self.pool.create_session(instance_id, spec)
        try:
            self._contexts[instance_id] = self._context(response)
        except BaseException:
            await self.pool.abort_session(instance_id, "reset_failed")
            raise
        return instance_id, self.get_session_context(instance_id)

    async def create(self, instance_id=None, **kwargs):
        instance_id, _ = await self.bootstrap(instance_id, **kwargs)
        return instance_id, ToolResponse()

    def get_session_context(self, instance_id):
        return deepcopy(self._contexts[instance_id])

    def get_observation(self, instance_id):
        return self._contexts[instance_id]["observation"]

    async def step(self, instance_id, action):
        if self._contexts[instance_id].get("done"):
            raise RuntimeError("Cannot step a terminal environment session")
        try:
            json.dumps(action, allow_nan=False)
            response = await self.pool.step_session(instance_id, action)
            self._contexts[instance_id] = self._context(response, self._contexts[instance_id])
        except BaseException:
            await self.abort(instance_id, "step_failed")
            raise
        return self.get_session_context(instance_id)

    async def execute(self, instance_id, parameters, **kwargs):
        context = await self.step(instance_id, parameters)
        return ToolResponse(), 0.0, {"done": bool(context.get("done")),
            "episode_result": context.get("episode_result"), "executed": context.get("executed", False),
            "format_error": context.get("format_error", False)}

    async def finalize(self, instance_id, reason="rollout_terminated"):
        context = self._contexts[instance_id]
        if context.get("result") is None:
            try:
                response = await self.pool.finalize_session(instance_id, reason)
                context = self._context(response, context)
                self._contexts[instance_id] = context
            except BaseException:
                await self.abort(instance_id, "finalize_failed")
                raise
        result = context.get("result")
        if not result or not result.get("metric_valid"):
            await self.abort(instance_id, "invalid_terminal_result")
            raise RuntimeError("Environment returned no valid terminal result")
        return deepcopy(result)

    async def calc_reward(self, instance_id, **kwargs):
        return (self._contexts[instance_id].get("result") or {}).get("official_reward")

    async def release(self, instance_id, **kwargs):
        try:
            if self._pool is not None and instance_id in self._contexts:
                await self._pool.close_session(instance_id)
        finally:
            self._contexts.pop(instance_id, None)

    async def abort(self, instance_id, reason="cancelled"):
        try:
            if self._pool is not None and instance_id in self._contexts:
                await self._pool.abort_session(instance_id, reason)
        finally:
            self._contexts.pop(instance_id, None)
