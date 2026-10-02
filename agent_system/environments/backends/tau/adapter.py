"""Transport-only tau adapter; official sessions own parsing and scoring."""
from __future__ import annotations

from copy import deepcopy
import json

from verl.tools.schemas import OpenAIFunctionToolSchema
from agent_system.environments.core.adapter import BaseEnvAdapter


class TauAdapter(BaseEnvAdapter):
    def get_tool_schema(self):
        # This bootstrap handle is never exposed as the policy's action schema.
        return OpenAIFunctionToolSchema.model_validate({
            "type": "function", "function": {
                "name": self.config.get("tool_name", "tau_session"),
                "description": "Runtime tau session transport",
                "parameters": {"type": "object", "properties": {
                    "raw_text": {"type": "string"}}, "required": []},
            },
        })

    def build_action(self, parameters):
        if not isinstance(parameters, dict):
            raise TypeError("Tau actions must be JSON objects")
        # No stringification, JSON repair, argument coercion or action-name guessing.
        json.dumps(parameters, allow_nan=False)
        return deepcopy(parameters)

    def extract_reward(self, response):
        result = response.get("result") or response.get("episode_result") or {}
        return result.get("official_reward")

    def format_tool_text(self, *, observation, available_actions):
        raise RuntimeError("Tau uses official agent messages, not observation wrappers")
