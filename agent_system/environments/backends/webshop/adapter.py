"""Native WebShop commands and continuous official outcome rewards."""
from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any

from verl.tools.schemas import OpenAIFunctionToolSchema
from agent_system.environments.core.adapter import BaseEnvAdapter


class WebShopAdapter(BaseEnvAdapter):
    def get_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": self.config.get("tool_name", "webshop_action"),
                "description": "Execute a native WebShop search[query] or click[target] command.",
                "parameters": {
                    "type": "object",
                    "properties": {"action": {"type": "string", "description": "search[query] or click[target]"}},
                    "required": ["action"],
                },
            },
        })

    def build_action(self, parameters: dict[str, Any]) -> str:
        value = parameters.get("raw_action", parameters.get("action"))
        if isinstance(value, Mapping):
            return self.build_action(dict(value))
        if not isinstance(value, str):
            raise ValueError("WebShop requires a native action string")
        # Strip envelope whitespace only, never change query case or internal spacing.
        action = value.strip()
        match = re.fullmatch(r"(search|click)\[(.*)\]", action, re.DOTALL)
        if match is None or not match.group(2):
            raise ValueError("Expected search[query] or click[target] with a nonempty argument")
        return action

    def extract_reward(self, response: dict[str, Any]) -> float:
        reward = float(response.get("reward", 0.0))
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError("Official WebShop reward must be finite and in [0, 1]")
        return reward

    def extract_available_actions(self, response: dict[str, Any]) -> list[str]:
        actions = response.get("available_actions", {})
        if isinstance(actions, dict):
            # Only public page elements. Search is an open argument, never a
            # hidden-goal-derived shortlist of suggested queries.
            result = [f"click[{target}]" for target in actions.get("clickables", []) if target != "search"]
            if actions.get("has_search_bar"):
                result.insert(0, "search[query]")
            return result
        if isinstance(actions, list) and all(isinstance(action, str) for action in actions):
            return list(actions)
        raise ValueError("WebShop available_actions must contain public clickables")

    def format_tool_text(self, *, observation: str, available_actions: list[Any]) -> str:
        # Official native ReAct observations, not an invented instruction/action wrapper.
        return observation
