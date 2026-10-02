from __future__ import annotations

from typing import Any

from verl.tools.schemas import OpenAIFunctionToolSchema

from agent_system.environments.core.adapter import BaseEnvAdapter


class GenericAgentGymAdapter(BaseEnvAdapter):
    """Generic text-action adapter for AgentGym-compatible envs."""

    def get_tool_schema(self) -> OpenAIFunctionToolSchema:
        tool_name = self.config.get("tool_name") or f"{self.config.get('env_type', 'env')}_step"

        schema_dict = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": (
                    "Execute one text action in an AgentGym-style environment. "
                    "Use action for the exact command accepted by the environment."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Exact environment action, for example 'go to kitchen' or 'click[Buy Now]'.",
                        },
                        "raw_action": {
                            "type": "string",
                            "description": "Alias for action. If provided, it overrides action.",
                        },
                        "command": {
                            "type": "string",
                            "description": "Alias for action.",
                        },
                    },
                    "required": ["action"],
                },
            },
        }

        return OpenAIFunctionToolSchema.model_validate(schema_dict)

    def build_action(self, parameters: dict[str, Any]) -> str:
        value = parameters.get("raw_action", parameters.get("action", parameters.get("command")))

        if value is None:
            raise ValueError("One of 'action', 'raw_action', or 'command' must be provided")

        action = str(value).strip()

        if not action:
            raise ValueError("Action is empty")

        return action