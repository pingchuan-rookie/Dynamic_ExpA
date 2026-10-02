from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verl.tools.schemas import OpenAIFunctionToolSchema
from agent_system.environments.core.adapter import BaseEnvAdapter


class AlfworldAdapter(BaseEnvAdapter):
    """ALFWorld adapter.

    Default strategy:
    - expose one structured function: alfworld_step
    - convert structured params into ALFWorld text actions
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        self.action_templates: dict[str, str] = self.config.get(
            "action_templates",
            {
                "goto": "go to {receptacle}",
                "take": "take {object} from {receptacle}",
                "put": "put {object} {preposition} {receptacle}",
                "open": "open {receptacle}",
                "close": "close {receptacle}",
                "look": "look",
                "inventory": "inventory",
                "examine": "examine {object}",
                "clean": "clean {object} with {receptacle}",
                "cool": "cool {object} with {receptacle}",
                "heat": "heat {object} with {receptacle}",
                "slice": "slice {object} with {receptacle}",
                "toggle": "toggle {object}",
            },
        )

        self.default_preposition = self.config.get("default_put_preposition", "in")
        self.allow_raw_action_fallback = bool(self.config.get("allow_raw_action_fallback", True))

    def get_tool_schema(self) -> OpenAIFunctionToolSchema:
        # Single-field schema: expose only action = the full ALFWorld command, copied verbatim from
        # Available actions. The former multi-field object/receptacle/preposition/raw_action/command
        # schema induced redundant filling (e.g. stuffing the whole command into action *and* also
        # filling object/receptacle); with one field the model only needs
        # {"action": "take mug 2 from coffeemachine 1"}.
        # Note: build_action still accepts structured params and raw_action (used by the Dyad path);
        # this schema is only what FC GRPO injects for the model to read.
        schema_dict = {
            "type": "function",
            "function": {
                "name": self.config.get("tool_name", "alfworld_step"),
                "description": (
                    "Execute one action in ALFWorld. Put the exact ALFWorld command "
                    "(copied verbatim from the current Available actions) in the `action` field."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": (
                                "The exact ALFWorld command to execute, copied verbatim from the current "
                                "Available actions, e.g. 'go to cabinet 1', 'take mug 2 from coffeemachine 1', "
                                "'put mug 2 in/on cabinet 1', 'open cabinet 1', 'look', 'inventory'."
                            ),
                        },
                    },
                    "required": ["action"],
                },
            },
        }
        return OpenAIFunctionToolSchema.model_validate(schema_dict)

    def _normalize_key(self, key: Any) -> str:
        return str(key).strip().strip("#").rstrip(":").strip()

    def _normalize_value(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip().strip("#").strip()
        return str(value)

    def _build_structured_action(self, parameters: Mapping[str, Any]) -> str:
        normalized: dict[str, str] = {}
        for k, v in parameters.items():
            if v is None:
                continue
            normalized[self._normalize_key(k)] = self._normalize_value(v)

        action = normalized.get("action")
        if action is None:
            raise ValueError("Either 'action' or 'raw_action' must be provided")

        action = action.strip()
        if action not in self.action_templates:
            if self.allow_raw_action_fallback and action:
                return action
            raise ValueError(
                f"Unsupported action={action!r}. Supported actions: {sorted(self.action_templates.keys())}"
            )

        if "target" in normalized:
            normalized.setdefault("receptacle", normalized["target"])
            normalized.setdefault("object", normalized["target"])
        if "item" in normalized:
            normalized.setdefault("object", normalized["item"])
        if "location" in normalized:
            normalized.setdefault("receptacle", normalized["location"])

        if action == "put" and "preposition" not in normalized:
            normalized["preposition"] = self.default_preposition

        template = self.action_templates[action]
        try:
            built = template.format(**normalized).strip()
        except KeyError as e:
            missing = e.args[0]
            raise ValueError(
                f"Action {action!r} requires parameter {missing!r}. Template={template!r}"
            ) from e

        if not built:
            raise ValueError(f"Built empty action from parameters={parameters}")
        return built

    def build_action(self, parameters: dict[str, Any]) -> str:
        raw_action = parameters.get("raw_action", parameters.get("command"))
        if isinstance(raw_action, Mapping):
            structured = {
                k: v
                for k, v in parameters.items()
                if k not in {"raw_action", "command"}
            }
            structured.update(raw_action)
            # The caller (dyad agent loop) wraps the whole tool_args once more as {"raw_action": tool_args}.
            # If the inner value is itself a *full command string* (Dyad raw-command with env_serialize=alfworld_raw),
            # it is still a raw passthrough after flattening and must not be fed to action_templates as
            # structured parameters.
            inner = structured.get("raw_action", structured.get("command"))
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
            return self._build_structured_action(structured)

        if raw_action is not None:
            raw_action = str(raw_action).strip()
            if not raw_action:
                raise ValueError("raw_action is empty")
            return raw_action

        return self._build_structured_action(parameters)
