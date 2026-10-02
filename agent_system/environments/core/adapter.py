from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from verl.tools.schemas import OpenAIFunctionToolSchema


def _first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


class BaseEnvAdapter(ABC):
    """Adapter layer for env-specific action/schema/format logic."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def get_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return env-specific function-calling schema."""

    @abstractmethod
    def build_action(self, parameters: dict[str, Any]) -> Any:
        """Convert structured parameters into the actual env action."""

    def extract_reward(self, response: dict[str, Any]) -> float:
        value = _first_present(response, ["reward", "score", "step_reward"])
        if value is None and isinstance(response.get("info"), dict):
            value = _first_present(response["info"], ["reward", "score", "step_reward"])
        if value is None:
            value = 0.0
        try:
            return float(value)
        except Exception:
            return 0.0

    def extract_done(self, response: dict[str, Any]) -> bool:
        value = _first_present(response, ["done", "terminated", "truncated", "success", "completed"])
        if value is None and isinstance(response.get("info"), dict):
            value = _first_present(response["info"], ["done", "terminated", "truncated", "success", "completed"])
        return bool(value)

    def extract_observation(self, response: dict[str, Any]) -> str:
        obs = _first_present(response, ["observation", "obs", "state", "text", "message"])
        if obs is None and isinstance(response.get("data"), dict):
            obs = _first_present(response["data"], ["observation", "obs", "state", "text", "message"])
        if obs is None:
            obs = ""
        return "" if obs is None else str(obs)

    def extract_available_actions(self, response: dict[str, Any]) -> list[Any]:
        actions = _first_present(
            response,
            ["available_actions", "valid_actions", "admissible_actions", "actions", "action_space"],
        )
        if actions is None and isinstance(response.get("info"), dict):
            actions = _first_present(
                response["info"],
                ["available_actions", "valid_actions", "admissible_actions", "actions", "action_space"],
            )
        if isinstance(actions, str):
            return [actions]
        return actions or []

    def extract_info(self, response: dict[str, Any]) -> dict[str, Any]:
        info = response.get("info", {})
        if isinstance(info, dict):
            return info
        return {"info": info} if info is not None else {}

    def format_tool_text(
        self,
        *,
        observation: str,
        available_actions: list[Any],
    ) -> str:
        actions_text = "\n".join(f"- {a}" for a in available_actions) if available_actions else "- <none>"
        return (
            f"Observation:\n{observation}\n"
            f"Available actions:\n{actions_text}\n"
        )
