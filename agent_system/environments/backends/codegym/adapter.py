from __future__ import annotations

import json
from typing import Any

from verl.tools.schemas import OpenAIFunctionToolSchema
from agent_system.environments.core.adapter import BaseEnvAdapter


class CodeGymAdapter(BaseEnvAdapter):
    """CodeGym env adapter.

    CodeGym's ``env.step(action)`` takes a JSON string
    ``{"name": <Action>, "parameters": {<p>: <v>, ...}}`` (see the step of each ``*Env.py``:
    ``json.loads(action) -> call_dict["name"]/["parameters"]``).

    On the Dyad path the codegym branch of tool_parser has already replayed the decision sequence
    into that structure via ``decisions_to_codegym_action`` and put it under the dedicated key
    ``codegym_action`` of FunctionCall.arguments. This adapter only reassembles it
    *deterministically* back into the JSON string the env expects.

    After the agent loop's ``_call_tool``, the parameters ``build_action`` actually receives look like:
        {"raw_action": {"codegym_action": {"name":..., "parameters":{...}}, "action": <name>}}
    (``_call_tool`` injects ``action=<FunctionCall.name>`` on the single-tool fallback).
    Hence the resolution order: codegym_action key > already {name,parameters} > rebuild from
    action plus the remaining keys.
    """

    def get_tool_schema(self) -> OpenAIFunctionToolSchema:
        tool_name = self.config.get("tool_name", "codegym_step")
        schema_dict = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": (
                    "Execute one structured action in a CodeGym environment. "
                    "Provide the action via 'codegym_action' as {name, parameters}, "
                    "or 'name' plus 'parameters'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "codegym_action": {
                            "type": "object",
                            "description": "Full action object {name, parameters}.",
                        },
                        "name": {
                            "type": "string",
                            "description": "Action name when codegym_action is absent.",
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Action parameters when codegym_action is absent.",
                        },
                    },
                    "required": [],
                },
            },
        }
        return OpenAIFunctionToolSchema.model_validate(schema_dict)

    def _extract_action(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            # B2 text protocol: action may already be a JSON string of {"name","parameters"}.
            if isinstance(raw, str):
                parsed = self._try_parse_action_str(raw)
                if parsed is not None:
                    return parsed
            raise ValueError(f"CodeGym action parameters must be a dict, got {type(raw)!r}")

        # 1) Dedicated key: the full action written by the codegym (Dyad) branch of tool_parser.
        act = raw.get("codegym_action")
        if isinstance(act, dict):
            return act

        # 2) Already a {name, parameters} structure.
        if "name" in raw and isinstance(raw.get("parameters"), dict):
            return {"name": raw["name"], "parameters": raw["parameters"]}

        # 3) B2 text protocol: action/raw_action/command is a JSON string of {"name","parameters"}
        #    (passed through by CodeGymToolParser, see agent_system/parsers/codegym.py).
        for key in ("action", "raw_action", "command"):
            v = raw.get(key)
            if isinstance(v, str):
                parsed = self._try_parse_action_str(v)
                if parsed is not None:
                    return parsed
            elif isinstance(v, dict) and "name" in v:
                return {"name": v["name"], "parameters": v.get("parameters", {})}

        # 4) Fallback: rebuild the parameters from action (the action name injected by _call_tool) plus the remaining keys.
        name = raw.get("action") or raw.get("name")
        if name is None:
            raise ValueError(f"Cannot resolve CodeGym action name from: {raw!r}")
        params = {
            k: v
            for k, v in raw.items()
            if k not in {"action", "name", "codegym_action", "parameters"}
        }
        if isinstance(raw.get("parameters"), dict):
            params.update(raw["parameters"])
        return {"name": name, "parameters": params}

    @staticmethod
    def _try_parse_action_str(s: str) -> dict[str, Any] | None:
        """Try to parse a string into {"name","parameters"}; return None on failure."""
        s = s.strip()
        if not (s.startswith("{") and "name" in s):
            return None
        try:
            obj = json.loads(s)
        except Exception:
            return None
        if isinstance(obj, dict) and "name" in obj:
            return {"name": obj["name"], "parameters": obj.get("parameters", {})}
        return None


    def build_action(self, parameters: dict[str, Any]) -> str:
        raw = parameters.get("raw_action", parameters)
        act = self._extract_action(raw)

        name = act.get("name")
        if not name:
            raise ValueError(f"CodeGym action missing 'name': {act!r}")
        params = act.get("parameters", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError(f"CodeGym action 'parameters' must be a dict, got {type(params)!r}")

        # The exact structure env.step expects; ensure_ascii=False preserves non-ASCII literals.
        return json.dumps({"name": str(name), "parameters": params}, ensure_ascii=False)

    def format_tool_text(
        self,
        *,
        observation: str,
        available_actions: list[Any],
    ) -> str:
        """Build the tool-result text sent back to the model.

        CodeGym is a function-calling env: its action space is the fixed function documentation
        embedded in the system prompt, and there is *no* per-step refreshed admissible-action list
        like ALFWorld has. The bridge (/create, /step, /reset) always returns
        ``available_actions=[]``, so the ``Available actions:\n- <none>`` that base's
        ``format_tool_text`` appends is meaningless noise for CodeGym (pure ALFWorld mimicry) and is
        dropped here. Only env.step's raw observation is returned (the ``Observation:`` prefix keeps
        the system prompt's "wait for the Observation: message" convention).
        """
        return f"Observation:\n{observation}\n"

