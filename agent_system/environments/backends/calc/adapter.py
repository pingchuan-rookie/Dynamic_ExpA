from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verl.tools.schemas import OpenAIFunctionToolSchema

from agent_system.environments.core.adapter import BaseEnvAdapter


class CalcAdapter(BaseEnvAdapter):
    """GSM8K calculator env adapter.

    Two kinds of actions (mirroring the calc/answer actions of the Dyad action head):
      - calc   : arithmetic expression (may end with '='), e.g. "1+2-3=" / "48/2".
      - answer : final answer, e.g. "#### 72" / "answer: 72".

    All three model-side text protocols are normalized into a single action string before
    reaching this adapter:
      - react : ReactToolParser extracts <action>cmd</action> -> {"raw_action": cmd}
      - fc     : ReactFcToolParser extracts the caculate/answer key of <action>{json}</action> -> {"raw_action": ...}
      - dyad   : the dyad tool parses the expression/answer produced by the action head -> {"raw_action": ...}

    build_action just takes the raw_action/action/command text (stripping trailing connectors is
    the server's job).
    """

    def get_tool_schema(self) -> OpenAIFunctionToolSchema:
        tool_name = self.config.get("tool_name") or "calculator"
        schema_dict = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": (
                    "A calculator for solving math problems step by step. "
                    "Pass an arithmetic expression ending with '=' to compute it (e.g. '48/2='), "
                    "or submit the final answer as '#### <number>'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Arithmetic expression to evaluate, e.g. '48/2='.",
                        },
                        "caculate": {
                            "type": "string",
                            "description": "Alias for expression (matches the <action> JSON key).",
                        },
                        "answer": {
                            "type": "string",
                            "description": "Final answer to submit, e.g. '72'.",
                        },
                        "raw_action": {
                            "type": "string",
                            "description": "Raw action text; overrides the structured fields if present.",
                        },
                        "action": {
                            "type": "string",
                            "description": "Alias for raw_action.",
                        },
                    },
                    "required": [],
                },
            },
        }
        return OpenAIFunctionToolSchema.model_validate(schema_dict)

    # ---- Dyad (codegym path) structured action -> calculator command text ----
    # The Dyad action head picks an action name (calculate/answer) plus open-vocabulary parameter
    # values; the codegym branch of tool_parser replays them into {"name","parameters"} via
    # decisions_to_codegym_actions, so after _call_tool the parameters reaching build_action look like
    # {"raw_action": {"codegym_action": {"name":..,"parameters":{..}}, "action": <name>}}.
    _CALC_NAMES = ("calculate", "caculate", "calc", "compute")
    _ANSWER_NAMES = ("answer", "the answer is", "final answer", "submit")

    @staticmethod
    def _extract_named_action(raw: Any) -> dict[str, Any] | None:
        """Pull {"name","parameters"} out of an Dyad/codegym structure; return None if unstructured."""
        if not isinstance(raw, dict):
            return None
        act = raw.get("codegym_action")
        if isinstance(act, dict) and act.get("name"):
            return act
        if raw.get("name") and isinstance(raw.get("parameters"), dict):
            return {"name": raw["name"], "parameters": raw["parameters"]}
        return None

    def _named_action_to_command(self, act: dict[str, Any]) -> str:
        """{name, parameters} -> calculator command text (calculate -> expression; answer -> 'answer <number>')."""
        name = str(act.get("name", "")).strip().lower()
        params = act.get("parameters") or {}
        if not isinstance(params, dict):
            params = {}

        def _first_value(*keys: str) -> str:
            for k in keys:
                if k in params and params[k] is not None:
                    return str(params[k]).strip()
            for v in params.values():
                if v is not None:
                    return str(v).strip()
            return ""

        if name.startswith(self._ANSWER_NAMES):
            val = _first_value("value", "answer", "number")
            return f"answer {val}".strip()
        # Default is calculate: hand over the raw expression text (CalcSession strips the prefix and
        # the trailing '=' before safe_eval).
        expr = _first_value("expression", "caculate", "calculate", "expr")
        return expr

    def build_action(self, parameters: dict[str, Any]) -> str:
        # Precedence: raw_action > action > command > expression/caculate > answer.
        value = parameters.get("raw_action")

        # Dyad (codegym path): raw_action is a structured {codegym_action/name,parameters} dict -> command text.
        named = self._extract_named_action(value) or self._extract_named_action(parameters)
        if named is not None:
            action = self._named_action_to_command(named).strip()
            if not action:
                raise ValueError(f"Cannot build calculator command from Dyad action: {named!r}")
            return action

        # Dyad raw-command (env_serialize=calc_raw): the agent loop wraps the
        # whole tool_args once more as {"raw_action": tool_args}, so raw_action is a Mapping whose
        # own "raw_action" holds the finished command string. AlfworldAdapter.build_action already
        # unwraps this; calc did not, and only the codegym-structured shape above was handled.
        #
        # The failure was silent and expensive to trace: str(dict) became the "expression", the env
        # classified "{'raw_action': 'answer 6', ...}" as a calculate (it does not start with an
        # answer prefix), safe_eval rejected it as "Expression contains disallowed characters", every
        # reward came back 0, and GRPO's group advantages were therefore all 0 too -- so pg_loss and
        # grad_norm were 0 as well, which reads exactly like a broken training path rather than a
        # mangled action string.
        if isinstance(value, Mapping):
            inner = value.get("raw_action", value.get("action", value.get("command")))
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
            raise ValueError(f"Cannot build calculator command from nested raw_action: {value!r}")

        if value is None:
            value = parameters.get("action")
        if value is None:
            value = parameters.get("command")
        if value is None:
            expr = parameters.get("expression") or parameters.get("caculate") or parameters.get("calculate")
            if expr is not None:
                value = str(expr)
        if value is None and parameters.get("answer") is not None:
            # answer is a named action: write it as "answer <value>" so the env takes its answer-scoring branch.
            value = f"answer {parameters.get('answer')}"
        if value is None:
            raise ValueError(
                "Provide one of 'raw_action'/'action'/'expression'/'caculate'/'answer' for the calculator"
            )
        action = str(value).strip()
        if not action:
            raise ValueError("Action is empty")
        return action

    def format_tool_text(self, *, observation: str, available_actions: list[Any]) -> str:
        # The calculator has no admissible action set; only echo the observation (computation result / answer feedback).
        return f"Observation: {observation}"
