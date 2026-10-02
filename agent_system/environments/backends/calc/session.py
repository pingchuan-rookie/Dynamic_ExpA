# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Pure logic core of the GSM8K calculator env (no FastAPI / no Ray / no verl dependency) -- single source of truth.

**Deliberately placed in the `agent_system/environments` package (not under `verl`)**: a Ray actor
process only needs to import this module, which never triggers `verl/__init__.py`->torch and thus
avoids the worker registration storm (see this package's __init__).
On the driver side, `from agent_system.environments.backends.calc.session import ...` references the very same logic.
Used by the Ray actor in agent_system/environments/backends/calc/worker.py (in-process Ray env pool, moved onto Ray,
same as ALFWorld).

The env is a multi-turn calculator where one question = one session. Two **named actions** (aligned
with the calc/answer actions of the Dyad action head):
  - calculate : evaluate an arithmetic expression (a trailing '=' is allowed), e.g. "48/2=" /
                "calculate 48/2" / bare "48/2". The result becomes the observation,
                reward=0, done=False.
  - answer    : submit the final answer, e.g. "answer 72" / "answer: 72" (the legacy "#### 72" is
                still accepted). The number is extracted and compared with ground_truth,
                reward=1.0/0.0, done=True.

step() returns the same dict contract as the ALFWorld env pool:
  {observation, reward, available_actions, done, step_count, action_name, max_turns_reached}
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Any

from agent_system.environments.core.session import BaseEnvSession

# ---------------------------------------------------------------------------
# Safe arithmetic evaluation (ast; arithmetic operators only, no names/calls/attributes)
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_POW_EXPONENT = 1000


class CalcError(Exception):
    """An illegal or non-evaluable expression."""


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalcError(f"Unsupported constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise CalcError(f"Operator not allowed: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Pow and isinstance(right, (int, float)) and abs(right) > _MAX_POW_EXPONENT:
            raise CalcError("Exponent too large")
        return _ALLOWED_BINOPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise CalcError(f"Unary operator not allowed: {op_type.__name__}")
        return _ALLOWED_UNARYOPS[op_type](_eval_node(node.operand))
    raise CalcError(f"Unsupported expression node: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """Safely evaluate an arithmetic expression string; raises CalcError when illegal or out of range."""
    expr = expression.strip()
    if not expr:
        raise CalcError("Empty expression")
    if not re.fullmatch(r"[0-9.\s+\-*/%()]+", expr):
        raise CalcError("Expression contains disallowed characters")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"Syntax error: {exc.msg}") from exc
    try:
        return _eval_node(tree)
    except CalcError:
        raise
    except ZeroDivisionError as exc:
        raise CalcError("Division by zero") from exc
    except (OverflowError, ValueError, ArithmeticError) as exc:
        raise CalcError(f"Math error: {exc}") from exc


def format_number(value: float) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Action classification (calculate / answer) and answer comparison
# ---------------------------------------------------------------------------

# Named-action prefixes for answer (case-insensitive); the legacy #### marker is still accepted.
_ANSWER_PREFIXES = ("answer", "the answer is", "final answer", "####")
_CALC_PREFIXES = ("calculate", "caculate", "calc", "compute")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Action-name constants (referenced uniformly by the projector / Dyad action head / dataset prompts).
ACTION_CALCULATE = "calculate"
ACTION_ANSWER = "answer"


def classify_action(action: str) -> str:
    """Return the action name: ACTION_ANSWER or ACTION_CALCULATE."""
    a = action.strip().lower()
    if a.startswith(_ANSWER_PREFIXES) or a.startswith("answer:"):
        return ACTION_ANSWER
    return ACTION_CALCULATE


def strip_action_prefix(action: str, prefixes: tuple[str, ...]) -> str:
    """Return the text left after stripping an action-name prefix (e.g. 'calculate ' / 'answer:')."""
    a = action.strip()
    low = a.lower()
    for p in prefixes:
        if low.startswith(p):
            rest = a[len(p):]
            return rest.lstrip(": \t")
    return a


def extract_number(text: str) -> str | None:
    """Extract the last number in the text (thousands separators removed)."""
    cleaned = str(text).replace(",", "")
    matches = _NUM_RE.findall(cleaned)
    return matches[-1] if matches else None


def answers_match(submitted: str | None, ground_truth: str | None) -> bool:
    if submitted is None or ground_truth is None:
        return False
    gt_num = extract_number(str(ground_truth))
    sub_num = extract_number(str(submitted))
    if gt_num is None or sub_num is None:
        return False
    try:
        return abs(float(sub_num) - float(gt_num)) < 1e-6
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Single-question calculator session (plain Python; reusable from a Ray actor / thread / HTTP server)
# ---------------------------------------------------------------------------

_READY_OBS = (
    "Calculator ready. Use the 'calculate' action with an arithmetic expression "
    "(e.g. 'calculate 48/2='), or the 'answer' action to submit your final answer (e.g. 'answer 72')."
)


class CalcSession(BaseEnvSession):
    """A multi-turn calculator session for one GSM8K question. Used serially within a thread."""

    def __init__(self, max_turns: int = 20) -> None:
        self.ground_truth: str | None = None
        self.done = False
        self.step_count = 0
        self.max_turns = int(max_turns)
        self.last_observation = ""

    def reset(self, ground_truth: str | None = None, max_turns: int | None = None, **_: Any) -> dict:
        self.ground_truth = ground_truth
        self.done = False
        self.step_count = 0
        if max_turns is not None:
            self.max_turns = int(max_turns)
        self.last_observation = _READY_OBS
        return {
            "observation": self.last_observation,
            "reward": 0.0,
            "available_actions": [ACTION_CALCULATE, ACTION_ANSWER],
            "done": False,
            "step_count": 0,
        }

    def step(self, action: str) -> dict:
        if self.done:
            return {
                "observation": self.last_observation,
                "reward": 0.0,
                "available_actions": [],
                "done": True,
                "step_count": self.step_count,
                "already_done": True,
            }

        self.step_count += 1
        action = (action or "").strip()
        action_name = classify_action(action)
        reward = 0.0
        done = False
        won = False

        if action_name == ACTION_ANSWER:
            value_text = strip_action_prefix(action, _ANSWER_PREFIXES + ("answer:",))
            submitted = extract_number(value_text if value_text else action)
            won = answers_match(submitted, self.ground_truth)
            reward = 1.0 if won else 0.0
            done = True
            obs = f"Final answer received: {submitted}. {'Correct.' if won else 'Incorrect.'}"
        else:
            expr = strip_action_prefix(action, _CALC_PREFIXES).rstrip("=").strip()
            try:
                value = safe_eval(expr)
                obs = f"{expr} = {format_number(value)}"
            except CalcError as exc:
                obs = f"Error: {exc}. Provide a valid arithmetic expression like '48/2='."

        max_turns_reached = self.step_count >= self.max_turns
        if max_turns_reached:
            done = True

        self.done = done
        self.last_observation = obs
        return {
            "observation": obs,
            "reward": float(reward),
            "available_actions": [] if done else [ACTION_CALCULATE, ACTION_ANSWER],
            "done": bool(done),
            "step_count": self.step_count,
            "action_name": action_name,
            "won": won,
            "max_turns_reached": bool(max_turns_reached),
        }

    def close(self) -> dict:
        self.done = True
        return {"closed": True}

    def health_check(self) -> dict:
        try:
            self.reset(ground_truth="3")
            s = self.step("1+2=")
            ok = s["observation"].endswith("= 3")
            a = self.step("answer 3")
            ok = ok and a["reward"] == 1.0 and a["done"]
            self.close()
            return {"ok": bool(ok)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": repr(exc)}
