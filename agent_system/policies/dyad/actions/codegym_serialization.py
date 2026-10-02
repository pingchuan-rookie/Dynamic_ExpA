"""Decode flat decisions into CodeGym action invocations."""
from __future__ import annotations

import ast
from typing import Any
from agent_system.policies.dyad.actions.flat_router import FlatActionRouter


def _decode_open_value(tokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    if tokenizer is None:
        raise ValueError("decisions_to_codegym_action needs a tokenizer to decode open param values")
    return tokenizer.decode(list(token_ids), skip_special_tokens=False)



def _parse_value_by_type(ptype: str, inner_text: str) -> Any:
    """Parse the free text collected in ARGUMENT_VALUE into a Python value according to the param's type.

    The model writes the whole value string itself (it may contain quotes/brackets), converted by the
    declared type:
      str  : strip a matched pair of outer quotes (if any), keep the rest verbatim.
      int/float/bool : direct numeric conversion.
      list/dict/unknown etc. : ast.literal_eval.
    Any parse failure **falls back to the raw text** -- better to let the env side raise than to
    swallow the value here.
    """
    t = (ptype or "").strip().lower()
    s = inner_text.strip()
    if t in ("str", "string", "char", "text"):
        return _strip_quotes(inner_text)
    if t in ("int", "integer"):
        try:
            return int(s)
        except Exception:
            pass
    elif t in ("bool", "boolean"):
        low = s.lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
    elif t in ("float", "double", "number"):
        try:
            return float(s)
        except Exception:
            pass
    # list/dict/tuple/json/unknown, or a failed numeric conversion above -> try a literal, then fall back to the raw text
    try:
        return ast.literal_eval(s)
    except Exception:
        return inner_text



def _strip_quotes(text: str) -> str:
    s = text.strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        return s[1:-1]
    return text



def decisions_to_codegym_actions(action_config, raw_token_ids, tokenizer=None) -> list[dict]:
    """Replay a decision sequence into **one or more** CodeGym actions (in order of appearance).

    Each action looks like {"name": <Action>, "parameters": {<pname>: <value>, ...}} and can be fed to
    CodeGym env.step() straight after json.dumps.

    Replayed with the **same FlatActionRouter** as build_flat_policy_trace, aligned decision by decision:
      - "force" decisions (forced tokens): advance the router, consume no raw_token_ids, produce no semantics.
      - "sample" decisions: consume one raw token and route it to an action name/param value by phase.
    NONE-phase decisions (reasoning text + the enter trigger) are ignored.

    When raw_token_ids is truncated (generation ended midway), the actions collected so far are still
    returned, without error.
    """
    value_end_set = set(int(x) for x in action_config.get("value_end_token_ids", []))
    value_end_set |= set(int(x) for x in action_config.get("exit_value_end_ids", []))
    surface_form = action_config["surface_form"]

    router = FlatActionRouter(action_config)
    actions: list[dict] = []
    cur: dict | None = None
    open_buf: list[int] = []

    raw_idx = 0
    n = len(raw_token_ids)
    guard = 0
    while raw_idx < n:
        guard += 1
        if guard > 10 * (n + 8):
            raise RuntimeError("decisions_to_codegym_actions replay loop guard")
        d = router.decision()
        if d["kind"] == "force":
            router.advance(d["forced_token"])
            continue

        # SAMPLE: consume one real decision. Note the router's current state must be read first
        # (advance mutates it).
        raw = int(raw_token_ids[raw_idx]); raw_idx += 1
        phase = d["phase"]

        if phase == "ACTION_NAME":
            if cur is not None:
                actions.append(cur)
            cur = {"name": action_config["id_to_str"][raw], "parameters": {}}
            open_buf = []
        elif phase == "ARGUMENT_VALUE":
            if raw in value_end_set:
                pr = surface_form[router.action_name]["params"][router.argument_index]
                inner = _decode_open_value(tokenizer, open_buf)
                cur["parameters"][pr["name"]] = _parse_value_by_type(pr.get("type", ""), inner)
                open_buf = []
            else:
                open_buf.append(raw)
        # phase == "NONE": reasoning / the enter trigger, no semantics, skipped.

        router.advance(raw)

    if cur is not None:
        actions.append(cur)
    return actions



def decisions_to_codegym_action(action_config, raw_token_ids, tokenizer=None) -> dict:
    """Single-action version (one tool call per turn): returns the first complete action.

    tool_parser.dyad_extract_tool_calls uses it to turn a decision sequence straight into a
    FunctionCall's {name, parameters}, then json.dumps(parameters) is fed to the CodeGym env.
    """
    actions = decisions_to_codegym_actions(action_config, raw_token_ids, tokenizer)
    if not actions:
        raise ValueError("no parsable CodeGym action in the decision sequence")
    return actions[0]

