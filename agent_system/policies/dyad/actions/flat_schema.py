"""Compile flat action schemas shared by environment protocols."""
from __future__ import annotations

from typing import Any


def _build_trigger_ids(tokenizer, marker_str: str) -> list[int]:
    marker = marker_str.strip()
    trigger = set(tokenizer.encode(marker_str, add_special_tokens=False))
    for tid in tokenizer.get_vocab().values():
        try:
            if tokenizer.decode([int(tid)]).strip() == marker:
                trigger.add(int(tid))
        except Exception:
            continue
    return sorted(trigger)



def _build_newline_ids(tokenizer) -> list[int]:
    """Collect the token ids in the vocabulary that decode exactly to the newline '\\n' -- the **terminator** of ARGUMENT_VALUE.

    The whole param value is generated freely by the model over the plain vocabulary; hitting one of
    these '\\n' tokens marks the end of that param value.
    """
    ids = set(int(x) for x in tokenizer.encode("\n", add_special_tokens=False))
    for tid in tokenizer.get_vocab().values():
        try:
            if tokenizer.decode([int(tid)]) == "\n":
                ids.add(int(tid))
        except Exception:
            continue
    return sorted(ids)



def _build_exit_value_end_ids(tokenizer, exit_str: str) -> list[int]:
    """Collect the token ids whose **decoded text contains the exit marker's leading fragment** -- the robust terminator for open values.

    Recognizing only the exit marker's first token (e.g. '</') misses cases: BPE frequently merges the
    value's last character with '</' into one token (in Qwen2.5 the '.</' of '1.</Action>' is the
    single token 3918), and a miss means the param value never ends and the action never closes.
    They are collected here by **decoded text containing the exit marker's first 2 characters**, which
    is only valid because that fragment cannot appear inside a legal param value ('</' holds for both
    arithmetic expressions and ALFWorld object names), so it is enabled only when the schema turns on
    value_end_on_exit explicitly.
    """
    marker = (exit_str or "").strip()
    if len(marker) < 2:
        return []
    needle = marker[:2]
    ids: set[int] = set(int(x) for x in tokenizer.encode(exit_str, add_special_tokens=False)[:1])
    for tid in tokenizer.get_vocab().values():
        try:
            if needle in tokenizer.decode([int(tid)]):
                ids.add(int(tid))
        except Exception:
            continue
    return sorted(ids)



def compile_flat_schema(tokenizer, vocab_size: int, raw: dict[str, Any]) -> dict[str, Any]:
    def enc(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False)) if text else []

    actions_schema = raw["actions_schema"]
    routing = raw.get("routing", {})
    enter_str = routing.get("enter_str", "<|FunctionCallBegin|>")
    # Exit marker (e.g. <|FunctionCallEnd|>): once the action surface form includes all argument values it is force-written
    # and then eos is forced, matching ALFWorld's </Action>+eos closing semantics. Left empty, only
    # eos is forced (the old behavior).
    exit_str = routing.get("exit_str", "")
    emit_eos = bool(routing.get("emit_eos", True))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    turn_end_ids = [int(eos_token_id)] if emit_eos and eos_token_id is not None else []

    str_to_id, id_to_seq, id_to_str, id_to_init_weight = {}, {}, {}, {}
    action_ids: list[int] = []
    cur = vocab_size

    # The only extended action id: the action name. Param values switched to free generation over the
    # plain vocabulary terminated by '\n', so no closed_int / closed_bool / VALUE_END extended ids are
    # allocated any more.
    action_name_ids: dict[str, int] = {}
    for action_name, cfg in actions_schema.items():
        str_to_id[action_name] = cur
        id_to_seq[cur] = enc(cfg["surface_form"])
        id_to_str[cur] = action_name
        id_to_init_weight[cur] = id_to_seq[cur][0] if id_to_seq[cur] else 0
        action_name_ids[action_name] = cur
        action_ids.append(cur)
        cur += 1

    # Pre-encode the fixed action-text fragments: each action's closing + each param's surface_form_suffix (the
    # separator between params / the terminator of the last one).
    # value_open/value_close ('"'/'[') are no longer written into the context -- the model writes the whole value
    # string itself. type is kept so reconstruction can parse by type (int/str/list/...).
    surface_form: dict[str, dict] = {}
    for action_name, cfg in actions_schema.items():
        params = cfg.get("params", {})
        arguments_surface_form = []
        for pname, pcfg in params.items():
            arguments_surface_form.append({
                "name": pname,
                "type": str(pcfg.get("type", "")),
                "surface_form_suffix_ids": enc(pcfg.get("surface_form_suffix", ".")),
            })
        surface_form[action_name] = {
            "closing_ids": enc(cfg.get("closing", ".")) if not params else [],
            "params": arguments_surface_form,
        }

    return {
        "mode": "codegym",
        "num_embeddings_size": vocab_size,
        "actions_schema": actions_schema,
        "enter_str": enter_str,
        "enter_trigger_ids": _build_trigger_ids(tokenizer, enter_str),
        # The canonical token sequence of a multi-token marker (e.g.
        # <|FunctionCallBegin|>=[27,91,5152,7220,11135,91,29]); the router uses it for
        # "prefix trigger + tail completion" entry detection (tokenizer-free, pure id-sequence matching).
        "enter_seq": enc(enter_str),
        # The exit marker's surface-form sequence (<|FunctionCallEnd|> / </Action>); on finish it is
        # force-written and then eos is forced.
        "exit_str": exit_str,
        "exit_surface_form_seq": enc(exit_str),
        "value_end_token_ids": _build_newline_ids(tokenizer),
        # Optional robust termination: the model emitting the exit marker straight from ARGUMENT_VALUE
        # also ends the value.
        # Enabled only when the schema sets routing.value_end_on_exit=true (used by the GSM8K
        # calculator / ALFWorld, since '</' cannot appear in an arithmetic expression or an object
        # name); codegym does not set this flag by default -> empty set -> behavior entirely unchanged.
        # It collects **every token whose decoding contains the exit marker's first two characters
        # ('</')**: BPE frequently merges the boundary character with '</' into a single token
        # (measured: in '1.</Action>' the '.</' is the single token 3918), and recognizing only '</'
        # would miss it, leaving the action forever unclosed.
        "exit_value_end_ids": (
            _build_exit_value_end_ids(tokenizer, exit_str) if routing.get("value_end_on_exit") else []
        ),
        "str_to_id": str_to_id,
        "id_to_seq": id_to_seq,
        "id_to_str": id_to_str,
        "id_to_init_weight": id_to_init_weight,
        "action_ids": action_ids,
        "action_name_ids": action_name_ids,
        "turn_end_ids": turn_end_ids,
        "surface_form": surface_form,
        "total_size": len(action_ids),
    }

