#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified action schema compiler (new format, router=unified).

Compiles the new unified yaml (markers / argument_order / head / actions / value_sets) into a single
**superset action_config**:

  1. Flat legacy keys (num_embeddings_size / total_size / id_to_seq / id_to_str / str_to_id /
     id_to_init_weight / action_ids / action_name_ids / value_end_token_ids /
     exit_value_end_ids / turn_end_ids / surface_form / enter_seq / exit_surface_form_seq / mode ...)
     -- so the existing codegym consumers can be reused with zero changes (tool_parser env
     serialization / head-init / dyad_tool_agent_loop diagnostics).

  2. Structured new keys (markers / actions / value_sets / head / argument_order /
     id_to_description / router="unified" / env_name) -- the single source of truth for the new
     ActionRouter.

The compiler reuses the CodeGym token encoder, then allocates enumerated value ids where
configured. Parameter names are schema metadata and do not allocate action-head rows.
"""
from __future__ import annotations

from typing import Any


def _enc(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False)) if text else []


# The slot name of the single free-text segment the compiler synthesizes per action under
# argument_order="free" (Dyad free-argument). There is no param-key concept here; this name is merely the
# dict key downstream consumers read the value under.
FREE_PARAM_NAME = "arguments"


# Open-value suffix boundaries avoid newline terminators that BPE can merge with content.
# The router detects the next surface segment's prefix, ends the value, then force-writes
# the remaining suffix to keep context canonical without another prompt-level protocol.
_DECODED_VOCAB_CACHE: dict[int, list[str]] = {}


def _decoded_vocab(tokenizer, vocab_size: int) -> list[str]:
    """Per-token decoded text for [vocab_size] (cached per tokenizer instance, scanned only once repo-wide)."""
    key = id(tokenizer)
    cached = _DECODED_VOCAB_CACHE.get(key)
    if cached is not None and len(cached) >= vocab_size:
        return cached
    try:
        texts = list(tokenizer.batch_decode([[i] for i in range(vocab_size)]))
    except Exception:
        texts = []
        for i in range(vocab_size):
            try:
                texts.append(tokenizer.decode([i]))
            except Exception:
                texts.append("")
    _DECODED_VOCAB_CACHE[key] = texts
    return texts


_ENTER_MATCHER_CACHE: dict[tuple[Any, int, str], dict[str, Any]] = {}


def _compile_enter_matcher(tokenizer, vocab_size: int, marker: str, canonical: list[int]) -> dict[str, Any]:
    """Compile sparse token transitions, independent of the sampled BPE segmentation.

    States are lengths of a matched marker prefix, never accumulated response text.
    A token that already includes text after the marker cannot select an action name
    retroactively: that crossing is represented by -1 and fails loudly at runtime.
    """
    key = (tokenizer, vocab_size, marker)
    if key in _ENTER_MATCHER_CACHE:
        return _ENTER_MATCHER_CACHE[key]
    if not marker or len(canonical) < 2:
        return {}
    trigger = tokenizer.decode(canonical[:-1])
    if not trigger or not marker.startswith(trigger):
        return {}
    transitions = [{} for _ in marker]
    for tid, text in enumerate(_decoded_vocab(tokenizer, vocab_size)):
        if not text:
            continue
        for state, row in enumerate(transitions):
            combined = marker[:state] + text
            end = combined.find(marker)
            if end >= 0:
                target = len(marker) if end + len(marker) == len(combined) else -1
            else:
                target = next((size for size in range(min(len(marker) - 1, len(combined)), 0, -1)
                               if combined.endswith(marker[:size])), 0)
            if target:
                row[tid] = target
    compiled = {
        "transitions": transitions,
        "length": len(marker),
        "completion": {size: _enc(tokenizer, marker[size:])
                       for size in range(len(trigger), len(marker))},
    }
    _ENTER_MATCHER_CACHE[key] = compiled
    return compiled


def _suffix_value_end(tokenizer, vocab_size: int, suffix: str, exit_str: str,
                      _cache: dict[tuple[str, str], list[int]]) -> list[int]:
    """suffix -> the list of **terminating token ids** (the model writing one of them ends this param value).

    Two kinds of token are collected:
      (1) tokens whose decoded text is a prefix of suffix (leading whitespace kept), e.g. ' from' /
          ' fr' / '.';
      (2) tokens whose decoded text starts with suffix and whose remainder is whitespace or a prefix
          of the exit marker -- BPE frequently merges the boundary character with the following
          characters into one token (measured: in '1.</Action>' the '.</' is the single token 3918),
          and not collecting it would make the value never end, leaving the action unclosed.

    Two filters (to prevent mis-triggering inside the value): pure-whitespace prefixes are dropped;
    a prefix must have at least 2 characters after whitespace stripping (otherwise only suffix itself
    is taken) -- ' from ' yields ' fr'/' from'/... but not ' f'; '.' is still kept because it
    equals the suffix. No lstrip variants are used ('fr'/'from' style tokens without a leading space
    would mis-trigger inside a value such as 'candle').
    """
    key = (suffix, exit_str)
    if key in _cache:
        return _cache[key]
    if not suffix:
        _cache[key] = []
        return _cache[key]

    prefixes: set[str] = set()
    for k in range(1, len(suffix) + 1):
        p = suffix[:k]
        core = p.strip()
        if not core:
            continue
        if len(core) < 2 and p != suffix:
            continue
        prefixes.add(p)

    exit_prefixes = {exit_str[:k] for k in range(1, len(exit_str) + 1)} if exit_str else set()

    ids: list[int] = []
    for tid, txt in enumerate(_decoded_vocab(tokenizer, vocab_size)):
        if not txt:
            continue
        if txt in prefixes:
            ids.append(int(tid))
            continue
        if txt.startswith(suffix):
            tail = txt[len(suffix):]
            if not tail.strip() or tail in exit_prefixes:
                ids.append(int(tid))
    _cache[key] = ids
    return ids


def _codegym_to_unified_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize an **old codegym format** yaml (actions_schema + routing) into a new unified-format raw.

    Lets the 22366 auto-generated codegym schemas run through the unified ActionRouter without being
    rewritten: all params open, fixed order, no closed sets, no param key.
    """
    routing = raw.get("routing", {}) or {}
    actions: dict[str, Any] = {}
    for name, a in (raw.get("actions_schema", {}) or {}).items():
        params: dict[str, Any] = {}
        for pname, p in (a.get("params", {}) or {}).items():
            params[pname] = {
                "description": p.get("desc", ""),
                "value_kind": "open",
                "type": p.get("type", "str"),
                "surface_form_suffix": p.get("surface_form_suffix", ""),
            }
        actions[name] = {
            "description": a.get("desc", ""),
            "surface_form": a.get("surface_form", ""),
            "init_weight_with": a.get("init_weight_with", 0),
            "params": params,
        }
    return {
        "env_name": raw.get("env_name", "codegym"),
        "source": raw.get("source", ""),
        "template_style": raw.get("template_style", "base"),
        "router": "unified",
        "mode": "codegym",
        "markers": {
            "enter": routing.get("enter_str", "<|FunctionCallBegin|>"),
            "exit": routing.get("exit_str", ""),
            "argument_value_end": "\n",
            "value_end_on_exit": bool(routing.get("value_end_on_exit", False)),
            "emit_eos": bool(routing.get("emit_eos", True)),
        },
        "argument_order": "fixed",
        "head": {"action_name": True, "argument_key": False, "closed_value": False, "init_from": "mean_pool"},
        "actions": actions,
        "value_sets": {},
    }


def _to_codegym_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate a new unified yaml into codegym-raw (so compile_flat_schema can do the flat compilation)."""
    markers = raw.get("markers", {}) or {}
    actions = raw.get("actions", {}) or {}
    actions_schema: dict[str, Any] = {}
    for name, a in actions.items():
        arguments_out: dict[str, Any] = {}
        for pname, p in (a.get("params", {}) or {}).items():
            # Whether open or closed, params are fed to the codegym compiler in the open_free shape --
            # codegym only allocates action-name extended ids and pre-encodes the action surface form and
            # the connectors, it **allocates no param-value ids**. The value extended ids of closed params are
            # allocated **natively** by compile_action_schema below (reading value_sets).
            # type is preserved (codegym int/str/list affects env deserialization parsing).
            arguments_out[pname] = {
                "argument_kind": "open_free",
                "type": p.get("type", "str"),
                "surface_form_suffix": p.get("surface_form_suffix", ""),
            }
        actions_schema[name] = {
            "surface_form": a.get("surface_form", ""),
            "init_weight_with": a.get("init_weight_with", 0),
            "desc": a.get("description", ""),
            "params": arguments_out,
        }
    return {
        "env_name": raw.get("env_name", "unified"),
        "source": raw.get("source", raw.get("env_name", "")),
        "mode": "codegym",
        "template_style": raw.get("template_style", "base"),
        "routing": {
            "enter_str": markers.get("enter", "<action>"),
            "exit_str": markers.get("exit", "</action>"),
            "emit_eos": bool(markers.get("emit_eos", True)),
            "value_end_on_exit": bool(markers.get("value_end_on_exit", False)),
        },
        "actions_schema": actions_schema,
    }


def _merge_mcp(raw: dict[str, Any], mcp, values=None) -> dict[str, Any]:
    """Merge MCP semantics and value sets into a surface schema.

    Keep each action's full MCP definition separate from router params, which can
    be empty for free-argument routing. Render value prefixes from the surface
    configuration so semantic names remain independent of tokenizer spacing.
    """
    out = dict(raw)
    prefix = str(raw.get("value_surface_prefix", "") or "")
    actions_out: dict[str, Any] = {}

    for name, action in (raw.get("actions") or {}).items():
        if name not in mcp:
            raise ValueError(
                f"surface form yaml 里的动作 {name!r} 在 {mcp.source} 里没有定义。"
                f"有定义的是 {', '.join(mcp.names())}。一个没有 MCP 定义的动作会占掉一行 head，"
                "而 encoder 无从知道它是什么。"
            )
        entry = dict(action)
        definition = mcp.tool(name)
        entry["description"] = definition.get("description", "")
        entry["mcp"] = definition

        mcp_params = mcp.params(name)
        params_out: dict[str, Any] = {}
        for pname, slot in (action.get("params") or {}).items():
            if pname not in mcp_params:
                raise ValueError(
                    f"surface form yaml 给 {name!r} 声明了槽 {pname!r}，但 {mcp.source} 的 "
                    f"{name!r} 没有这个参数（有的是 {', '.join(mcp_params) or '（无）'}）。"
                    "槽必须对应一个真实参数，否则 router 会为一个不存在的东西留出位置。"
                )
            merged = dict(slot)
            merged["description"] = mcp_params[pname].get("description", "")
            params_out[pname] = merged
        entry["params"] = params_out
        actions_out[name] = entry

    out["actions"] = actions_out
    # Apply surface prefixes to values loaded independently of MCP semantics.
    # Preserve union references so compilation reuses member ids instead of allocating duplicate rows.
    value_sets_out: dict[str, Any] = {}
    if values is not None:
        for set_name in values.names():
            if values.is_union(set_name):
                value_sets_out[set_name] = {"union": values.union_members(set_name)}
            else:
                value_sets_out[set_name] = [prefix + v for v in values.values(set_name)]
    out["value_sets"] = value_sets_out
    return out


def compile_action_schema(tokenizer, vocab_size: int, raw: dict[str, Any], mcp=None,
                          values=None) -> dict[str, Any]:
    """Compile unified YAML into flat legacy keys and structured action_config keys.

    Require router="unified". Optional mcp and values inputs provide the separate
    semantic definitions and inventories; otherwise raw must contain the legacy
    descriptions and value sets.
    """
    from agent_system.policies.dyad.actions.flat_schema import compile_flat_schema

    # Normalize the input: old codegym format (actions_schema + routing, no new markers/actions)
    # -> new unified format.
    if "actions" not in raw and "actions_schema" in raw:
        raw = _codegym_to_unified_raw(raw)

    if mcp is not None:
        raw = _merge_mcp(raw, mcp, values)

    if raw.get("argument_order", "fixed") not in ("free", "fixed") or (raw.get("head") or {}).get("argument_key"):
        raise ValueError("Only free arguments or fixed value slots are supported")

    # 1) Route through the codegym compiler to obtain every flat key (extended-id allocation,
    #    enter/exit/value_end pre-encoding included).
    codegym_raw = _to_codegym_raw(raw)
    cfg = compile_flat_schema(tokenizer, vocab_size, codegym_raw)

    markers = raw.get("markers", {}) or {}
    actions = raw.get("actions", {}) or {}
    head = raw.get("head", {}) or {}
    str_to_id: dict[str, int] = cfg["str_to_id"]

    # 2) Structured markers (what ActionRouter reads; the values match the flat keys, only the naming
    #    is consolidated).
    cfg["markers"] = {
        "enter_seq": list(cfg.get("enter_seq", [])),
        "enter_str": markers.get("enter", cfg.get("enter_str", "")),
        "exit_seq": list(cfg.get("exit_surface_form_seq", [])),
        "value_end_ids": list(cfg.get("value_end_token_ids", [])),
        "exit_value_end_ids": list(cfg.get("exit_value_end_ids", [])),
        "turn_end_ids": list(cfg.get("turn_end_ids", [])),
        # Token cap for a single open value (0 = unlimited): beyond it the router forces termination,
        # so a missed terminator cannot deadlock the state machine.
        "max_value_tokens": int(markers.get("max_value_tokens", 0) or 0),
    }

    enter_matcher = _compile_enter_matcher(
        tokenizer, vocab_size, cfg["markers"]["enter_str"], cfg["markers"]["enter_seq"])
    if enter_matcher:
        cfg["markers"]["enter_matcher"] = enter_matcher

    if markers.get("preserve_exit_prefix", False):
        # Native bracket commands cannot discard the ']' (or value text) in a BPE
        # token such as ']</'. Opt in only for FREE_TEXT, leaving existing slot
        # and environment termination semantics unchanged.
        if raw.get("argument_order") != "free" or not markers.get("value_end_on_exit"):
            raise ValueError("preserve_exit_prefix requires free arguments and value_end_on_exit")
        exit_str = str(markers.get("exit") or "")
        if len(exit_str) < 2:
            raise ValueError("preserve_exit_prefix requires an exit marker of at least two characters")
        prefixes = {}
        partial = bool(markers.get("preserve_partial_exit_prefix", False))
        if partial:
            # Some tokenizers merge the final JSON bracket with '<', but keep '|'
            # separate. This is opt-in for the versioned native CodeGym protocol.
            merged = [tid for tid, text in enumerate(_decoded_vocab(tokenizer, vocab_size))
                      if text.endswith(exit_str[0])]
            cfg["markers"]["exit_value_end_ids"] = sorted(set(cfg["markers"]["exit_value_end_ids"] + merged))
        for tid in cfg["markers"]["exit_value_end_ids"]:
            text = tokenizer.decode([tid])
            fragment = exit_str[:2]
            if partial and fragment not in text and text.endswith(exit_str[0]):
                fragment = exit_str[0]
            prefix, boundary, _ = text.partition(fragment)
            if not boundary:
                raise ValueError(f"Exit token {tid} does not contain the exit marker prefix")
            prefixes[tid] = _enc(tokenizer, prefix)
        cfg["markers"]["exit_value_prefix_ids"] = prefixes

    # 3) Natively allocate the extended ids of closed values / param keys (codegym only allocated
    #    action-name ids). Action names occupy [V, V+total_size); allocation continues right after them.
    id_to_description: dict[int, str] = {}
    _cur = int(cfg["num_embeddings_size"]) + int(cfg["total_size"])

    def _register(token_str: str, surface_form_seq: list[int], description: str) -> int:
        nonlocal _cur
        aid = _cur
        _cur += 1
        cfg["id_to_str"][aid] = token_str
        cfg["id_to_seq"][aid] = list(surface_form_seq)
        cfg["id_to_init_weight"][aid] = int(surface_form_seq[0]) if surface_form_seq else 0
        id_to_description[aid] = description
        cfg["action_ids"].append(aid)
        return aid

    # 3a) Closed value sets: one extended id per value (surface_form_seq = the value itself; init uses
    #     the value's first token).
    #     WARNING: only allocated when head.closed_value=true -- otherwise closed values do not enter
    #     the expanded action space. Generated values come from the vocabulary; enumerated values
    #     are selected from the expanded action space when the schema enables head.closed_value.
    value_sets_out: dict[str, list[int]] = {}
    if head.get("closed_value"):
        raw_sets = raw.get("value_sets", {}) or {}
        # A `{union: [a, b]}` entry reuses the ids already allocated for a and b instead of
        # allocating its own. That distinction is the whole point: alfworld's `examine` accepts an
        # object *or* a receptacle, and re-listing the names would allocate a second, separate
        # embedding for every one of them -- the same word would occupy two head rows that train
        # independently and can drift apart, and the head would be 83 rows larger than the action
        # space actually is. Resolved in a second pass so a union may name a set declared after it.
        for set_name, items in raw_sets.items():
            if isinstance(items, dict) and "union" in items:
                continue
            ids = []
            for item in (items or []):
                if isinstance(item, dict):
                    token = item["token"]
                    desc = item.get("description", token)
                else:
                    token = str(item)
                    desc = token
                ids.append(_register(token, _enc(tokenizer, token), desc))
            value_sets_out[set_name] = ids
        for set_name, items in raw_sets.items():
            if not (isinstance(items, dict) and "union" in items):
                continue
            members = list(items["union"])
            unknown = [m for m in members if m not in value_sets_out]
            if unknown:
                raise ValueError(
                    f"value_sets.{set_name} is a union over {unknown}, which are not declared "
                    "as plain value sets (a union may only reference sets that allocate ids)"
                )
            merged: list[int] = []
            for member in members:
                merged.extend(value_sets_out[member])
            if len(set(merged)) != len(merged):
                raise ValueError(
                    f"value_sets.{set_name} unions {members}, which share ids; the mask would "
                    "then list the same value twice"
                )
            value_sets_out[set_name] = merged
    cfg["value_sets"] = value_sets_out

    # The head dimension = the total number of extended ids (action names + closed values).
    cfg["total_size"] = len(cfg["action_ids"])

    # 4) Structured actions (surface-form prefix + ordered param slots, carrying value_kind/value_set).
    #    Open params additionally compile their suffix terminators into value_end_ids (see _suffix_value_end).
    on_suffix = bool(markers.get("value_end_on_suffix", False))
    exit_str = markers.get("exit", "") or ""
    argument_order = raw.get("argument_order", "fixed")
    suffix_cache: dict[tuple[str, str], list[int]] = {}
    all_value_end_ids: set[int] = set()
    actions_struct: dict[str, Any] = {}
    for name, a in actions.items():
        name_id = str_to_id[name]
        id_to_description[name_id] = a.get("description", "")
        arguments_struct = []
        for pname, p in (a.get("params", {}) or {}).items():
            surface_form_suffix = p.get("surface_form_suffix", "")
            slot = {
                "name": pname,
                # Carried so the compiled config alone is enough to describe an action in prose.
                # action_descriptions writes it into the MCP inputSchema fed to the LLM encoder, and
                # that runs from action_config -- the yaml is not available at that point.
                "description": p.get("description", "") or "",
                "value_kind": p.get("value_kind", "open"),
                "value_set": p.get("value_set"),
                "surface_form_suffix": surface_form_suffix,
                "surface_form_suffix_seq": _enc(tokenizer, surface_form_suffix),
                "value_end_ids": [],
            }
            # When surface_form_suffix is empty the boundary is the exit marker, handled by the
            # exit_value_end_ids path; do not collect it twice.
            if on_suffix and slot["value_kind"] != "closed" and surface_form_suffix:
                slot["value_end_ids"] = _suffix_value_end(
                    tokenizer, vocab_size, surface_form_suffix, exit_str, suffix_cache
                )
                all_value_end_ids.update(slot["value_end_ids"])
            arguments_struct.append(slot)
        if argument_order == "free":
            # Dyad free-argument: the schema declares no param at all (`params: {}` in the yaml), so the
            # compiler synthesizes **exactly one free-text segment** per action -- after the action
            # name the model generates freely all the way to the exit marker.
            #
            # Since W6 the router does NOT consume this slot as a slot: argument_order="free" goes
            # ACTION_NAME -> Phase.FREE_TEXT directly, never touching _begin_next_param /
            # _enter_argument_value / value_end / surface_form_suffix. The entry is kept for exactly one
            # reason -- **its name**: replay_unified_actions keys the recovered free text under
            # params[FREE_PARAM_NAME], and unified_action_to_raw_command reads it back out
            # under that key to rebuild the raw env command. Everything else in it is inert
            # (value_end_ids=[], surface_form_suffix_seq=[]).
            if arguments_struct:
                raise ValueError(
                    f"argument_order=free requires actions.{name}.params to be empty (the Dyad free-argument schema has no param slots), "
                    f"but it declared {[s['name'] for s in arguments_struct]}"
                )
            arguments_struct = [{
                "name": FREE_PARAM_NAME,
                "description": "",
                "value_kind": "open",
                "value_set": None,
                "surface_form_suffix": "",
                "surface_form_suffix_seq": [],
                "value_end_ids": [],
            }]
        actions_struct[name] = {
            "name_id": name_id,
            # The action surface form template (env-serialize uses it to reassemble the original command;
            # surface_form_seq is its token sequence).
            "surface_form": a.get("surface_form", "") or "",
            "surface_form_seq": list(cfg["id_to_seq"][name_id]),
            "params": arguments_struct,
            # Keep full MCP semantics separate from router slots, which are empty for free arguments.
            "mcp": a.get("mcp") or {},
        }
        # Every action selection must render ordinary tokens before the next model input.
        if head.get("action_name", True) and not cfg["id_to_seq"][name_id]:
            raise ValueError(
                f"action {name!r} in schema {raw.get('env_name', '?')!r} has an empty surface_form, "
                "but head.action_name is true. Every decision must write at least one base vocabulary "
                "token (one decode step writes one token), so an action name with no surface form "
                "cannot be sampled. Give it a short verb prefix that the env tolerates -- calc_base "
                "uses 'calc', which calc_session._CALC_PREFIXES strips back off."
            )
    cfg["actions"] = actions_struct
    # Diagnostics/logging: the union of every param's suffix terminator tokens (dyad_tool_agent_loop
    # uses it to slice out the raw param-value text).
    cfg["all_value_end_ids"] = sorted(all_value_end_ids)

    # 5) The remaining structured metadata.
    cfg["router"] = "unified"
    cfg["env_name"] = raw.get("env_name", cfg.get("env_name", "unified"))
    cfg["argument_order"] = raw.get("argument_order", "fixed")
    # env-serialize selector: alfworld = replay ActionRouter into env commands;
    # codegym (default) = decisions_to_codegym_actions.
    # Note this is decoupled from mode: alfworld keeps mode:codegym (to reuse head-init/compilation)
    # while its env-serialize goes through alfworld.
    cfg["env_serialize"] = raw.get("env_serialize", "codegym")
    cfg["head"] = {
        "action_name": bool(head.get("action_name", True)),
        "argument_key": bool(head.get("argument_key", False)),
        "closed_value": bool(head.get("closed_value", False)),
        "init_from": head.get("init_from", "mean_pool"),
    }
    cfg["id_to_description"] = id_to_description
    return cfg


# A mixed-schema batch shares one head, so schemas need disjoint extended-action IDs.
# Offset env_i into [V+off_i, V+off_i+n_i); total head size is sum(n_i).
# union["schemas"][env] stores the offset configuration used by its ActionRouter.
# Each sequence is masked to its own schema range; head initialization and serialization
# continue through the mode:codegym path.

def _shift_id(i: int, off: int, vocab_size: int) -> int:
    """Shift an extended id (>=V) right by off; base-vocabulary ids (<V, e.g. surface-form/marker/init representative tokens) stay put."""
    i = int(i)
    return i + off if i >= vocab_size else i


def _offset_extended_ids(cfg: dict[str, Any], off: int, vocab_size: int) -> dict[str, Any]:
    """Return a cfg copy with **every extended id shifted right by off** (moving a single env's [V,V+n) into the global segment).

    The **values** of id_to_init_weight are representative vocab tokens (<V) and are not shifted;
    only its **keys** (extended ids) are. markers (enter/exit/value_end etc., all base tokens <V)
    are left alone.
    """
    V = vocab_size

    def shift(i):
        return _shift_id(i, off, V)

    def shift_keys(m):
        return {shift(int(k)): v for k, v in (m or {}).items()}

    new = dict(cfg)
    new["id_to_str"] = shift_keys(cfg.get("id_to_str"))
    new["id_to_seq"] = {shift(int(k)): list(v) for k, v in (cfg.get("id_to_seq") or {}).items()}
    new["id_to_init_weight"] = shift_keys(cfg.get("id_to_init_weight"))
    new["id_to_description"] = shift_keys(cfg.get("id_to_description"))
    new["action_ids"] = [shift(i) for i in cfg.get("action_ids", [])]
    new["action_name_ids"] = {k: shift(v) for k, v in (cfg.get("action_name_ids") or {}).items()}
    new["str_to_id"] = {k: shift(v) for k, v in (cfg.get("str_to_id") or {}).items()}
    new["value_sets"] = {s: [shift(i) for i in ids] for s, ids in (cfg.get("value_sets") or {}).items()}
    acts = {}
    for name, a in (cfg.get("actions") or {}).items():
        params = [dict(p) for p in a.get("params", [])]
        acts[name] = {
            "name_id": shift(a["name_id"]),
            "surface_form_seq": list(a.get("surface_form_seq", [])),
            "params": params,
        }
    new["actions"] = acts
    return new


def compile_multi_schema(tokenizer, vocab_size: int, env_raws: dict[str, dict]) -> dict[str, Any]:
    """Compile several codegym env raw yamls into a **union action_config** (single head + per-seq masking).

    env_raws: {env_name: raw_yaml_dict} (the ordering determines the global id layout and must match
    between the rollout and training sides).
    Returns the union cfg:
      router="multi", mode="codegym", env_serialize="codegym"
      total_size = sum of each env's total_size (= the single head size)
      action_ids / id_to_str / id_to_seq / id_to_init_weight / id_to_description = each env's, offset then merged
      schemas = {env_name: that env's full cfg offset into the global segment}  -> used directly by a per-seq ActionRouter
      markers = shared (identical across codegym envs: <|call|> etc.; the first env's are taken)
      env_offsets = {env_name: (start, end)}  the global sub-range (for diagnostics/validation)
    """
    V = int(vocab_size)
    schemas: dict[str, Any] = {}
    env_offsets: dict[str, tuple] = {}
    union_action_ids: list[int] = []
    union_id_to_str: dict[int, str] = {}
    union_id_to_seq: dict[int, list[int]] = {}
    union_id_to_init_weight: dict[int, int] = {}
    union_id_to_description: dict[int, str] = {}
    markers = None
    union_head = None

    off = 0
    for env_name, raw in env_raws.items():
        per = compile_action_schema(tokenizer, V, raw)
        if per.get("mode") != "codegym":
            raise ValueError(f"compile_multi_schema only supports mode:codegym envs; {env_name} has mode={per.get('mode')!r}")
        n = int(per["total_size"])
        shifted = _offset_extended_ids(per, off, V)
        schemas[env_name] = shifted
        env_offsets[env_name] = (V + off, V + off + n)
        for gid in shifted["action_ids"]:
            if gid in union_id_to_str:
                raise ValueError(f"union id collision: {gid} (env {env_name}) -- the offset logic is wrong")
            union_action_ids.append(gid)
            union_id_to_str[gid] = shifted["id_to_str"][gid]
            union_id_to_seq[gid] = list(shifted["id_to_seq"][gid])
            union_id_to_init_weight[gid] = int(shifted["id_to_init_weight"][gid])
            union_id_to_description[gid] = (shifted.get("id_to_description") or {}).get(gid, "")
        if markers is None:
            markers = per.get("markers")
        if union_head is None:
            union_head = per.get("head")
        off += n

    return {
        "router": "multi",
        "mode": "codegym",
        "env_serialize": "codegym",
        "num_embeddings_size": V,
        "total_size": off,
        "action_ids": union_action_ids,
        "id_to_str": union_id_to_str,
        "id_to_seq": union_id_to_seq,
        "id_to_init_weight": union_id_to_init_weight,
        "id_to_description": union_id_to_description,
        "head": union_head or {"init_from": "mean_pool"},
        "schemas": schemas,
        "env_offsets": env_offsets,
        "markers": markers,
    }
