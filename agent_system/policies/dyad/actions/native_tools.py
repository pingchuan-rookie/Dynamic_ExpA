"""Dynamic tool-name selection with ordinary LM-generated open arguments."""
from __future__ import annotations

from copy import deepcopy
import json


def compile_tools(tokenizer, vocab_size, tools, capacity, *, environment="native_tools"):
    from agent_system.policies.dyad.actions.codegym_schema import build_codegym_schema
    from agent_system.policies.dyad.actions.codegym_tasks import compile_task
    from agent_system.parsers.native_tools import native_protocol

    protocol = native_protocol(tokenizer)
    if max(tokenizer.get_vocab().values(), default=-1) >= vocab_size:
        raise ValueError("Action vocabulary must include native tool special tokens")
    definitions = [deepcopy(tool["function"]) for tool in tools]
    names = [definition["name"] for definition in definitions]
    if not names or len(names) != len(set(names)) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Tools require unique, nonempty names")
    raw = build_codegym_schema({"env_name": environment, "actions": names, "params": {}})
    # The native tool-call special token is stable across following text. BPE can
    # merge '=' with the first tool-name character, so '<function=' is not a safe trigger.
    enter = "<tool_call>"
    header = "\n<function=" if protocol == "qwen_xml" else '\n{"name": '
    raw["markers"].update(enter=enter, exit="", argument_value_end="", value_end_on_exit=False,
                          max_value_tokens=0, emit_eos=False)
    for definition in definitions:
        name = definition["name"]
        surface = name + ">" if protocol == "qwen_xml" else json.dumps(name) + ', "arguments":'
        raw["actions"][name].update(description=definition.get("description", name), surface_form=surface,
                                   mcp={"name": name, "description": definition.get("description", ""),
                                        "inputSchema": definition["parameters"]})
    cfg = compile_task(tokenizer, vocab_size, raw, capacity)
    cfg["native_tool_protocol"] = protocol
    cfg["markers"].update(value_end_ids=[], exit_value_end_ids=[], turn_end_ids=[],
                          native_action_prefix_seq=tokenizer.encode(header, add_special_tokens=False),
                          native_exit_sequences=[tokenizer.encode("</tool_call>", add_special_tokens=False)])
    cfg["value_end_token_ids"] = []
    cfg["exit_value_end_ids"] = []
    cfg["turn_end_ids"] = []
    return cfg


def verify_trace(trace, cfg):
    """Verify every actual expanded selection, including an empty selection list."""
    from agent_system.policies.dyad.actions.action_router import ActionRouter

    if not trace or trace.get("unified") is not True or trace.get("action_config") != cfg:
        raise ValueError("Native sampling action context is missing or mismatched")
    router = ActionRouter(cfg)
    names = {value: name for name, value in cfg["action_name_ids"].items()}
    selections = []
    for raw in trace["raw_token_ids"]:
        while router.decision().kind == "force":
            router.advance(router.decision().forced_token)
        decision = router.decision()
        if decision.kind == "expanded":
            if raw not in decision.allowed_ids:
                raise ValueError("Native action lies outside its task's allowed set")
            selections.append({"name": names[raw], "selected_id": raw,
                               "allowed_ids": decision.allowed_ids, "replay_verified": True})
        elif not 0 <= raw < router.V:
            raise ValueError("Native expanded selection appeared outside the action phase")
        router.advance(raw)
    return selections
