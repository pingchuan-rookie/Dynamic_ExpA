"""Strict native function-call parsing shared by independent tool environments."""
from __future__ import annotations

import json
import re

from agent_system.parsers.action_envelope import extract_native_decision


class NativeToolFormatError(ValueError):
    """The assistant produced an incomplete or invalid native tool call."""


def native_protocol(tokenizer):
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        template = template.get("tool_use") or template.get("default")
    if not isinstance(template, str) or "<tool_call>" not in template:
        raise ValueError("Native tools require a native tool-call chat template")
    return "qwen_xml" if "<function=" in template else "json"


def decode_response(tokenizer, token_ids):
    ignored = {value for value in (getattr(tokenizer, "eos_token_id", None),
                                   getattr(tokenizer, "pad_token_id", None)) if value is not None}
    return tokenizer.decode([token for token in token_ids if token not in ignored], skip_special_tokens=False)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise NativeToolFormatError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(text):
    try:
        return json.loads(text, object_pairs_hook=_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(NativeToolFormatError("Nonfinite JSON number")))
    except (ValueError, TypeError) as exc:
        raise NativeToolFormatError("Invalid JSON tool arguments") from exc


def _xml_call(body, definitions):
    match = re.fullmatch(r"\s*<function=([^<>\s]+)>\s*(.*?)\s*</function>\s*", body, re.DOTALL)
    if match is None:
        raise NativeToolFormatError("Invalid native function block")
    name, params = match.groups()
    if name not in definitions:
        raise NativeToolFormatError(f"Unknown tool: {name}")
    properties = definitions[name].get("parameters", {}).get("properties", {})
    arguments = {}
    pattern = re.compile(r"<parameter=([^<>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
    offset = 0
    for item in pattern.finditer(params):
        if params[offset:item.start()].strip():
            raise NativeToolFormatError("Unexpected text between parameters")
        key, value = item.groups()
        if key in arguments:
            raise NativeToolFormatError(f"Duplicate parameter: {key}")
        schema = properties.get(key, {})
        arguments[key] = value if schema.get("type") == "string" else _json(value.strip())
        offset = item.end()
    if params[offset:].strip():
        raise NativeToolFormatError("Incomplete native parameter block")
    return name, arguments


def parse_response(text: str, tools: list[dict], protocol: str, *, call_prefix: str = "native",
                   require_reasoning: bool = True) -> dict:
    """Preserve call order and string values; never repair a partial response."""
    definitions = {tool["function"]["name"]: tool["function"] for tool in tools}
    if require_reasoning or text.lstrip().startswith("<think>"):
        try:
            _, visible = extract_native_decision(text)
        except ValueError as exc:
            raise NativeToolFormatError(str(exc)) from exc
    else:
        visible = text
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    calls, content, offset = [], [], 0
    for match in pattern.finditer(visible):
        between = visible[offset:match.start()]
        if "<tool_call>" in between or "</tool_call>" in between:
            raise NativeToolFormatError("Incomplete tool-call block")
        if between.strip():
            raise NativeToolFormatError("Unexpected text before tool calls")
        content.append(between)
        body = match.group(1)
        if protocol == "qwen_xml":
            name, arguments = _xml_call(body, definitions)
        elif protocol == "json":
            value = _json(body)
            if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
                raise NativeToolFormatError("Tool call requires name and arguments")
            name, arguments = value["name"], value["arguments"]
        else:
            raise ValueError(f"Unsupported native tool protocol: {protocol}")
        if not isinstance(name, str) or name not in definitions:
            raise NativeToolFormatError("Unknown tool name")
        if not isinstance(arguments, dict):
            raise NativeToolFormatError("Tool arguments must be a JSON object")
        calls.append({"id": f"{call_prefix}_{len(calls)}", "name": name, "arguments": arguments})
        offset = match.end()
    remaining = visible[offset:]
    if any(marker in remaining for marker in ("<tool_call", "</tool_call", "<function=", "<parameter=")):
        raise NativeToolFormatError("Incomplete tool-call block")
    if calls and remaining.strip():
        raise NativeToolFormatError("Unexpected answer after tool calls")
    if not calls and re.search(r"</?think>", remaining):
        raise NativeToolFormatError("Expected exactly one reasoning block")
    content.append(remaining)
    return {"raw_text": text, "content": "".join(content).strip(), "tool_calls": calls}
