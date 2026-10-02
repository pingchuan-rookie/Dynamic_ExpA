"""The shared decision envelope; environment parsers own the action payload."""
from __future__ import annotations

import re


def extract_action(text: str) -> tuple[str, str]:
    """Require one closed reasoning block followed by one nonempty action block.

    Do not strip or rewrite the payload: tool arguments and final answers must
    retain their original values. CodeGym uses its own function-call protocol.
    """
    if not isinstance(text, str):
        raise ValueError("Expected a text decision")
    match = re.fullmatch(r"\s*<think>(.*?)</think>\s*<action>(.*?)</action>\s*", text, re.DOTALL)
    if match is None or not match[2].strip():
        raise ValueError("Expected <think>...</think><action>...</action>")
    if "</think>" in match[1] or re.search(r"</action>\s*<action>", match[2]):
        raise ValueError("Expected exactly one decision envelope")
    return match[1], match[2]


def extract_native_decision(text: str) -> tuple[str, str]:
    """Require one closed reasoning block followed by a nonempty native decision.

    Native tool templates instruct the model to reply with bare tool calls, so
    these environments use the model's own call syntax as the action instead of
    an outer <action> block. The decision is returned unaltered; the native parser
    rejects any text around calls, and tool arguments may contain arbitrary tags.
    """
    if not isinstance(text, str):
        raise ValueError("Expected a text decision")
    match = re.fullmatch(r"\s*<think>(.*?)</think>(.*)", text, re.DOTALL)
    if match is None or not match[2].strip():
        raise ValueError("Expected <think>...</think> followed by native tool calls or a final answer")
    if "<think>" in match[1]:
        raise ValueError("Expected exactly one reasoning block")
    return match[1], match[2]
