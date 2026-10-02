# Copyright 2025 dyad2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Load project tool parsers only when their protocol is selected.

Call ``ensure_parser(name)`` before ``ToolParser.get_tool_parser(name, tokenizer)``.
Protocols owned by verl keep using its built-in registration and error handling.
"""
from importlib import import_module


_PARSER_MODULES = {
    "react": "agent_system.parsers.react",
    "react_fc": "agent_system.parsers.react",
    "codegym": "agent_system.parsers.codegym",
    "dive": "agent_system.parsers.dive",
    "dyad": "agent_system.policies.dyad.parser",
}


def ensure_parser(name: str) -> None:
    module = _PARSER_MODULES.get(name)
    if module is not None:
        import_module(module)


__all__ = ["ensure_parser"]
