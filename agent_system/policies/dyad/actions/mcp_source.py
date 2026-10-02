# Copyright 2025 ExpA_sys
"""Load action semantics and value inventories independently of surface forms.

mcp.json defines tools and parameters; <form>.yaml defines serialization;
values.yaml defines environment instances. Different surface forms share the
same MCP semantics. The surface schema selects a subset of defined actions;
an action absent from MCP is invalid.

ALFWorld and GSM8K definitions are authored directly. CodeGym definitions are
generated per task by experiments/shared/dataset/codegym.py schema. Inventories
belong in values.yaml, while surface value_set references bind slots to them.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

# Unions reuse their member sets' extended ids instead of allocating duplicate rows.
UNION = "union"


class McpDefinition:
    """MCP tool definitions indexed by action name, without value inventories."""

    def __init__(self, doc: dict[str, Any], source: Path | str = ""):
        self.source = str(source)
        self.env = str(doc.get("env") or "")
        if "$defs" in doc:
            raise ValueError(
                f"{self.source} 里有 `$defs`。值域搬到同目录的 values.yaml 了 —— mcp.json 描述"
                "动作是什么，不描述这个环境此刻有哪些实例。见本模块 docstring。"
            )
        self._tools: dict[str, Any] = {}
        for tool in doc.get("tools") or []:
            name = tool.get("name")
            if not name:
                raise ValueError(f"{self.source}: 有一个 tool 没有 name，无法被 yaml 引用")
            if name in self._tools:
                raise ValueError(
                    f"{self.source}: 动作 {name!r} 定义了两次。后一份会赢，而两份的参数结构"
                    "若不同，head 的某几行就会读到另一个动作的定义。"
                )
            self._tools[name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        """Return action names in MCP declaration order."""
        return list(self._tools)

    def tool(self, name: str) -> dict[str, Any]:
        """Return a deep copy of an action definition so callers cannot mutate shared state."""
        if name not in self._tools:
            raise KeyError(
                f"{self.source} 里没有动作 {name!r}；有的是 {', '.join(self.names())}。"
                "surface form yaml 只能挑 mcp.json 里已经定义的动作 —— 一个没有定义的动作"
                "会占掉一行 head，而 encoder 无从知道它是什么。"
            )
        return deepcopy(self._tools[name])

    def params(self, name: str) -> dict[str, Any]:
        """Return parameter slots in declaration order, or an empty dict for no parameters."""
        return dict(self.tool(name).get("inputSchema", {}).get("properties") or {})


class ValueSets:
    """Value sets loaded from values.yaml, with unions resolved independently of MCP."""

    def __init__(self, raw: dict[str, Any], source: Path | str = ""):
        self.source = str(source)
        self._raw = {k: v for k, v in (raw or {}).items() if not str(k).startswith("_")}

    def names(self) -> list[str]:
        return list(self._raw)

    def is_union(self, set_name: str) -> bool:
        node = self._raw.get(set_name)
        return isinstance(node, dict) and UNION in node

    def union_members(self, set_name: str) -> list[str]:
        """Return union member names in declaration order, or [] for a non-union.

        The compiler uses member identities to reuse allocated extended ids. Flattening
        the union here would allocate duplicate head rows for the same values.
        """
        node = self._raw.get(set_name)
        return list(node[UNION]) if self.is_union(set_name) else []

    def values(self, set_name: str) -> list[str]:
        """Return expanded values in declaration order, including resolved unions.

        Preserve this order because it determines action-head row identities.
        """
        return self._values_of(set_name, seen=())

    def all(self) -> dict[str, list[str]]:
        return {name: self.values(name) for name in self.names()}

    def _values_of(self, set_name: str, seen: tuple) -> list[str]:
        if set_name in seen:
            raise ValueError(
                f"{self.source}: {set_name!r} 的 union 成环：{' -> '.join(seen + (set_name,))}")
        node = self._raw.get(set_name)
        if node is None:
            raise KeyError(
                f"{self.source} 里没有值域 {set_name!r}；有的是 "
                f"{', '.join(self.names()) or '（空）'}"
            )
        if isinstance(node, list):
            return [str(v) for v in node]
        if self.is_union(set_name):
            out: list[str] = []
            for member in node[UNION]:
                for value in self._values_of(str(member), seen + (set_name,)):
                    if value not in out:
                        # Overlapping union members are valid; each value still occupies one head row.
                        out.append(value)
            return out
        raise ValueError(
            f"{self.source}: {set_name!r} 既不是清单也不是 union，无法展开成取值")


def load_mcp(path: Path | str) -> McpDefinition:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 MCP 定义 {path}。每个 env 的 `schemas/<env>/mcp.json` 是手写的动作定义，"
            "surface form yaml 只描述这些动作怎么写进 context。"
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} 不是合法 JSON：{exc}") from exc
    return McpDefinition(doc, source=path)


def load_values(path: Path | str | None) -> ValueSets:
    """Load values.yaml, returning empty sets when no file is provided or found.

    Open slots need no inventory. The compiler rejects a missing set when a closed
    slot actually references it.
    """
    if path is None:
        return ValueSets({}, source="(none)")
    path = Path(path)
    if not path.exists():
        return ValueSets({}, source=str(path))
    import yaml

    return ValueSets(yaml.safe_load(path.read_text(encoding="utf-8")) or {}, source=path)


__all__ = ["McpDefinition", "ValueSets", "load_mcp", "load_values", "UNION"]
