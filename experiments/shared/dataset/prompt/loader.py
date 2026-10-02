#!/usr/bin/env python3
# Copyright 2025 ExpA_verl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Read the system prompts of a dataset generator out of its markdown file.

The three generators in `experiments/shared/dataset/` used to hold their prompts as Python string literals
buried among a few hundred lines of parquet reading, expert-plan caching and process-pool code. The
prompt is the one part of a generator a human needs to read and edit, and it was the hardest part to
find. It now lives in `experiments/shared/dataset/prompt/<env>.md`, one section per prompt, and this module
is how the generator gets it back.

## The format

    ## <key>

    ```text
    the prompt, verbatim
    ```

plus, for a key that holds tabular prompt content (ALFWorld's action descriptions), a two-column
pipe table instead of a fenced block.

Nothing else is allowed in the file -- no title, no preamble, no note about what the prompts are
for. Text outside a section raises. Such a note would be a second copy of what the generator's
comments and `experiments/tools/check_dataset_prompts.py` already say, and because markdown is never
executed, a stale copy reads exactly like a current one.

## Two decisions that are not stylistic

**Leading and trailing blank lines inside a fence are stripped.** They are invisible, and an editor
that trims trailing whitespace -- or a markdown formatter that collapses blank lines before a fence
-- would otherwise change a shipped prompt without changing anything a reviewer can see. Every
newline that actually matters is therefore written at the call site instead, where it is a visible
`+ "\\n"`, `"\\n\\n" + ...`. Several of these prompts genuinely differ in exactly that byte: the
CodeGym suffixes open with two newlines because they are appended to a system message, while
`_RAW_COMMAND_EXAMPLE` ends without one because it is the last thing in its prompt.

**Substitution is `<<NAME>>` and `str.replace`, not `{name}` and `str.format`.** The CodeGym
suffixes contain literal JSON -- `[{"name": "<FunctionName>", "parameters": {<args>}}]` -- so
`str.format` would either raise `KeyError: '"name"'` or silently eat the braces. A syntax that
cannot appear in the prompts is the only one that cannot collide with them.

Anything ambiguous raises. A prompt that loads with a wrong value does not fail: it produces a
dataset with the right shape, the right row count and the right metrics, whose model was taught
something nobody wrote.

## How a generator gets hold of this module

`experiments/` is not a package, and the generators run both as `python experiments/shared/dataset/<env>.py`
(where `sys.path[0]` is their own directory) and as a module imported by path from
`dyad_test/test_dataset_prompts.py` (where it is not). Only `spec_from_file_location` works in both,
so each generator carries the same five-line bootstrap:

    def _prompt_file(env):
        spec = importlib.util.spec_from_file_location(
            "dyad_dataset_prompt_loader", Path(__file__).resolve().parent / "prompt" / "loader.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.load(env)

That is duplicated three times rather than shared, because sharing it would need exactly the import
mechanism it exists to replace. The alternative -- inserting the directory into `sys.path` so
`from prompt.loader import load` works -- was rejected: it makes an imported generator mutate global
import state, and `prompt` is a plausible enough name to shadow.

**This module must not use `@dataclass`.** A dataclass resolves its annotations through
`sys.modules[cls.__module__]`, and a module built by `module_from_spec` is not registered there, so
the class body raises `AttributeError: 'NoneType' object has no attribute '__dict__'` before any of
this runs. `NamedTuple` has no such dependency.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple, Optional

#: `<<NAME>>`. Uppercase and underscores only, so it cannot match anything a prompt would say.
_PLACEHOLDER = re.compile(r"<<[A-Z][A-Z0-9_]*>>")

_FENCE = "```"


class ActionRow(NamedTuple):
    """One row of a prompt table: an action and the one-line description shown to the model."""

    name: str
    desc: str


class PromptFile:
    """The sections of one `prompt/<env>.md`, and which of them have been read.

    `unused()` exists so a test can fail on a section nobody asks for. A dead prompt is worse than
    a dead function: it reads as the text the model sees, so the next person edits it and measures
    nothing.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._sections = _parse_sections(self.path)
        self._read: set[str] = set()

    # -- reading --------------------------------------------------------------------------------

    def text(self, key: str, **subs: str) -> str:
        """The fenced body of `## <key>`, with `<<NAME>>` replaced from `subs`.

        `subs` are given lowercase (`action_space=...`) and match `<<ACTION_SPACE>>`, so the call
        site reads like Python while the placeholder stays impossible to confuse with prompt text.
        """
        body = self._body(key)
        if body.table is not None:
            raise ValueError(
                f"{self.path}: section '{key}' holds a table, not a fenced block; use table('{key}')"
            )
        out = body.text
        for name, value in subs.items():
            out = out.replace(f"<<{name.upper()}>>", value)

        leftover = _PLACEHOLDER.findall(out)
        if leftover:
            raise ValueError(
                f"{self.path}: section '{key}' still contains {leftover} after substitution "
                f"(given: {sorted(subs)}). An unsubstituted placeholder is not a crash -- it is "
                "shipped to the model verbatim, and every shape and metric downstream stays normal."
            )
        return out

    def table(self, key: str) -> list[ActionRow]:
        """The two-column pipe table under `## <key>`, in file order."""
        body = self._body(key)
        if body.table is None:
            raise ValueError(
                f"{self.path}: section '{key}' holds a fenced block, not a table; use text('{key}')"
            )
        return body.table

    # -- introspection, for the tests and the gate ----------------------------------------------

    def keys(self) -> set[str]:
        return set(self._sections)

    def unused(self) -> set[str]:
        """Sections that were never read. Empty is the only healthy value."""
        return set(self._sections) - self._read

    def placeholders(self, key: str) -> set[str]:
        """The `<<NAME>>` markers a section declares, lowercased to match `text()`'s kwargs."""
        body = self._sections[key]
        if body.text is None:
            return set()
        return {m[2:-2].lower() for m in _PLACEHOLDER.findall(body.text)}

    def _body(self, key: str) -> "_Section":
        if key not in self._sections:
            raise KeyError(
                f"{self.path}: no section '## {key}'; it has {sorted(self._sections)}"
            )
        self._read.add(key)
        return self._sections[key]


class _Section(NamedTuple):
    text: Optional[str]
    table: Optional[list]


def load(env_or_path: Path | str) -> PromptFile:
    """Open a prompt file. Takes an env name (`"gsm8k"`) or an explicit path."""
    path = Path(env_or_path)
    if path.suffix != ".md":
        path = Path(__file__).resolve().parent / f"{env_or_path}.md"
    return PromptFile(path)


# --------------------------------------------------------------------------- parsing


def _parse_sections(path: Path) -> dict[str, _Section]:
    if not path.exists():
        raise FileNotFoundError(f"prompt file missing: {path}")
    lines = path.read_text(encoding="utf-8").split("\n")

    raw: dict[str, list[str]] = {}
    key: str | None = None
    for number, line in enumerate(lines, start=1):
        if line.startswith("## "):
            key = line[3:].strip()
            if key in raw:
                raise ValueError(
                    f"{path}: duplicate section '## {key}'. One of the two is dead text that reads "
                    "exactly like the live one."
                )
            raw[key] = []
        elif key is not None:
            raw[key].append(line)
        elif line.strip():
            # These files hold prompts and nothing else. A preamble explaining what they are for
            # would be a second copy of what the generator and the gate already say, and the two
            # copies drift with nobody noticing -- the markdown is not executed, so a stale
            # explanation of a prompt looks exactly like a current one.
            raise ValueError(
                f"{path}:{number}: text before the first '## <key>' section: {line.strip()[:60]!r}. "
                "This file holds prompts only; explanation belongs next to the code that composes "
                "them."
            )

    if not raw:
        raise ValueError(f"{path}: no '## <key>' section found")
    return {key: _parse_body(path, key, body) for key, body in raw.items()}


def _parse_body(path: Path, key: str, lines: list[str]) -> _Section:
    if any(line.lstrip().startswith("|") for line in lines):
        return _Section(text=None, table=_parse_table(path, key, lines))

    opens = [i for i, line in enumerate(lines) if line.startswith(_FENCE)]
    if len(opens) < 2:
        raise ValueError(
            f"{path}: section '{key}' has no ```text block and no table. Every section must be one "
            "or the other -- a section whose body is bare prose would silently ship that prose."
        )
    if len(opens) > 2:
        raise ValueError(
            f"{path}: section '{key}' has {len(opens)} fence lines. Exactly two are expected; a "
            "prompt containing a ``` line would close its own block early and ship truncated."
        )
    start, end = opens
    return _Section(text=_strip_blank_edges(lines[start + 1:end]), table=None)


def _strip_blank_edges(lines: list[str]) -> str:
    """Join the fenced lines, dropping blank lines at both ends. See the module docstring."""
    body = list(lines)
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    return "\n".join(body)


def _parse_table(path: Path, key: str, lines: list[str]) -> list[ActionRow]:
    """`| name | description |`, header and separator skipped."""
    rows: list[ActionRow] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) != 2:
            raise ValueError(
                f"{path}: section '{key}' has a row with {len(cells)} cells, expected 2 "
                f"(name | description): {stripped!r}"
            )
        name, desc = cells
        if name.lower() == "name" or set(name) <= set("-: "):
            continue  # header row or its separator
        rows.append(ActionRow(name=name, desc=desc))
    if not rows:
        raise ValueError(f"{path}: section '{key}' looks like a table but has no data rows")
    return rows
