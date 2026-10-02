# Copyright 2025 ExpA_sys
"""Actions described in plain English.

Deliberately **not** a paraphrase of the MCP JSON. This form exists to answer "does the encoder
need the schema, or just the words", and a prose form that is JSON-with-sentences-around-it cannot
answer it.

Still derived from the compiled schema rather than hand-written: a hand-written description drifts
from the schema the moment a parameter is added, and nothing would report it -- the encoder would
simply be reading a stale account of the action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_system.policies.dyad.models.action_descriptions import CATALOGUE_TEMPLATE

if TYPE_CHECKING:  # pragma: no cover
    from agent_system.policies.dyad.models.action_descriptions import ActionDoc

# Match the MCP wrapper and row selection; only the body changes to prose.
PROMPT_TEMPLATE = """\
You are reading the definition of "{name}" that an agent can take in this environment.

{body}

{label}: {name}
Description: {description}

Understand this "{name}": what it does, what input it needs, and when an agent would choose it.\
"""

# Closed-value rows describe their full value set independently of any referencing tool.
VALUE_TEMPLATE = """\
You are reading the definition of "{name}" that an agent can take in this environment.

{body}

Understand this "{name}": what it does, what input it needs, and when an agent would choose it.\
"""

LABELS = {"tool": "Action"}


def body(doc: "ActionDoc") -> str:
    """Render prose from the same doc.mcp used by the MCP form.

    Derive descriptions from the shared structure to avoid stale parallel schemas.
    Do not reintroduce enum/const inventories removed before rendering.
    """
    mcp = doc.mcp or {}
    lines = [str(mcp.get("description") or "").strip() or f"The `{doc.owner or doc.name}` action."]
    schema = mcp.get("inputSchema") or {}
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    if not props:
        return "\n".join(lines + ["", "It takes no input."])

    lines += ["", "It takes the following input:"]
    for pname, prop in props.items():
        bits = [f"- `{pname}`", "(required)" if pname in required else "(optional)"]
        ptype = prop.get("type")
        if ptype:
            bits.append(f"of type {ptype}")
        desc = str(prop.get("description") or "").strip()
        if desc:
            bits.append(f"- {desc}")
        lines.append(" ".join(bits))
    return "\n".join(lines)


def value_body(doc: "ActionDoc") -> str:
    """Render the complete value set as prose, without naming an owning action."""
    choices = ", ".join(f"`{v}`" for v in doc.value_choices)
    return (f"It is one of the {len(doc.value_choices)} values that the `{doc.value_set}` "
            f"slot accepts:\n{choices}")


def format_action(doc: "ActionDoc") -> str:
    if doc.kind == "value":
        return VALUE_TEMPLATE.format(name=doc.name, body=value_body(doc))
    return PROMPT_TEMPLATE.format(
        name=doc.name,
        body=body(doc),
        label=LABELS.get(doc.kind, "Action"),
        description=doc.description,
    )


def format_catalogue(docs: list["ActionDoc"]) -> str:
    """Render one catalogue entry per tool, using its name and description.

    Match the MCP form's selection and omit synthetic kind labels so the forms
    differ only in how they express the same definitions.
    """
    entries = [
        f"- `{doc.name}`: {(doc.mcp or {}).get('description', doc.description).strip()}"
        for doc in docs if doc.kind == "tool"
    ]
    return CATALOGUE_TEMPLATE.format(catalogue_body="\n".join(entries))
