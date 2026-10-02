# Copyright 2025 ExpA_sys
"""Build the exact action descriptions consumed by the action encoder.

Compile action_config into ActionDoc records, then render a shared catalogue
and each action definition through description_forms. MCP and natural-language
forms read the same structured definitions so the ablation changes presentation.
Add a form as a module under description_forms and register it in _MODULES.

Rendering deliberately accepts no environment name. Benchmark identities must
not affect embeddings of otherwise identical definitions; the action catalogue
provides the environment context. Tool definitions omit enum/const inventories;
closed-value rows describe their own value sets.
"""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Render the semantic action inputs used by both training and sampling.
# Extension point: encoder_prompts -> action encoder

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

# The whole-environment preface. Every per-action prompt is prefixed with it, so the encoder reads
# one action *in the context of the set it belongs to* rather than in isolation -- "go to" only
# means something next to "take" and "open". Shared by every form so that the two differ only in
# how an action is written, not in whether the preface exists.
#
# **The environment is never named.** See this module's docstring: the set of actions is the only
# thing about the environment an encoder input may contain.
CATALOGUE_TEMPLATE = """\
The environment offers the following actions to the agent.

{catalogue_body}

"""

#: form name -> module implementing `format_action` / `format_catalogue`.
_MODULES = {
    "mcp": "agent_system.policies.dyad.models.description_forms.mcp",
    "natural_language": "agent_system.policies.dyad.models.description_forms.natural_language",
}


def known_forms() -> list[str]:
    return sorted(_MODULES)


def _form(name: str):
    key = str(name).strip().lower()
    if key not in _MODULES:
        raise ValueError(
            f"unknown description form {name!r}; known: {', '.join(known_forms())}. "
            "Adding one is a new module under models/description_forms/ plus an entry in "
            "_MODULES -- see this package's docstring."
        )
    return import_module(_MODULES[key])


def _sanitised(doc: "ActionDoc") -> "ActionDoc":
    """Sanitize MCP fields at the final rendering boundary.

    ActionDoc can be constructed directly, so constructor-side sanitization alone
    cannot prevent enum/const inventories from leaking into tool prompts.
    """
    from dataclasses import replace

    return replace(doc, mcp=_for_prompt(doc.mcp or {}))


def build_action_prompt(doc: "ActionDoc", *, form: str, catalogue: str = "") -> str:
    """The exact string fed to the encoder for one action: `catalogue + per-action`.

    `catalogue=""` yields the per-action part alone -- which is what the encoder read before the
    preface existed, so the two are directly comparable rather than silently swapped.

    Takes no environment argument, and must not grow one: see this module's docstring.
    """
    body = _form(form).format_action(_sanitised(doc))
    return f"{catalogue}{body}" if catalogue else body


def build_catalogue(docs: list["ActionDoc"], *, form: str) -> str:
    """The whole-environment preface, in head-row order.

    Row order, not alphabetical: it is the order of `action_head`'s rows, so a reader comparing the
    catalogue against a head dump is comparing the same sequence.
    """
    return _form(form).format_catalogue([_sanitised(d) for d in docs])


def register(name: str, module_path: str) -> None:
    """Add a form at runtime. Exists so the tests can prove a third form needs no edit here."""
    _MODULES[str(name).strip().lower()] = module_path


__all__ = [
    "CATALOGUE_TEMPLATE",
    "build_action_prompt",
    "build_catalogue",
    "known_forms",
    "register",
]


@dataclass(frozen=True)
class ActionDoc:
    """One extended id, described well enough that an LLM can encode it."""

    action_id: int
    kind: str                       # tool | value
    name: str                       # "take" | "object.cabinet"
    description: str
    mcp: dict[str, Any] = field(default_factory=dict)
    # Owning tool, or the first tool referencing a shared value.
    owner: str = ""
    # Owning value-set name for a closed-value row.
    value_set: str = ""
    # Complete value inventory in head-row order, used only by closed-value prompts.
    value_choices: tuple = ()

    def prompt(self, form: str = "mcp", catalogue: str = "") -> str:
        """This action as the encoder will read it. `form` selects the description module."""
        return build_action_prompt(self, form=form, catalogue=catalogue)


# Compiler metadata is not part of MCP and must not enter rendered tool definitions.
_INTERNAL_KEYS = ("valueSet",)

# Exclude candidate inventories from tool prompts.
_BANNED_KEYS = ("enum", "enumTruncated", "const")


def _for_prompt(mcp: dict[str, Any]) -> dict[str, Any]:
    """Remove enum/const inventories and internal markers from rendered MCP.

    Inventories describe environment instances rather than tool semantics. Keeping
    them out of tool definitions avoids exposing candidate answers through every
    action prompt. This changes rendering only; compilation retains value sets.
    """
    out: dict[str, Any] = {}
    for key, node in mcp.items():
        if key in _INTERNAL_KEYS or key in _BANNED_KEYS:
            continue
        out[key] = _for_prompt(node) if isinstance(node, dict) else node
    return out


def _param_property(slot: dict[str, Any], value_sets: dict[str, list[int]], id_to_str: dict) -> dict[str, Any]:
    """One entry of the inputSchema's `properties`.

    Structured, not prose: both description forms read this, and the natural-language one turns it
    into sentences rather than being handed sentences.

    A closed slot writes exactly like an open one. Its value set is deliberately **not** listed --
    see `_for_prompt` for why the prompt may not carry one.
    """
    prop: dict[str, Any] = {"type": "string"}
    description = slot.get("description") or ""
    if description:
        prop["description"] = description
    return prop


def _tool_mcp(name: str, action: dict[str, Any], description: str,
              value_sets: dict[str, list[int]], id_to_str: dict) -> dict[str, Any]:
    """Return the action's MCP definition, preferring action["mcp"].

    Reconstruct from router parameters only for legacy single-file schemas. Free
    argument routing does not retain the full semantic parameter structure.
    """
    declared = action.get("mcp")
    if declared:
        return _for_prompt(declared)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for slot in action.get("params", []) or []:
        pname = slot["name"]
        properties[pname] = _param_property(slot, value_sets, id_to_str)
        required.append(pname)
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            # Every declared slot is required: the router walks the slots in order and there is no
            # path that skips one, so an "optional" parameter would be a lie about the state machine.
            "required": required,
        },
    }


def _singular(set_name: str) -> str:
    """Singularize regular value-set names for individual head-row names.

    Handle -ies and trailing -s, preserving -ss. Collection names in surface YAML
    remain plural; only member names such as object.alarmclock use this form.
    """
    name = str(set_name)
    if name.endswith("ies") and len(name) > 3:
        return name[:-3] + "y"
    if name.endswith("s") and not name.endswith("ss"):
        return name[:-1]
    return name


def describe_action_config(action_config: dict[str, Any]) -> list[ActionDoc]:
    """Return one ActionDoc per extended id in action_ids order.

    This order must match action-head rows. Tool rows read MCP definitions, and
    closed-value rows read the semantics of their owning value set.
    """
    action_ids = list(action_config.get("action_ids", []))
    id_to_str = _int_keyed(action_config.get("id_to_str", {}))
    id_to_description = _int_keyed(action_config.get("id_to_description", {}))
    actions = action_config.get("actions", {}) or {}
    value_sets = action_config.get("value_sets", {}) or {}

    value_owner: dict[int, str] = {}
    for set_name, ids in value_sets.items():
        for i in ids or []:
            # The allocating set owns the row; unions only reference that same extended id.
            value_owner.setdefault(int(i), set_name)
    # Record the first referencing action and parameter for each value set.
    set_referrer: dict[str, tuple[str, str]] = {}
    for name, action in actions.items():
        for slot in action.get("params", []) or []:
            set_name = slot.get("value_set")
            if set_name:
                set_referrer.setdefault(set_name, (name, slot["name"]))

    name_ids = {int(a["name_id"]): n for n, a in actions.items() if "name_id" in a}

    docs: list[ActionDoc] = []
    for raw_id in action_ids:
        aid = int(raw_id)
        description = id_to_description.get(aid, "") or ""
        if aid in name_ids:
            name = name_ids[aid]
            docs.append(ActionDoc(
                action_id=aid, kind="tool", name=name, description=description,
                mcp=_tool_mcp(name, actions[name], description, value_sets, id_to_str),
                owner=name,
            ))
        elif aid in value_owner:
            set_name = value_owner[aid]
            token = str(id_to_str.get(aid, aid)).strip()
            owner, param = set_referrer.get(set_name, ("", ""))
            docs.append(ActionDoc(
                # Use a singular set prefix to identify a member without renaming the collection.
                action_id=aid, kind="value", name=f"{_singular(set_name)}.{token}",
                description=description.strip() or token,
                # A shared value belongs to its set, independent of any one action that references it.
                mcp={},
                owner=owner, value_set=set_name,
                value_choices=tuple(
                    str(id_to_str.get(int(i), i)).strip()
                    for i in (value_sets.get(set_name) or [])
                ),
            ))
        else:
            # Reaching here means an extended id was allocated by a path this module does not know
            # about. Failing loudly beats emitting a doc-less row: a silently missing row would make
            # the encoder produce garbage for that action and nothing would say so.
            raise ValueError(
                f"extended id {aid} ({id_to_str.get(aid)!r}) is neither an action name "
                "nor a closed value -- action_descriptions cannot describe it"
            )
    return docs


def _int_keyed(mapping: dict) -> dict[int, Any]:
    """Round-tripping an action_config through json turns int keys into strings."""
    out: dict[int, Any] = {}
    for key, value in (mapping or {}).items():
        try:
            out[int(key)] = value
        except (TypeError, ValueError):
            continue
    return out
