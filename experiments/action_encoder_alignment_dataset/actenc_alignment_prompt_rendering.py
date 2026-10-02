"""The two action description formats, and the prompts assembled from them (strategy document sections 1.4 and 1.9).

One base case becomes two samples with format-specific action descriptions and call syntax.
The candidate order, target, context and reasoning stay paired. Action-description construction
is a pure function of (form, visible actions, catalogues), without per-sample state.

Two rules the assembly enforces, both from section 1.9:

  1. `prompt(a) = action_set_text + def(a)`: every displayed action participates in
     scoring, and every encoder input carries the full displayed action set.
  2. `cat(e)` and `def(a)` use the same representation. A tool-specification catalogue followed by
     a natural-language definition is never produced; the form is chosen once, at the top.

The MCP text is hand-written rather than emitted by `yaml.dump`. `dump` sorts keys, re-wraps long
scalars and quotes according to its own rules, none of which the strategy document's examples do,
and all of which would make the shipped prompts drift from the document the day the yaml library is
upgraded.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.action_encoder_alignment_dataset.actenc_alignment_generation_config import FORM_MCP, FORM_NL

#: What separates `cat(e)` from `def(a)` in each form. Fixed strings, because the dataset gate
#: rebuilds every encoder input from its parts and compares byte for byte.
MCP_DEFINITION_SEPARATOR = "---"
NL_DEFINITION_HEADER = "Target action:"

_NEEDS_QUOTES = (":", "#", "'", '"', "{", "}", "[", "]", ",", "&", "*", "!", "|", ">", "%", "@", "`")


def _scalar(value: Any) -> str:
    """A yaml scalar that round-trips, quoted only when it has to be.

    The strategy document's examples are unquoted, so quoting everything would make the shipped
    prompts look nothing like the document. Quoting nothing would break the moment a description
    contains a colon.
    """
    text = str(value)
    if text == "":
        return '""'
    if text != text.strip() or any(token in text for token in _NEEDS_QUOTES) or "\n" in text:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{escaped}"'
    return text


def mcp_tool_lines(tool: Mapping[str, Any]) -> list[str]:
    """One MCP Tool object as yaml lines at indent 0, in the document's field order."""
    schema = tool.get("inputSchema") or {}
    properties = schema.get("properties") or {}
    required = list(schema.get("required") or [])

    lines = [
        f"name: {_scalar(tool['name'])}",
        f"description: {_scalar(tool['description'])}",
        "inputSchema:",
        f"  type: {_scalar(schema.get('type', 'object'))}",
        "  properties:",
    ]
    for prop_name, prop in properties.items():
        lines.append(f"    {_scalar(prop_name)}:")
        lines.append(f"      type: {_scalar(prop.get('type', 'string'))}")
        lines.append(f"      description: {_scalar(prop.get('description', ''))}")
    lines.append("  required:")
    for name in required:
        lines.append(f"    - {_scalar(name)}")
    return lines


def mcp_tool_text(tool: Mapping[str, Any]) -> str:
    return "\n".join(mcp_tool_lines(tool))


def mcp_action_set_text(names: Sequence[str], mcp: Mapping[str, Mapping[str, Any]]) -> str:
    """`tools:` followed by one list item per visible action, blank-line separated."""
    blocks = []
    for name in names:
        lines = mcp_tool_lines(mcp[name])
        block = ["  - " + lines[0]] + ["    " + line for line in lines[1:]]
        blocks.append("\n".join(block))
    return "tools:\n" + "\n\n".join(blocks)


def nl_action_set_text(names: Sequence[str], nl: Mapping[str, str]) -> str:
    return "\n\n".join(nl[name].strip() for name in names)


def action_set_text(
    form: str,
    names: Sequence[str],
    mcp: Mapping[str, Mapping[str, Any]],
    nl: Mapping[str, str],
) -> str:
    if form == FORM_MCP:
        return mcp_action_set_text(names, mcp)
    if form == FORM_NL:
        return nl_action_set_text(names, nl)
    raise ValueError(f"unknown action_set_form {form!r}; known: {FORM_MCP}, {FORM_NL}")


def definition_text(
    form: str,
    name: str,
    mcp: Mapping[str, Mapping[str, Any]],
    nl: Mapping[str, str],
) -> str:
    if form == FORM_MCP:
        return mcp_tool_text(mcp[name])
    if form == FORM_NL:
        return nl[name].strip()
    raise ValueError(f"unknown action_set_form {form!r}; known: {FORM_MCP}, {FORM_NL}")


def action_encoder_prompt(form: str, catalogue: str, definition: str,
                          instruction: str) -> str:
    """`cat(e) + def(a)`, joined by the separator this form uses, then the instruction.

    The instruction goes **last**, and the position is the point. The encoder is a causal model, so
    the final positions are the ones that have attended to everything before them; without it the
    tool-specification prompt ends on `required:\n  - object_id`, which is about the least
    informative tail available. Ending on a sentence that names the action puts the action's own
    token where the context is richest.

    A sentence that is identical for every admissible action would be close to inert either way: it
    adds the same direction to every `w_a`, and a shared additive component shifts every logit in the
    softmax over `C_t` by the same amount and cancels. What carries signal is the part that varies
    per action -- the action name. So the instruction is expected to contain it, and
    check_action_encoder_alignment_dataset.py F11b/F11c hold it to varying in *only* that: an instruction that named
    a second admissible action would leak cross-action information into a prompt that is supposed to
    be about one action.

    **Required, not optional.** It used to default to `""`, which built a prompt ending at `def(a)`
    -- a second prompt shape that nothing in the schema distinguished from this one. Two shapes
    behind one field name is how a dataset ends up half one thing and half the other with every
    gate passing.
    """
    if not instruction.strip():
        raise ValueError(
            "action_encoder_prompt needs a non-empty instruction. The empty case used to build a "
            "prompt ending at def(a); that shape no longer exists, so an empty string here means "
            "the caller read a config key that is missing rather than one that is switched off."
        )
    if form == FORM_MCP:
        body = f"{catalogue}\n\n{MCP_DEFINITION_SEPARATOR}\n{definition}"
    elif form == FORM_NL:
        body = f"{catalogue}\n\n{NL_DEFINITION_HEADER}\n{definition}"
    else:
        raise ValueError(f"unknown action_set_form {form!r}; known: {FORM_MCP}, {FORM_NL}")
    return f"{body}\n\n{instruction.strip()}"


def format_entry_marker_rule(rule: str, entry_marker: str) -> str:
    """Substitute the marker into the rule. The `{entry_marker}` placeholder must be there.

    A rule that lost its placeholder writes to a sentence that names no marker at all, and the
    sample still looks well-formed: the marker is appended to the history either way, so nothing
    downstream notices that the instruction stopped saying which string to write after.
    """
    if "{entry_marker}" not in rule:
        raise ValueError(f"entry-marker rule has no {{entry_marker}} placeholder: {rule!r}")
    return rule.replace("{entry_marker}", entry_marker)


POLICY_PROMPT_VERSION = "mcp-json-name-v1"


def action_decision_prefix(form: str, entry_marker: str) -> str:
    """Fixed input syntax before the selected action, never the action itself."""
    if form == FORM_MCP:
        return entry_marker + '{"name": "'
    if form == FORM_NL:
        return entry_marker
    raise ValueError(f"unknown action_set_form {form!r}")


def policy_lm_prompt(
    formatted_rule: str,
    catalogue: str,
    context: str,
    reasoning: str,
    entry_marker: str,
    format_example: str,
    *,
    form: str,
) -> str:
    """Policy input through the fixed action-decision prefix, without the target action.

    The rule, example, catalogue, context and reasoning retain their existing order.
    The last token is the scoring position: MCP now stops at the opening quote of the
    JSON name value, while natural language still stops at the marker.
    This input is deliberately incomplete JSON; Alignment selects an action embedding
    rather than generating the action name or arguments token by token.

    The offline reconstruction and pipeline tests protect two constraints:

      - the example demonstrates a legal action name: it comes from this sample's own
        `cat(e)`, drawn **uniformly**. Uniformly is the whole point. Never showing the label would
        be a leak in the negative direction -- it eliminates one admissible action, lifting chance
        from 1/k to 1/(k-1) -- and always showing it is the obvious leak. Uniform makes the example
        statistically independent of the label, so no strategy that reads it beats chance;
      - it renders with **this sample's** marker, not a hardcoded one. There are 24 marker
        configurations, and an example that always says `tool:` teaches the wrong string in 23.

    **Required, not optional.** It used to default to `""`, which produced a prompt with no example
    and one fewer blank line -- a second prompt shape sharing this one's field name, distinguishable
    only by counting newlines in the shipped file.
    """
    if not format_example.strip():
        raise ValueError(
            "policy_lm_prompt needs a non-empty format_example. The empty case used to build a "
            "prompt without an example; that shape no longer exists, so an empty string here means "
            "the caller read a config key that is missing rather than one that is switched off."
        )
    # The blank-line layout is load-bearing only in that it must not change: `runs/` holds finished
    # runs on the byte-exact strings this produces, and a tidier prompt would silently make them
    # incomparable to anything built afterwards.
    example = f"{format_example.strip()}\n\n"
    return (
        f"{formatted_rule.strip()}\n"
        f"{example}"
        f"\n{catalogue.strip()}\n\n"
        f"{context.strip()}\n"
        f"{reasoning.strip()}\n"
        f"{action_decision_prefix(form, entry_marker)}"
    )
