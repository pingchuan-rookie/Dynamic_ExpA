"""The Alignment dataset pipeline: config -> catalogues -> case plan -> context/reasoning -> samples.

The order is the strategy document's, and section 1.6 is emphatic about one part of it: the visible
MCP subset is fixed **before** the strong LLM writes anything. Every function here is written so
that order cannot be inverted by accident -- `build_context_reasoning` reads a case plan off disk
and has no way to change it.

Three places where the document leaves a choice and this module makes one. Each is recorded here
rather than buried, because each is a thing a reader might otherwise assume was forced.

**Prompt B is called once per tool, not once per catalogue.** `nl.yaml` is specified as
"natural-language action descriptions derived from mcp.yaml" -- one entry per action. Calling
Prompt B on a whole domain would mean splitting a free-form response back into ten blocks, and, worse,
the same action would be worded differently in different visible subsets. Section 1.9 requires
`cat(e)` and `def(a)` to be the same representation; if `def(move_to)` and the `move_to` paragraph
inside `cat(e)` are separately generated strings, they are the same representation only by luck.
A single-tool catalogue is a valid instance of Prompt B's input and removes both problems.

**Entry-marker configurations are assigned cyclically over a shuffled permutation of all 24.**
The document says to sample the rule and the marker independently. Independent uniform sampling of
1050 cases leaves each of the 24 combinations present with overwhelming probability but not
certainty, and "which marker did that run actually cover" is not a question worth answering from a
seed. A shuffled cycle is uniform in the same sense and exhaustive by construction.

**"Do not mention actions outside the catalogue" is enforced on the reasoning, not the context.**
Half of these action names are ordinary English words. "The refrigerator is open" in a context is
not a reference to the `open` tool, and rejecting it would burn attempts on cases that are fine --
Prompt C's own requirement 4 concedes that action words "naturally occur in ordinary language". In
the reasoning the same word does create real ambiguity about the label, so there it is rejected.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from agent_system.policies.dyad.data.actenc_alignment_parquet import SCHEMA_VERSION, SPLIT_POLICY, SPLITS
from experiments.action_encoder_alignment_dataset import actenc_alignment_generation_config as cfg
from experiments.action_encoder_alignment_dataset import actenc_alignment_prompt_rendering as sf
from experiments.action_encoder_alignment_dataset.actenc_alignment_balanced_sampling import build_balanced_mcp_subsets
from experiments.action_encoder_alignment_dataset.actenc_alignment_holdout_split import (
    DEFAULT_SPLIT_SEED,
    split_metadata,
    split_unseen_holdout,
)
from experiments.action_encoder_alignment_dataset.actenc_alignment_parquet_writer import write_dataset
from experiments.action_encoder_alignment_dataset.actenc_alignment_teacher_client import (
    StrongLlm,
    StrongLlmConfig,
    ValidationError,
    extract_json,
    run_parallel,
    temperature_for,
)

MCP_TOOL_FIELDS = ("name", "description", "inputSchema")
PROPERTY_FIELDS = ("type", "description")

#: Two contexts inside one domain whose token sets overlap this much are the "trivial paraphrase"
#: Prompt C requirement 12 forbids. High on purpose: the threshold is there to catch a model that
#: repeated itself, not to police two cases that share a domain vocabulary.
CONTEXT_JACCARD_LIMIT = 0.9


def log(message: str) -> None:
    print(f"[alignment] {message}", flush=True)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def mentions(text: str, name: str) -> bool:
    """Whole-word occurrence of an action name. `lock` must not match inside `unlock`."""
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text) is not None


def argument_covered(text: str, argument: str) -> bool:
    """Did the natural-language conversion keep this argument (Prompt B requirement 9)?

    Two rules, because argument names come in two shapes and one criterion cannot serve both.

    A single-token name (`object`, `destination`, `container`) is an ordinary English word, and the
    document's own worked example uses it verbatim -- "It takes the destination location as input."
    So the literal token is required, and its absence really is a dropped argument.

    A compound name (`object_id`, `target_location`) has no natural English form. Demanding the
    literal `object_id` would reject "the identifier of the object to move", which preserves the
    argument perfectly well; the model would then be re-sampled until it wrote a snake_case token
    into prose, which is not what the prompt asks for. So a compound is covered when its own
    content tokens are all present. Tokens shorter than three characters are dropped: `id` matches
    nothing in "identifier" and everything in a hundred other words.
    """
    if mentions(text, argument):
        return True
    tokens = [t for t in argument.split("_") if len(t) >= 3]
    if len(argument.split("_")) == 1 or not tokens:
        return False
    if mentions(text, argument.replace("_", " ")):
        return True
    return all(mentions(text, token) for token in tokens)



def make_llm(generation: dict[str, Any]) -> StrongLlm:
    return StrongLlm(StrongLlmConfig.from_generation(generation), cache_dir=cfg.cache_dir())


# --------------------------------------------------------------------------- Prompt A


def _validate_tool(tool: Any, expected_names: set[str]) -> dict[str, Any]:
    if not isinstance(tool, dict):
        raise ValidationError(f"tool is {type(tool).__name__}, expected an object")
    extra = sorted(set(tool) - set(MCP_TOOL_FIELDS))
    if extra:
        raise ValidationError(
            f"tool {tool.get('name')!r} has extra top-level fields {extra}; Prompt A requirement 12 "
            "forbids title / outputSchema / annotations / execution / icons"
        )
    missing = [f for f in MCP_TOOL_FIELDS if f not in tool]
    if missing:
        raise ValidationError(f"tool {tool.get('name')!r} is missing {missing}")
    name = tool["name"]
    if name not in expected_names:
        raise ValidationError(f"{name!r} is not one of the requested action names")
    description = " ".join(str(tool["description"]).split())
    if not description:
        raise ValidationError(f"{name!r} has an empty description")
    if description.strip().lower() == name.lower():
        raise ValidationError(f"{name!r}'s description just repeats its name")

    schema = tool["inputSchema"]
    if not isinstance(schema, dict):
        raise ValidationError(f"{name!r} inputSchema is not an object")
    if schema.get("type") != "object":
        raise ValidationError(f"{name!r} inputSchema.type is {schema.get('type')!r}, expected object")
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise ValidationError(f"{name!r} has no inputSchema.properties")
    clean_properties: dict[str, Any] = {}
    for prop_name, prop in properties.items():
        if not isinstance(prop, dict):
            raise ValidationError(f"{name}.{prop_name} is not an object")
        prop_extra = sorted(set(prop) - set(PROPERTY_FIELDS))
        if prop_extra:
            raise ValidationError(
                f"{name}.{prop_name} has extra fields {prop_extra}; Prompt A requirement 6 asks for "
                "type and description only"
            )
        prop_missing = [f for f in PROPERTY_FIELDS if f not in prop]
        if prop_missing:
            raise ValidationError(f"{name}.{prop_name} is missing {prop_missing}")
        prop_description = " ".join(str(prop["description"]).split())
        if not prop_description:
            raise ValidationError(f"{name}.{prop_name} has an empty description")
        clean_properties[str(prop_name)] = {
            "type": str(prop["type"]),
            "description": prop_description,
        }
    required = schema.get("required")
    if not isinstance(required, list):
        raise ValidationError(f"{name!r} inputSchema.required is not a list")
    unknown = [r for r in required if r not in clean_properties]
    if unknown:
        raise ValidationError(f"{name!r} requires {unknown}, which are not properties")

    return {
        "name": str(name),
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": clean_properties,
            "required": [str(r) for r in required],
        },
    }


def _prompt_a_validator(domain: str, action_names: list[str]) -> Callable[[str], list[dict[str, Any]]]:
    expected = list(action_names)
    expected_set = set(expected)

    def validate(text: str) -> list[dict[str, Any]]:
        payload = extract_json(text)
        if not isinstance(payload, dict) or "tools" not in payload:
            raise ValidationError("response has no 'tools' key")
        tools = payload["tools"]
        if not isinstance(tools, list):
            raise ValidationError("'tools' is not a list")
        cleaned = [_validate_tool(tool, expected_set) for tool in tools]
        got = [t["name"] for t in cleaned]
        if sorted(got) != sorted(expected):
            missing = sorted(expected_set - set(got))
            added = sorted(set(got) - expected_set)
            raise ValidationError(
                f"{domain}: tool names do not match the request; missing={missing} added={added}"
            )
        if len(set(got)) != len(got):
            raise ValidationError(f"{domain}: duplicate tool names in the response")
        descriptions = [t["description"].lower() for t in cleaned]
        if len(set(descriptions)) != len(descriptions):
            raise ValidationError(
                f"{domain}: two tools share a description; Prompt A requirement 9 asks for "
                "semantically distinguishable actions"
            )
        by_name = {t["name"]: t for t in cleaned}
        return [by_name[name] for name in expected]

    return validate


def build_mcp_catalogue(llm: StrongLlm, generation: dict[str, Any], *, unseen: bool) -> Path:
    """Prompt A over each domain. Writes `intermediate/mcp[_unseen].yaml`."""
    template = cfg.load_prompt("a")
    names_by_domain = cfg.load_action_names(unseen=unseen)
    temperature = temperature_for(generation, "a")

    def worker(domain: str) -> list[dict[str, Any]]:
        actions = names_by_domain[domain]
        prompt = (
            template
            .replace("{domain}", domain)
            .replace("{action_names}", "\n".join(actions))
        )
        return llm.generate(
            prompt,
            _prompt_a_validator(domain, actions),
            temperature,
            label=f"prompt_a[{domain}]",
        )

    domains = list(cfg.DOMAIN_ORDER)
    results = run_parallel(domains, worker, llm.config.concurrency,
                           on_done=lambda i, n: log(f"prompt A {i}/{n}"),
                           describe=lambda d: f"prompt_a[{d}]")
    tools = [tool for group in results for tool in group]
    path = cfg.catalogue_path("mcp", unseen=unseen)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _catalogue_header(unseen, "Prompt A") + yaml.safe_dump(
            {"tools": tools}, sort_keys=False, allow_unicode=True, width=100000, default_flow_style=False
        ),
        encoding="utf-8",
    )
    log(f"wrote {len(tools)} MCP tools to {path}")
    return path


def _catalogue_header(unseen: bool, prompt: str) -> str:
    pool = "unseen evaluation" if unseen else "training"
    return (
        f"# Generated by {prompt} of experiments/dyad_training/action_encoder_alignment -- the canonical {pool} action catalogue.\n"
        "# Do not hand-edit: every downstream product derives from this file, and an edit here that\n"
        "# is not reflected in data/actenc_alignment/intermediate/manifest.yaml makes the shipped dataset unattributable.\n"
    )


# --------------------------------------------------------------------------- Prompt B


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_DECORATION = "*`_ "


def normalise_nl(body: str, name: str) -> str:
    """Strip the artefacts of asking for a "catalogue" and getting a document back.

    Prompt B is called with a single-tool catalogue (see the module docstring for why), and the
    model reliably answers with a document: a title line, then a heading for the tool, then the
    prose. Measured across the 70 tools, the title line took three different forms -- `## Action
    Catalogue`, `# Action Catalogue`, `# Natural-Language Action Catalogue`. Left in, that variation
    becomes part of the natural-language action description, and a sample's catalogue would carry a
    per-tool document title repeated once for every visible action.

    Re-sampling does not fix it: at temperature 0.2, 0.6 and 1.0, across three tools, every single
    response opened with `## Action Catalogue`. It is what the prompt asks for, read literally.

    So the fix is deterministic and content-preserving rather than another retry:

      - a leading heading whose text does not name this tool is the document title -- dropped;
      - the tool's own heading loses its `#` markers and becomes a plain first line, which is the
        shape section 1.4's worked example uses ("place_in places an object inside a container.");
      - nothing else is touched. No word is added, removed or reordered.
    """
    lines = [line.rstrip() for line in body.strip().split("\n")]
    out: list[str] = []
    for line in lines:
        match = _HEADING.match(line.strip())
        if match is None:
            out.append(line)
            continue
        title = match.group(2).strip().strip(_DECORATION)
        if not any(l.strip() for l in out) and not mentions(title, name):
            continue
        out.append(title if mentions(title, name) else line)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    if out:
        head = out[0].strip()
        undecorated = head.strip(_DECORATION)
        if undecorated.startswith(name) and not head.startswith(name):
            out[0] = head.replace("*", "").replace("`", "").strip()
    return "\n".join(out)


def _prompt_b_validator(tool: dict[str, Any], other_names: list[str]) -> Callable[[str], str]:
    name = tool["name"]
    arguments = list(tool["inputSchema"]["properties"])

    def validate(text: str) -> str:
        body = text.strip()
        if body.startswith("```"):
            lines = body.split("\n")
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            body = "\n".join(lines[1:]).strip()
        body = normalise_nl(body, name)
        if not body:
            raise ValidationError(f"{name}: empty natural-language description")
        if not body.split("\n")[0].startswith(name):
            raise ValidationError(
                f"{name}: after normalisation the first line is {body.split(chr(10))[0][:60]!r}; "
                "section 1.4's natural-language form opens with the tool name, and a catalogue "
                "whose blocks do not is one a reader cannot attribute to an action"
            )
        if not mentions(body, name):
            raise ValidationError(f"{name}: the tool name does not appear (Prompt B requirement 1)")
        missing = [a for a in arguments if not argument_covered(body, a)]
        if missing:
            raise ValidationError(
                f"{name}: arguments {missing} are not mentioned (Prompt B requirement 9)"
            )
        for marker in ("inputSchema", "properties:", "required:", "type: object"):
            if marker in body:
                raise ValidationError(
                    f"{name}: the response still carries MCP structure ({marker!r}); it should be prose"
                )
        # A second tool sneaking in is what would corrupt a concatenated catalogue, and a tool block
        # in this form starts at the beginning of a line with the tool's name.
        for line in body.split("\n"):
            head = line.strip()
            for other in other_names:
                if head.startswith(other + " ") or head.startswith(other + ","):
                    raise ValidationError(
                        f"{name}: a line begins a block for {other!r}; Prompt B requirement 5 "
                        "forbids adding actions"
                    )
        if len(body) > 1200:
            raise ValidationError(f"{name}: {len(body)} characters is not a tool description")
        return body

    return validate


def build_nl_catalogue(llm: StrongLlm, generation: dict[str, Any], *, unseen: bool) -> Path:
    """Prompt B over each tool. Writes `intermediate/nl[_unseen].yaml`."""
    template = cfg.load_prompt("b")
    mcp = cfg.load_mcp_catalogue(unseen=unseen)
    all_names = sorted(set(cfg.load_mcp_catalogue()) | set(cfg.load_mcp_catalogue(unseen=True)))
    temperature = temperature_for(generation, "b")

    def worker(name: str) -> tuple[str, str]:
        tool = mcp[name]
        prompt = template.replace(
            "{visible_mcp}", sf.mcp_action_set_text([name], mcp)
        )
        others = [n for n in all_names if n != name]
        text = llm.generate(
            prompt,
            _prompt_b_validator(tool, others),
            temperature,
            label=f"prompt_b[{name}]",
        )
        return name, text

    names = list(mcp)
    results = run_parallel(names, worker, llm.config.concurrency,
                           on_done=lambda i, n: log(f"prompt B {i}/{n}"),
                           describe=lambda n: f"prompt_b[{n}]")
    table = {name: text for name, text in results}
    duplicates = _duplicate_values(table)
    if duplicates:
        raise RuntimeError(
            f"two tools got the same natural-language description: {duplicates}. The two would be "
            "indistinguishable to the encoder, and the label would be arbitrary."
        )
    path = cfg.catalogue_path("nl", unseen=unseen)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _catalogue_header(unseen, "Prompt B")
        + yaml.safe_dump({"tools": table}, sort_keys=False, allow_unicode=True,
                         width=100000, default_style="|"),
        encoding="utf-8",
    )
    log(f"wrote {len(table)} natural-language descriptions to {path}")
    return path


def _duplicate_values(table: dict[str, str]) -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    out = []
    for name, text in table.items():
        key = " ".join(text.lower().split())
        if key in seen:
            out.append((seen[key], name))
        seen[key] = name
    return out


# --------------------------------------------------------------------------- the case plan


def build_case_plan(generation: dict[str, Any], *, unseen: bool) -> Path:
    """Draw the visible catalogue and action-entry marker. Every visible action is a candidate."""
    rules, markers = cfg.load_entry_markers()
    combos = [(r, m) for r in range(len(rules)) for m in range(len(markers))]
    seen_names = cfg.load_action_names()
    unseen_names = cfg.load_action_names(unseen=True)
    seed = int(generation["seed"])

    if unseen:
        n_per_size = int(generation["unseen"]["n_per_size"])
    else:
        n_per_size = int(generation["n_per_size"])

    rows: list[dict[str, Any]] = []
    for domain_index, domain in enumerate(cfg.DOMAIN_ORDER):
        if unseen:
            targets = unseen_names[domain]
            pool = unseen_names[domain] + seen_names[domain]
        else:
            targets = seen_names[domain]
            pool = seen_names[domain]
        # A distinct stream per domain, derived from the one configured seed. Sharing one stream
        # would make every domain's draw depend on how many cases the previous domain needed, so
        # changing n_per_size for one pool would silently re-roll all five.
        domain_seed = seed * 1000 + domain_index + (500 if unseen else 0)
        plan = build_balanced_mcp_subsets(
            pool,
            n_per_size,
            min_actions=int(generation["min_actions"]),
            max_actions=int(generation["max_actions"]),
            seed=domain_seed,
            targets=targets,
            distractor_pool=pool,
        )
        marker_rng = random.Random(domain_seed + 7)
        cycle = list(combos)
        marker_rng.shuffle(cycle)

        for i, row in enumerate(plan):
            rule_id, marker_id = cycle[i % len(cycle)]
            visible = list(row["action_set"])
            target = row["target_action"]
            # No second draw: the displayed catalogue is the candidate set, in its order.
            suffix = "_unseen" if unseen else ""
            rows.append({
                "case_id": f"{domain}{suffix}_{i + 1:06d}",
                "domain": domain,
                "mcp_size": row["mcp_size"],
                "target_action": target,
                "action_set": visible,
                "entry_marker_rule_id": rule_id,
                "entry_marker_id": marker_id,
            })

    path = cfg.intermediate_dir() / ("case_plan_unseen.jsonl" if unseen else "case_plan.jsonl")
    count = cfg.write_jsonl(path, rows)
    log(f"wrote {count} cases to {path}")
    return path


def validate_case_plan(case: dict[str, Any]) -> None:
    """Reject legacy case plans before rendering prompts or requesting generated text."""
    legacy = [field for field in ("visible_actions", "admissible_actions", "action_catalogue") if field in case]
    if legacy:
        raise ValueError(
            f"{case.get('case_id', '?')}: legacy case plan has fields {legacy}; "
            "Alignment uses only action_set. Legacy plans are unsupported; "
            "use current case plans in a separate data root, leaving historical data unchanged."
        )
    if "action_set" not in case:
        raise ValueError(f"{case.get('case_id', '?')}: case plan is missing action_set")


# --------------------------------------------------------------------------- Prompt C


def _prompt_c_validator(case: dict[str, Any], mcp: dict[str, Any], entry_marker: str,
                        domain_actions: list[str]) -> Callable[[str], dict[str, Any]]:
    validate_case_plan(case)
    visible = set(case["action_set"])
    forbidden = [a for a in domain_actions if a not in visible]
    target_description = " ".join(mcp[case["target_action"]]["description"].lower().split())

    def validate(text: str) -> dict[str, Any]:
        payload = extract_json(text)
        if not isinstance(payload, dict):
            raise ValidationError("response is not a json object")
        missing = [k for k in ("context", "reasoning") if k not in payload]
        if missing:
            raise ValidationError(f"response is missing {missing}")
        context = " ".join(str(payload["context"]).split())
        reasoning = " ".join(str(payload["reasoning"]).split())
        if not context:
            raise ValidationError("empty context")
        if not reasoning:
            raise ValidationError("empty reasoning")
        for field, value in (("context", context), ("reasoning", reasoning)):
            if entry_marker in value:
                raise ValidationError(
                    f"{field} contains the entry marker {entry_marker!r}; Prompt C requirements 10 "
                    "and 11 leave the marker to the builder"
                )
        if target_description and target_description in reasoning.lower():
            raise ValidationError(
                "reasoning copies the target's description verbatim (Prompt C requirement 5)"
            )
        named = [a for a in forbidden if mentions(reasoning, a)]
        row = {"context": context, "reasoning": reasoning, "unavailable_actions": named}
        if named:
            # Recoverable: carry the value so `best_effort` can keep the least-bad attempt, and
            # rank by how many actions were named so a cleaner sample always wins.
            raise ValidationError(
                f"reasoning names {named}, which are not in this case's catalogue "
                "(Prompt C requirement 6)",
                payload=row,
                penalty=len(named),
            )
        return row

    return validate


def _prompt_c_text(case: dict[str, Any], template: str, mcp: dict[str, Any],
                   rules: list[str], markers: list[str]) -> str:
    validate_case_plan(case)
    formatted_rule = sf.format_entry_marker_rule(
        rules[case["entry_marker_rule_id"]], markers[case["entry_marker_id"]]
    )
    return (
        template
        .replace("{domain}", case["domain"])
        .replace("{visible_mcp}", sf.mcp_action_set_text(case["action_set"], mcp))
        .replace("{target_action}", case["target_action"])
        .replace("{rendered_entry_marker_rule}", formatted_rule)
    )


def build_context_reasoning(llm: StrongLlm, generation: dict[str, Any], *, unseen: bool) -> Path:
    """Prompt C over every case in the plan, then a repair pass over near-duplicate contexts."""
    template = cfg.load_prompt("c")
    mcp, _ = cfg.load_both_catalogues()
    rules, markers = cfg.load_entry_markers()
    plan_path = cfg.intermediate_dir() / ("case_plan_unseen.jsonl" if unseen else "case_plan.jsonl")
    cases = cfg.read_jsonl(plan_path)
    for case in cases:
        validate_case_plan(case)
    temperature = temperature_for(generation, "c")
    seen_names = cfg.load_action_names()
    unseen_names = cfg.load_action_names(unseen=True)

    def domain_actions(case: dict[str, Any]) -> list[str]:
        """The actions this case's own pool can offer.

        Scoped to the split, not to the domain name. A seen case draws its ten admissible actions from
        `actenc_alignment_action_names.yaml` alone -- `lift` is not an action it could ever show, so a reasoning that
        says "lift the tray" is using an English verb, not naming an unavailable tool. Including the
        held-out pool here was the first version, and it rejected 61 cases over one word.
        """
        if unseen:
            return unseen_names[case["domain"]] + seen_names[case["domain"]]
        return seen_names[case["domain"]]

    def worker(job: tuple[dict[str, Any], int]) -> dict[str, Any]:
        case, offset = job
        prompt = _prompt_c_text(case, template, mcp, rules, markers)
        result = llm.generate(
            prompt,
            _prompt_c_validator(case, mcp, markers[case["entry_marker_id"]], domain_actions(case)),
            temperature,
            label=f"prompt_c[{case['case_id']}]",
            attempt_offset=offset,
            best_effort=True,
        )
        return {"case_id": case["case_id"], **result}

    jobs = [(case, 0) for case in cases]
    rows = run_parallel(jobs, worker, llm.config.concurrency,
                        on_done=lambda i, n: log(f"prompt C {i}/{n}") if i % 25 == 0 else None,
                        describe=lambda job: job[0]["case_id"])

    by_id = {row["case_id"]: row for row in rows}
    case_by_id = {case["case_id"]: case for case in cases}
    for repair_round in range(1, 5):
        offenders = _near_duplicate_contexts(cases, by_id)
        if not offenders:
            break
        log(f"repair round {repair_round}: regenerating {len(offenders)} duplicated contexts")
        jobs = [(case_by_id[case_id], llm.config.max_attempts * repair_round) for case_id in offenders]
        for row in run_parallel(jobs, worker, llm.config.concurrency,
                                describe=lambda job: job[0]["case_id"]):
            by_id[row["case_id"]] = row
    offenders = _near_duplicate_contexts(cases, by_id)
    if offenders:
        raise RuntimeError(
            f"{len(offenders)} contexts are still near-duplicates of another case in their domain "
            f"after four repair rounds, e.g. {offenders[:5]}. Prompt C requirement 12 asks for a "
            "new semantic case each time; a dataset of paraphrases inflates every metric."
        )

    path = cfg.intermediate_dir() / (
        "context_reasoning_unseen.jsonl" if unseen else "context_reasoning.jsonl"
    )
    count = cfg.write_jsonl(path, [by_id[case["case_id"]] for case in cases])
    violated = [r for r in by_id.values() if r.get("unavailable_actions")]
    log(f"wrote {count} context/reasoning rows to {path}")
    log(f"requirement 6 fallbacks: {len(violated)}/{count} "
        f"({100.0 * len(violated) / max(1, count):.2f}%)")
    return path


def _near_duplicate_contexts(cases: list[dict[str, Any]], by_id: dict[str, dict[str, str]]) -> list[str]:
    """case_ids whose context repeats an earlier one in the same domain.

    Only the later member of each pair is returned: regenerating both would be twice the work for
    the same outcome, and it would churn cases that are not the problem.
    """
    offenders: list[str] = []
    per_domain: dict[str, list[tuple[str, set[str]]]] = {}
    for case in cases:
        row = by_id.get(case["case_id"])
        if row is None:
            continue
        tokens = _words(row["context"])
        bucket = per_domain.setdefault(case["domain"], [])
        clash = False
        for _, other in bucket:
            union = tokens | other
            if union and len(tokens & other) / len(union) >= CONTEXT_JACCARD_LIMIT:
                clash = True
                break
        if clash:
            offenders.append(case["case_id"])
        else:
            bucket.append((case["case_id"], tokens))
    return offenders


# --------------------------------------------------------------------------- the builder


def build_sample(
    case: dict[str, Any],
    text: dict[str, str],
    form: str,
    mcp: dict[str, Any],
    nl: dict[str, str],
    rules: list[str],
    markers: list[str],
    format_example: str,
    encoder_instruction: str,
) -> dict[str, Any]:
    """One base case plus one action description format -> one Alignment SFT sample (section 1.10's schema).

    `format_example` and `encoder_instruction` are both required. They were optional while each was
    being evaluated against its absence; both won, the absent variants were deleted, and leaving the
    defaults in place would keep two retired prompt shapes reachable from a config typo.
    """
    if not format_example.strip():
        raise ValueError("build_sample needs a non-empty format_example")
    if not encoder_instruction.strip():
        raise ValueError("build_sample needs a non-empty encoder_instruction")
    validate_case_plan(case)
    rule = rules[case["entry_marker_rule_id"]]
    marker = markers[case["entry_marker_id"]]
    formatted_rule = sf.format_entry_marker_rule(rule, marker)
    visible = list(case["action_set"])
    # The example's action is drawn uniformly from cat(e), from a stream keyed on the case id so
    # both forms show the same action, despite their different catalogue and call syntax.
    # Uniform, not "anything but the label": excluding the label would eliminate an admissible
    # action for free and lift the chance baseline from 1/k to 1/(k-1). Uniform keeps the example
    # independent of the answer, which is what makes it safe to put in every prompt.
    stream = random.Random(int(hashlib.sha256(case["case_id"].encode()).hexdigest()[:12], 16))
    example_action = visible[stream.randrange(len(visible))]
    catalogue = sf.action_set_text(form, visible, mcp, nl)
    return {
        "case_id": case["case_id"],
        "domain": case["domain"],
        "entry_marker_rule": rule,
        "entry_marker": marker,
        "action_set_form": form,
        "action_set": visible,
        "label": case["target_action"],
        "context": text["context"],
        "reasoning": text["reasoning"],
        "action_family": case["domain"],
        "mcp_size": case["mcp_size"],
        "format_example_action": example_action,
        "policy_lm_prompt": sf.policy_lm_prompt(
            formatted_rule, catalogue, text["context"], text["reasoning"], marker,
            format_example=(format_example.replace("{entry_marker}", marker)
                            .replace("{action_json}", json.dumps(example_action, ensure_ascii=False))
                            .replace("{action}", example_action)),
            form=form,
        ),
        # Store the definition separately from the encoder input and its appended instruction.
        "action_definitions": {
            action: sf.definition_text(form, action, mcp, nl) for action in visible
        },
        "action_encoder_prompts": {
            action: sf.action_encoder_prompt(
                form, catalogue, sf.definition_text(form, action, mcp, nl),
                instruction=encoder_instruction.replace("{action}", action),
            )
            for action in visible
        },
    }


def _require_current_prompt_version(generation: dict[str, Any]) -> None:
    """Reject old data before a pipeline step can change its upstream products."""
    if cfg.dataset_path().exists():
        manifest_path = cfg.manifest_path()
        previous = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if isinstance(previous, dict) and previous.get("split_policy") != SPLIT_POLICY:
            raise ValueError(
                f"existing dataset uses an older split policy; expected {SPLIT_POLICY}. "
                "Historical data cannot be overwritten; use a separate DYAD_ALIGNMENT_DATA root."
            )
        if not isinstance(previous, dict) or previous.get("policy_prompt_version") != sf.POLICY_PROMPT_VERSION:
            raise ValueError(
                f"existing dataset has an older policy prompt version; expected {sf.POLICY_PROMPT_VERSION}. "
                "Historical data cannot be overwritten; use a separate DYAD_ALIGNMENT_DATA root."
            )
        if previous.get("holdout_split") != split_metadata(generation.get("split_seed", DEFAULT_SPLIT_SEED)):
            raise ValueError("existing holdout assignment is immutable; use a separate DYAD_ALIGNMENT_DATA root")


def build_dataset(generation: dict[str, Any]) -> dict[str, int]:
    """Join existing inputs, splitting unseen cases into paired validation/test holdouts."""
    _require_current_prompt_version(generation)
    mcp, nl = cfg.load_both_catalogues()
    rules_by_form, markers, examples = cfg.policy_prompt_inputs(generation)
    format_example = examples[cfg.FORM_NL]
    encoder_instruction = str((generation.get("encoder_prompt") or {}).get("instruction", "") or "")
    # Both are required. A missing key reads as "" here, which used to mean "build the variant
    # without it"; that variant is gone, so the same "" now means a config that lost a key and
    # would otherwise ship a dataset in a shape nothing else in the pipeline expects.
    missing = [name for name, value in
               (("policy_prompt.format_example", format_example),
                ("encoder_prompt.instruction", encoder_instruction)) if not value.strip()]
    if missing:
        raise ValueError(
            f"actenc_alignment_generation.yaml is missing {', '.join(missing)}. Both are required: the prompt "
            "variants that omitted them no longer exist."
        )
    log(f"policy prompts carry a format example: {format_example!r}")
    log(f"encoder prompts carry an instruction: {encoder_instruction!r}")

    def rows_for(unseen: bool) -> list[dict[str, Any]]:
        suffix = "_unseen" if unseen else ""
        cases = cfg.read_jsonl(cfg.intermediate_dir() / f"case_plan{suffix}.jsonl")
        for case in cases:
            validate_case_plan(case)
        texts = {
            row["case_id"]: row
            for row in cfg.read_jsonl(cfg.intermediate_dir() / f"context_reasoning{suffix}.jsonl")
        }
        missing = [c["case_id"] for c in cases if c["case_id"] not in texts]
        if missing:
            raise RuntimeError(
                f"{len(missing)} cases have no context/reasoning, e.g. {missing[:5]}. The join is "
                "by case_id and must be total; a partial join would silently shrink the dataset."
            )
        out = []
        for case in cases:
            for form in cfg.FORMS:
                out.append(build_sample(case, texts[case["case_id"]], form, mcp, nl,
                                        rules_by_form[form], markers, examples[form], encoder_instruction))
        return out

    main = rows_for(False)
    unseen_rows = rows_for(True)

    rows = split_unseen_holdout(
        [{**row, "split": "train"} for row in main]
        + [{**row, "split": "test"} for row in unseen_rows],
        seed=generation.get("split_seed", DEFAULT_SPLIT_SEED),
    )
    counts = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    if cfg.dataset_path().exists():
        from agent_system.policies.dyad.data.actenc_alignment_parquet import read_dataset
        previous = read_dataset(cfg.dataset_path())
        keyed = {(r["case_id"], r["action_set_form"]): r for r in rows}
        old_keys = [(r["case_id"], r["action_set_form"]) for r in previous]
        if set(old_keys) == set(keyed):
            rows = [keyed[key] for key in old_keys]
    write_dataset(cfg.dataset_path(), rows)
    log(f"written {counts} to {cfg.dataset_path()}")
    return counts


# --------------------------------------------------------------------------- manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:32]


def write_manifest(generation: dict[str, Any], counts: dict[str, int]) -> Path:
    mcp = cfg.load_mcp_catalogue()
    mcp_unseen = cfg.load_mcp_catalogue(unseen=True)
    manifest = {
        "generated_by": "experiments/action_encoder_alignment_dataset/actenc_alignment_generate_dataset.py",
        "dataset_schema_version": SCHEMA_VERSION,
        "policy_prompt_version": sf.POLICY_PROMPT_VERSION,
        "storage_format": "parquet",
        "split_column": "split",
        "split_policy": SPLIT_POLICY,
        "holdout_split": split_metadata(generation.get("split_seed", DEFAULT_SPLIT_SEED)),
        "historical_usage": "New split assignment does not establish that source cases were historically unused.",
        "candidate_policy": "visible_actions_are_candidates",
        "strong_llm": {
            "model": (generation.get("strong_llm") or {}).get("model"),
            "temperature": (generation.get("strong_llm") or {}).get("temperature"),
        },
        "generation": {
            "n_per_size": int(generation["n_per_size"]),
            "unseen_n_per_size": int(generation["unseen"]["n_per_size"]),
            "min_actions": int(generation["min_actions"]),
            "max_actions": int(generation["max_actions"]),
            "seed": int(generation["seed"]),
            "policy_prompt": generation.get("policy_prompt"),
            "encoder_prompt": generation.get("encoder_prompt"),
        },
        "inputs": {
            name: _sha256(path)
            for name, path in {
                "config/action_names.yaml": cfg.CONFIG_DIR / "actenc_alignment_action_names.yaml",
                "config/unseen_action_names.yaml": cfg.CONFIG_DIR / "actenc_alignment_unseen_action_names.yaml",
                "config/entry_marker.yaml": cfg.CONFIG_DIR / "actenc_alignment_entry_marker.yaml",
                "config/generation.yaml": cfg.CONFIG_DIR / "actenc_alignment_generation.yaml",
                "prompts/prompt_a.txt": cfg.PROMPT_DIR / "actenc_alignment_prompt_a.txt",
                "prompts/prompt_b.txt": cfg.PROMPT_DIR / "actenc_alignment_prompt_b.txt",
                "prompts/prompt_c.txt": cfg.PROMPT_DIR / "actenc_alignment_prompt_c.txt",
                "intermediate/mcp.yaml": cfg.catalogue_path("mcp"),
                "intermediate/nl.yaml": cfg.catalogue_path("nl"),
                "intermediate/mcp_unseen.yaml": cfg.catalogue_path("mcp", unseen=True),
                "intermediate/nl_unseen.yaml": cfg.catalogue_path("nl", unseen=True),
            }.items()
        },
        "products": {
            str(path.relative_to(cfg.data_root())): _sha256(path)
            for path in [
                *(cfg.intermediate_dir() / name for name in (
                    "case_plan.jsonl", "case_plan_unseen.jsonl",
                    "context_reasoning.jsonl", "context_reasoning_unseen.jsonl",
                )),
                cfg.dataset_path(),
            ]
        },
        "actions": {"seen": len(mcp), "unseen": len(mcp_unseen)},
        "counts": dict(counts),
        "expected": {
            "n_base": 5 * len(cfg.MCP_SIZES) * int(generation["n_per_size"]),
            "n_sft": 2 * 5 * len(cfg.MCP_SIZES) * int(generation["n_per_size"]),
        },
    }
    path = cfg.manifest_path()
    if path.exists():
        previous = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(previous, dict):
            for key in ("prompt_revision", "split_revision", "holdout_migration", "historical_usage"):
                if key in previous:
                    manifest[key] = previous[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log(f"wrote {path}")
    return path


# --------------------------------------------------------------------------- driver


STEPS = ("catalogues-mcp", "catalogues-nl", "case-plan", "context-reasoning", "dataset")


def run(steps: list[str], *, generation: Optional[dict[str, Any]] = None) -> None:
    generation = generation or cfg.load_generation()
    _require_current_prompt_version(generation)
    needs_llm = any(step.startswith("catalogues") or step == "context-reasoning" for step in steps)
    llm = make_llm(generation) if needs_llm else None
    for step in steps:
        log(f"=== {step} ===")
        if step == "catalogues-mcp":
            build_mcp_catalogue(llm, generation, unseen=False)
            build_mcp_catalogue(llm, generation, unseen=True)
        elif step == "catalogues-nl":
            build_nl_catalogue(llm, generation, unseen=False)
            build_nl_catalogue(llm, generation, unseen=True)
        elif step == "case-plan":
            build_case_plan(generation, unseen=False)
            build_case_plan(generation, unseen=True)
        elif step == "context-reasoning":
            build_context_reasoning(llm, generation, unseen=False)
            build_context_reasoning(llm, generation, unseen=True)
        elif step == "dataset":
            counts = build_dataset(generation)
            write_manifest(generation, counts)
            from experiments.action_encoder_alignment_dataset.actenc_alignment_gen_sample_records import (
                main as generate_samples,
            )
            if generate_samples([]) != 0:
                raise ValueError("failed to generate sample documentation")
        else:
            raise ValueError(f"unknown step {step!r}; known: {list(STEPS)}")
    if llm is not None:
        log(f"strong LLM: {llm.calls} live calls, {llm.cache_hits} cache hits")


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", action="append", choices=list(STEPS),
                        help="run one step; repeatable. Default: every step, in order.")
    args = parser.parse_args(argv)
    run(args.step or list(STEPS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
