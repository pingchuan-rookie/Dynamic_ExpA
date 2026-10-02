#!/usr/bin/env python3
# Copyright 2025 ExpA_verl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Decode one run into token-aligned trajectories.

Views: steps (default), conversation, aligned, turns. All views use the same
trajectory representation. Alignment failures remain visible; tokens and masks
are never padded or truncated to make a broken trace look valid.

Decoded text is verbatim. Token provenance and special-token status are separate
columns. Extended action ids are rendered as action labels, since they have no
vocabulary text.

Usage (cwd=Dynamic_ExpA):
  python experiments/shared/analysis/run/decode_trajectories.py RUN_DIR --show 3
  python experiments/shared/analysis/run/decode_trajectories.py RUN_DIR --view aligned
  python experiments/shared/analysis/run/decode_trajectories.py RUN_DIR --out

RUN_DIR defaults to the newest diagnostic run under the project's
outputs/<site>/agentic_rl/. The tokenizer comes from the run metadata or --model.
Each invocation writes <run>/analysis/decode_trajectories.md; --out selects all
trajectories instead of the default sample. Verifiers in this directory share
resolve_dir and save_report with this module.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator



# Reports are written beside their source run under <run>/analysis/.
# Stable Markdown names are overwritten on rerun and follow the run's archival policy.
# save_report mirrors console output to that run-local report.


ANALYSIS_SUBDIR = "analysis"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def analysis_dir(debug_dir) -> str:
    """Return <debug_dir>/analysis/ and make sure it exists.

    debug_dir is the run directory the analysis reads from (outputs/<algo>/<env>/<run>), so the
    report lands inside the run rather than in a separate tree keyed by a flattened copy of the
    same path. Flat, with no sub-directories: one analysis, one file, one place to look.
    """
    d = os.path.join(str(debug_dir).rstrip(os.sep), ANALYSIS_SUBDIR)
    os.makedirs(d, exist_ok=True)
    return d


def results_path(name: str, debug_dir, ext: str = "md") -> str:
    """Return <debug_dir>/analysis/<name>.<ext>.

    Deliberately without a timestamp: the name states what the file holds, so running the same
    analysis twice replaces it. Anything that needs to distinguish two runs should differ in
    `name` (view, filter, trajectory id), because that is what a reader actually searches by.
    """
    return os.path.join(analysis_dir(debug_dir), f"{name}.{ext}")


class _Tee:
    """Write to console and file at once; the file side has ANSI colour codes stripped."""

    def __init__(self, console, fh):
        self._console = console
        self._fh = fh

    def write(self, s: str) -> int:
        self._console.write(s)
        self._fh.write(_ANSI.sub("", s))
        return len(s)

    def flush(self) -> None:
        self._console.flush()
        self._fh.flush()


@contextlib.contextmanager
def save_report(name: str, debug_dir):
    """Mirror stdout into <debug_dir>/analysis/<name>.md (ANSI stripped).

    Pass the analysed run directory, not a derived group name: the report belongs to that run.
    """
    path = results_path(name, debug_dir)
    old = sys.stdout
    fh = open(path, "w", encoding="utf-8")
    sys.stdout = _Tee(old, fh)
    try:
        yield path
    finally:
        sys.stdout = old
        fh.close()
        print(f"[saved] analysis result -> {os.path.relpath(path)}")


# Shared JSONL reader for offline analysis. Probe event component names instead of
# assuming a particular environment pool name.


# Agent-side components, in the order decode should prefer them when a directory holds both.
AGENT_COMPONENTS = ("dyad_tool_agent", "grpo_tool_agent")
ENV_POOL_SUFFIX = "_env_pool"


@dataclass
class Events:
    """All events of one debug directory, indexed the few ways the views actually need."""

    debug_dir: str
    records: list[dict[str, Any]] = field(default_factory=list)
    components: dict[str, int] = field(default_factory=dict)
    env_pool_component: str | None = None
    # component -> {(pid, dump_seq)}: the contract's way of counting what actually landed (§4).
    dump_pairs: dict[str, set[tuple[int, int]]] = field(default_factory=dict)

    def of(self, component: str, event: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r.get("component") == component and r.get("event") == event]

    def agent_component(self) -> str | None:
        for name in AGENT_COMPONENTS:
            if self.components.get(name):
                return name
        return None

    def summaries(self) -> list[dict[str, Any]]:
        """trajectory_summary events across both agent components, time ordered."""
        return [
            r for r in self.records
            if r.get("event") == "trajectory_summary" and r.get("component") in AGENT_COMPONENTS
        ]

    def by_request(self, event: str) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for r in self.records:
            if r.get("event") != event:
                continue
            rid = r.get("request_id")
            if rid is None:
                continue
            out.setdefault(str(rid), []).append(r)
        return out

    def env_binds(self) -> dict[str, list[dict[str, Any]]]:
        """env_step_bind keyed by session_id. The agent loop passes request_id as session_id, so
        this keys the same way as by_request() -- that is what makes the two-source action check
        possible at all."""
        out: dict[str, list[dict[str, Any]]] = {}
        if not self.env_pool_component:
            return out
        for r in self.of(self.env_pool_component, "env_step_bind"):
            out.setdefault(str(r.get("session_id")), []).append(r)
        return out

    def run_meta(self) -> dict[str, Any]:
        metas = [r for r in self.records if r.get("event") == "run_meta"]
        return metas[-1] if metas else {}

    def dump_counts(self) -> dict[str, tuple[int, int]]:
        """component -> (unique (pid, dump_seq) pairs, number of dumping processes). Contract §4."""
        return {comp: (len(pairs), len({p for p, _ in pairs})) for comp, pairs in self.dump_pairs.items()}


def _iter_lines(path: str) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def component_of(path: str) -> str:
    """`calc_env_pool_pid1234.jsonl` -> `calc_env_pool`."""
    return os.path.basename(path).rsplit("_pid", 1)[0]


def load(debug_dir: str) -> Events:
    """Read every `*_pid*.jsonl` under debug_dir, merge, sort by time, index."""
    ev = Events(debug_dir=debug_dir.rstrip("/"))
    for path in sorted(glob.glob(os.path.join(ev.debug_dir, "*_pid*.jsonl"))):
        comp = component_of(path)
        for rec in _iter_lines(path):
            rec.setdefault("component", comp)
            ev.records.append(rec)
    ev.records.sort(key=lambda r: r.get("time", 0.0))

    for rec in ev.records:
        comp = str(rec.get("component"))
        ev.components[comp] = ev.components.get(comp, 0) + 1
        if comp.endswith(ENV_POOL_SUFFIX) and ev.env_pool_component is None:
            ev.env_pool_component = comp
        seq, pid = rec.get("dump_seq"), rec.get("pid")
        if seq is not None and pid is not None:
            ev.dump_pairs.setdefault(comp, set()).add((int(pid), int(seq)))
    return ev


PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))
from agent_system.utils.artifact_paths import artifact_root

OUTPUTS = artifact_root(PROJECT) / "outputs"


def resolve_dir(arg: str | None) -> str:
    """Resolve an explicit directory or the newest diagnostic run under outputs/<site>/agentic_rl/."""
    if arg:
        directory = Path(arg).expanduser().resolve()
        if not directory.is_dir():
            raise SystemExit(f"run directory does not exist: {directory}; pass an existing run directory")
        return str(directory)

    candidates = set()
    for stage in OUTPUTS.glob("*/agentic_rl"):
        for diagnostic in stage.rglob("*_pid*.jsonl"):
            # Generated analysis artifacts are not source runs.
            if "analysis" not in diagnostic.relative_to(stage).parts:
                candidates.add(diagnostic.parent)
    if candidates:
        return str(max(candidates, key=lambda directory: (directory.stat().st_mtime_ns, str(directory))).resolve())
    raise SystemExit(
        f"no diagnostic run found under {OUTPUTS}/<site>/agentic_rl/; pass a run directory "
        "containing diagnostic JSONL files explicitly"
    )


# Convert events to the Traj representation shared by all views.
# Infer mode from contract fields, never directory names. Alignment is a four-state
# verdict: report MISALIGNED with ? columns rather than padding or truncating evidence.
# Raw sampled IDs omit force-written tokens; only policy_trace defines their mapping
# to full context. Decisions are displayed in a separate column.


OK = "OK"
PADDED = "PADDED"
MISALIGNED = "MISALIGNED"
ABSENT = "ABSENT"

PROMPT, GEN, ENV = "PROMPT", "GEN", "ENV"

# Mirrors agent_system.utils.diagnostics.TRUNCATION_KEY. Duplicated rather than imported on purpose: this
# layer reads jsonl files and nothing else, so the key is part of the file format it parses, the
# same way "allowed_strs" is. Importing dyad would make offline analysis need the training package.
TRUNCATION_KEY = "__truncated_items__"


def strip_truncation(items: Any) -> tuple[list, int]:
    """`(values, dropped)` for a dumped list that may end in the truncation sentinel.

    The diagnostics layer caps lists at `DYAD_DIAG_MAX_VALUES` (64 in the debug scripts) and
    appends `{"__truncated_items__": N}` **as a list element**, so the sentinel occupies a slot
    where a value is expected. Consumers that do not skip it either crash (`int()` on a dict) or
    quietly count it as one more action. Only the closed variant reaches the cap today -- its
    `examine` slot offers 83 admissible values -- which is why this went unnoticed until that run.
    """
    if not isinstance(items, (list, tuple)):
        return ([], 0)
    values, dropped = [], 0
    for item in items:
        if isinstance(item, dict) and TRUNCATION_KEY in item:
            dropped += int(item[TRUNCATION_KEY])
            continue
        values.append(item)
    return (values, dropped)



def parse_ids(value: Any) -> list[int]:
    """Contract stores id sequences as comma strings (to dodge the diagnostics list truncation);
    older dumps used plain lists, and both `null` and `""` mean "no sequence"."""
    if value is None:
        return []
    if isinstance(value, str):
        return [int(x) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


@dataclass
class Align:
    status: str
    resp_start: int | None = None       # index into token_ids where the response segment begins
    mask: list[int] = field(default_factory=list)   # trimmed to positions that own a token
    pad_positions: int = 0              # trailing mask positions with no token (PADDED only)
    note: str = ""


def classify_align(n_tokens: int, resp_mask: list[int], num_response_tokens: int | None) -> Align:
    """The four states of ../dynamic-expa-design/progress/tasks/done_W8_decode_layer.md §B. Never repairs, never guesses."""
    m = len(resp_mask)
    if m == 0:
        return Align(ABSENT, note="no response_mask in the dump")
    r = int(num_response_tokens) if num_response_tokens is not None else sum(resp_mask)
    if m <= n_tokens and sum(resp_mask) == r:
        return Align(OK, resp_start=n_tokens - m, mask=list(resp_mask))
    if m > n_tokens:
        ones = r
        clean_block = resp_mask[:ones] == [1] * ones and set(resp_mask[ones:]) <= {0}
        if clean_block and ones <= n_tokens:
            return Align(PADDED, resp_start=n_tokens - ones, mask=list(resp_mask[:ones]),
                         pad_positions=m - ones)
        if not clean_block:
            return Align(MISALIGNED, note=f"mask({m}) wider than tokens({n_tokens}) and not r ones + zeros")
        return Align(MISALIGNED, note=f"num_response_tokens({r}) exceeds token count({n_tokens})")
    return Align(MISALIGNED, note=f"sum(mask)={sum(resp_mask)} != num_response_tokens={r} with mask({m}) <= tokens({n_tokens})")


@dataclass
class Seg:
    kind: str
    start: int
    end: int
    turn: int | None = None


def segments(align: Align, n_tokens: int, assistant_turns: int | None = None) -> list[Seg]:
    """Split token_ids into PROMPT / GEN / ENV runs and number the GEN runs by turn.

    The turn column goes `None` (shown as `?`) whenever the GEN run count disagrees with
    `assistant_turns`: forcing a numbering there would hide exactly the bug worth finding.
    """
    if align.resp_start is None:
        return [Seg(PROMPT, 0, n_tokens)] if n_tokens else []
    segs: list[Seg] = []
    if align.resp_start > 0:
        segs.append(Seg(PROMPT, 0, align.resp_start))
    pos = align.resp_start
    gen_index = 0
    i = 0
    while i < len(align.mask):
        bit = align.mask[i]
        j = i
        while j < len(align.mask) and align.mask[j] == bit:
            j += 1
        seg = Seg(GEN if bit else ENV, pos + i, pos + j)
        if bit:
            seg.turn = gen_index
            gen_index += 1
        else:
            seg.turn = gen_index - 1 if gen_index else None
        segs.append(seg)
        i = j
    if assistant_turns is not None and gen_index != int(assistant_turns):
        for seg in segs:
            seg.turn = None
    return segs


@dataclass
class Turn:
    index: int
    gen_text: str | None = None
    span: tuple[int, int] | None = None      # in response_mask coordinates
    decisions: list[dict[str, Any]] = field(default_factory=list)
    raw_decisions: list[int] = field(default_factory=list)   # this turn's Dyad decision tokens
    decisions_dropped: int = 0               # decision tokens the diagnostics cap left out
    action_config: dict[str, Any] = field(default_factory=dict)
    triggered_dyad: bool = False             # the enter marker fired, so the head was consulted
    action_sent: str | None = None           # agent side (env_step)
    action_in: str | None = None             # env side (<env>_env_pool.env_step_bind)
    observation: str | None = None
    obs_truncated: bool = False
    turn_score: float | None = None
    reward: float | None = None
    done: bool | None = None
    invalid_action: bool = False
    fallback: bool = False                   # missing_action_content_fallback fired this turn

    @property
    def action_mismatch(self) -> bool:
        """Both sources present and not byte-identical. The only handle on "the action string was
        rewritten in transit"; nothing else in the repo checks it."""
        return self.action_sent is not None and self.action_in is not None and self.action_sent != self.action_in


@dataclass
class Traj:
    mode: str
    meta: dict[str, Any]
    token_ids: list[int]
    resp_mask: list[int]
    align: Align
    segments: list[Seg]
    turns: list[Turn]
    decisions: list[int] = field(default_factory=list)       # raw_token_ids (Dyad sampled decisions)
    decision_mask: list[int] = field(default_factory=list)   # 1 = action head, 0 = vocabulary
    recorded_spans: list[dict[str, Any]] = field(default_factory=list)
    recorded_spans_dropped: int = 0          # turn_spans the diagnostics cap left out
    full_text: str | None = None
    gen_turns_match: bool | None = None
    span_agreement: str | None = None        # recorded turn_spans vs mask-derived segments
    bind_gap: int = 0                        # env_step reaching the env minus binds; non-zero blocks pairing

    @property
    def request_id(self) -> str:
        return str(self.meta.get("request_id") or "")

    @property
    def n_gen_segments(self) -> int:
        return len([s for s in self.segments if s.kind == GEN])


_META_KEYS = (
    "request_id", "validate", "dump_seq", "pid", "termination_reason", "sum_turn_scores",
    "won", "env_won", "assistant_turns", "length_truncated", "response_len", "prompt_len",
    "num_response_tokens", "seq_len", "missing_action_fallback_count", "response_length_limit",
)


def detect_mode(summary: dict[str, Any], turns: list["Turn"] | None = None) -> str:
    """dyad / react / std, from contract fields only.

    `raw_token_ids` empty has two recorded shapes -- `null` and `""` -- and both mean "this
    trajectory made no action-head decision", not "field missing" (contract §2 footnote).

    Env interaction is read from the assembled turns, not just `summary["turns"]`: dumps taken
    before `turns[].env_action` was filled in on the react path carry the same evidence in their
    `env_step` events, and those are contract fields too. What is banned is deciding by directory
    or file name -- an dyad run analysed out of a grpo_debug/ folder is still dyad.
    """
    if parse_ids(summary.get("raw_token_ids")):
        return "dyad"
    # The contract's §1 path table: `dyad_tool_agent` is emitted by the Dyad loop and nothing else,
    # whereas `grpo_tool_agent` is shared by react and std. Checked before the evidence rules below
    # because an Dyad trajectory that terminated on `no_tool_call` has neither decisions nor env
    # steps -- without this it would come out labelled `std`, which is simply false.
    if str(summary.get("component") or "") == "dyad_tool_agent":
        return "dyad"
    for turn in summary.get("turns") or []:
        if (turn or {}).get("env_action") is not None:
            return "react"
    for turn in turns or []:
        if turn.action_sent is not None or turn.action_in is not None:
            return "react"
    # A react trajectory can end without ever reaching the env (the model emitted no parsable
    # action), leaving no env evidence at all. `single_turn` is std by construction; every other
    # reason is only ever set by a multi-turn loop.
    if str(summary.get("termination_reason") or "") not in ("", "single_turn"):
        return "react"
    return "std"


def _turn_score_of(summary: dict[str, Any], index: int) -> float | None:
    turns = summary.get("turns") or []
    if index < len(turns):
        return (turns[index] or {}).get("turn_score")
    scores = summary.get("turn_scores") or []
    return scores[index] if index < len(scores) else None


def build_traj(
    summary: dict[str, Any],
    gens: list[dict[str, Any]] | None = None,
    steps: list[dict[str, Any]] | None = None,
    binds: list[dict[str, Any]] | None = None,
    fallbacks: int = 0,
) -> Traj:
    token_ids = parse_ids(summary.get("full_token_ids"))
    resp_mask = parse_ids(summary.get("response_mask"))
    assistant_turns = summary.get("assistant_turns")
    align = classify_align(len(token_ids), resp_mask, summary.get("num_response_tokens"))
    segs = segments(align, len(token_ids), assistant_turns)
    n_gen = len([s for s in segs if s.kind == GEN])
    gen_match = None if assistant_turns is None else (n_gen == int(assistant_turns))

    meta = {k: summary.get(k) for k in _META_KEYS}
    meta["component"] = summary.get("component")
    meta["fallbacks"] = fallbacks or summary.get("missing_action_fallback_count") or 0

    built_turns = _build_turns(summary, gens or [], steps or [], binds or [])
    # `turn_spans` is a dumped list, so the diagnostics cap applies to it like any other: past
    # DYAD_DIAG_MAX_VALUES entries it holds `{"__truncated_items__": N}` **as an element**. Reading it
    # raw made `_compare_spans` do `int(sentinel["start"])` -> KeyError, which killed decode_trajectories with
    # exit 1 on any run long enough to cross the cap. Measured on codegym / 48 tool turns: 12 of 16
    # trajectories, because one turn contributes two spans (GEN + ENV) and the cap is 64.
    recorded_spans, spans_dropped = strip_truncation(summary.get("turn_spans"))
    traj = Traj(
        mode=detect_mode(summary, built_turns),
        meta=meta,
        token_ids=token_ids,
        resp_mask=resp_mask,
        align=align,
        segments=segs,
        turns=built_turns,
        decisions=parse_ids(summary.get("raw_token_ids")),
        decision_mask=parse_ids(summary.get("decision_mask")),
        recorded_spans=recorded_spans,
        recorded_spans_dropped=spans_dropped,
        full_text=summary.get("full_text"),
        gen_turns_match=gen_match,
        bind_gap=len([s for s in (steps or []) if not s.get("invalid_action")]) - len(binds or []),
    )
    traj.span_agreement = _compare_spans(traj)
    return traj


def _build_turns(
    summary: dict[str, Any],
    gens: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    binds: list[dict[str, Any]],
) -> list[Turn]:
    """One Turn per assistant generation, with the env interaction that followed it stitched on.

    Deliberately index based rather than mode based: a std trajectory simply has no gens/steps/binds
    and degrades to a single Turn holding its score. Writing `if mode == "std"` here would split the
    three paths back into three formatters, which is the disease this stage is treating.
    """
    contract_turns = summary.get("turns") or []
    # env_step (agent side) and env_step_bind (env side) share no key, so they can only be paired by
    # order -- and only after accounting for the one systematic gap between them: an action the tool
    # rejects as invalid never reaches the env, so it logs an env_step but no bind. Measured on
    # alfworld: 29 env_step vs 28 binds with exactly 1 invalid_action, and 29 vs 26 with exactly 3.
    # Pairing naively made every later turn look like "the action was rewritten in transit" (17
    # bogus violations). If the counts still disagree after the adjustment, refuse to pair at all.
    reached_env = [s for s in steps if not s.get("invalid_action")]
    bind_for: list[dict[str, Any]] = [{}] * len(steps)
    if binds and len(binds) == len(reached_env):
        remaining = iter(binds)
        bind_for = [{} if s.get("invalid_action") else next(remaining) for s in steps]
    count = max(len(gens), len(steps), len(contract_turns), len(binds))
    turns: list[Turn] = []
    for i in range(count):
        gen = gens[i] if i < len(gens) else {}
        step = steps[i] if i < len(steps) else {}
        bind = bind_for[i] if i < len(bind_for) else {}
        contract = contract_turns[i] if i < len(contract_turns) else {}
        span = None
        if gen.get("span_start") is not None and gen.get("span_end") is not None:
            span = (int(gen["span_start"]), int(gen["span_end"]))
        raw_decisions, dropped, action_config = decision_chunk(gen)
        turns.append(Turn(
            index=i,
            gen_text=gen.get("surface_text"),
            span=span,
            decisions=list(gen.get("dyad_decisions") or []),
            raw_decisions=raw_decisions,
            decisions_dropped=dropped,
            action_config=action_config,
            triggered_dyad=bool(gen.get("has_action_content")),
            action_sent=step.get("action_sent", contract.get("env_action")),
            action_in=bind.get("action_in"),
            # env_step names it observation_text (dyad) / observation (react); the contract's
            # turns[].env_obs is the fallback when only the summary was dumped.
            observation=(step.get("observation_text") or step.get("observation")
                         or contract.get("env_obs")),
            obs_truncated=bool(contract.get("env_obs_truncated", False)),
            turn_score=_turn_score_of(summary, i),
            reward=step.get("reward"),
            done=step.get("done"),
            invalid_action=bool(step.get("invalid_action", False)),
        ))
    return turns


def decision_chunk(gen: dict[str, Any]) -> tuple[list[int], int, dict[str, Any]]:
    """One generation's Dyad decision record: `(decision token ids, dropped, action_config)`.

    `action_content` is keyed by server_request_id, and a turn that never reached the enter marker
    has none at all -- that absence is the evidence for "Dyad was not triggered this turn", so it is
    reported as an empty chunk rather than skipped.

    The ids go through `strip_truncation` because this one **is** a dumped list (unlike the
    trajectory-level `raw_token_ids`, which the contract stores as a comma string precisely to dodge
    the cap). A silently short decision list would misalign every later token in the turn.
    """
    raw: list[int] = []
    dropped = 0
    config: dict[str, Any] = {}
    for chunk in gen.get("action_content") or []:
        for payload in (chunk or {}).values():
            if not isinstance(payload, dict):
                continue
            ids, drop = strip_truncation(payload.get("raw_token_ids"))
            raw += [int(x) for x in ids if isinstance(x, int)]
            dropped += drop
            config = config or (payload.get("action_config") or {})
    return raw, dropped, config


def _compare_spans(traj: Traj) -> str | None:
    """Recorded turn_spans vs the segments derived from the mask -- two independent sources.

    A truncated recording is reported as uncomparable rather than compared. Comparing the surviving
    prefix against the full derived list would report DIFFER for every long trajectory, which is a
    false alarm on exactly the runs where a real disagreement would matter most.
    """
    if not traj.recorded_spans or traj.align.resp_start is None:
        return None
    if traj.recorded_spans_dropped:
        return (f"UNCOMPARABLE: turn_spans hit the diagnostics cap, "
                f"{traj.recorded_spans_dropped} of {len(traj.recorded_spans) + traj.recorded_spans_dropped} "
                f"spans were not dumped (raise DYAD_DIAG_MAX_VALUES to compare)")
    recorded = [(s.get("kind"), int(s["start"]), int(s["end"])) for s in traj.recorded_spans]
    derived = [(s.kind, s.start - traj.align.resp_start, s.end - traj.align.resp_start)
               for s in traj.segments if s.kind in (GEN, ENV)]
    if recorded == derived:
        return f"MATCH ({len(recorded)} spans)"
    return f"DIFFER recorded={recorded} derived={derived}"


def build_all(events, limit: int | None = None) -> list[Traj]:
    """Assemble every dumped trajectory of one debug directory. `events` is an _events.Events."""
    gens = events.by_request("generation_result")
    steps = events.by_request("env_step")
    fallbacks = events.by_request("missing_action_content_fallback")
    binds = events.env_binds()
    out: list[Traj] = []
    for summary in events.summaries():
        if not summary.get("full_token_ids"):
            continue
        rid = str(summary.get("request_id") or "")
        out.append(build_traj(
            summary,
            gens=gens.get(rid, []),
            steps=steps.get(rid, []),
            binds=binds.get(rid, []),
            fallbacks=len(fallbacks.get(rid, [])),
        ))
        if limit and len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------- token provenance


VOCAB, HEAD, FORCED = "vocab", "action head", "forced"


@dataclass
class TokenSource:
    """One generated token and who decided it."""
    token_id: int
    kind: str            # VOCAB | HEAD | FORCED
    note: str = ""


def token_provenance(
    conv_ids: list[int],
    turn: Turn,
    formatter: "TokenFormatter",
) -> tuple[list[TokenSource], str | None]:
    """Label every token of one assistant turn `vocab` / `action head` / `forced`.

    Returns `(rows, None)` when the labelling is *proved*, and `([], reason)` when it is not. There is
    no third outcome on purpose: a partly-guessed provenance table is worse than none, because a
    reader cannot see which rows were guessed.

    ## Why this is a reconstruction and not a lookup

    Contract §3: force-written tokens never enter `raw_token_ids`, so the dump holds two streams of
    different lengths -- what the policy *decided* (`raw_token_ids`) and what was *written into the
    context* (`full_token_ids`). Neither carries the other's indices. Replaying `ActionRouter` would
    give the mapping directly, but the `action_config` in the dump is a diagnostics *view*: its
    nested values arrive as `repr()` strings (`surface_form: "'calc'"`), so a router built from it is
    not the router that ran. Reading the yaml off disk instead is the second-source-of-truth mistake
    `action_id_names` already refuses: the schema may have been edited since the run.

    So the two streams are aligned against each other, using only fields the dump preserves intact,
    and every step is checked rather than assumed:

      - a decision token found in the context      -> `vocab` at that position;
      - tokens skipped on the way to it            -> `forced` (the router wrote them);
      - an extended action id                      -> the context shows the action's `surface_form`
                                                      instead, so those tokens are `action head`,
                                                      and the *tail* of the gap must equal
                                                      `id_to_seq` or the turn is rejected;
      - tokens before the very first decision      -> `vocab`. The decision sequence is packed and
                                                      cleared at every clean boundary, so a turn's
                                                      chunk starts at the enter marker; the thinking
                                                      text before it was sampled from the base
                                                      vocabulary just the same, it is simply in an
                                                      earlier chunk.

    One router rule has to be mirrored here because it makes the two streams disagree by design: when
    the model samples a token that *ends* an open value (`.</` merged by BPE), the router **rewrites**
    it into the exit marker's first token (router.py `_run_after` / the ARGUMENT_VALUE_OPEN branch).
    The decision id is then absent from the context and the exit marker's first token is present
    instead. Measured on gsm8k/base with Qwen2.5-0.5B: 3 of 65 Dyad turns take that path, and without
    this case they were the only 3 that failed to resolve.
    """
    raw = turn.raw_decisions
    if not raw:
        return [], "this turn recorded no Dyad decisions"
    if turn.decisions_dropped:
        return [], (f"{turn.decisions_dropped} decision token(s) beyond the diagnostics cap were not "
                    f"dumped, so the two streams cannot be aligned")
    config = turn.action_config or {}
    id_to_seq = config.get("id_to_seq") or {}
    exit_seq = [int(x) for x in (config.get("exit_surface_form_seq") or [])]
    turn_end = {int(x) for x in (config.get("turn_end_ids") or [])}

    rows: list[TokenSource] = []
    ci = di = dec_i = 0
    while di < len(raw):
        token = raw[di]

        if formatter.is_action(token):
            if dec_i >= len(turn.decisions):
                return [], "more extended action ids than recorded dyad_decisions"
            decision = turn.decisions[dec_i]
            dec_i += 1
            surface_ids = _surface_ids(token, decision, id_to_seq, formatter)
            if not surface_ids:
                return [], f"no surface form recorded for action id {token}"
            # The next decision token pins where this gap ends; without one the gap runs to the end.
            end = len(conv_ids)
            if di + 1 < len(raw):
                end = _find(conv_ids, raw[di + 1], ci)
                if end is None:
                    end = len(conv_ids)
            gap = conv_ids[ci:end]
            if len(gap) < len(surface_ids) or gap[-len(surface_ids):] != surface_ids:
                return [], (f"the surface form of {decision.get('chosen_str')!r} is not at the tail "
                            f"of the tokens written for it")
            chosen = decision.get("chosen_str")
            allowed, _ = strip_truncation(decision.get("allowed_strs"))
            for tid in gap[:-len(surface_ids)]:
                rows.append(TokenSource(tid, FORCED, "enter marker completed by the router"))
            for tid in surface_ids:
                rows.append(TokenSource(tid, HEAD, f"head chose {chosen!r} from {allowed}"))
            ci = end
            di += 1
            continue

        at = _find(conv_ids, token, ci)
        if at is None:
            # The exit rewrite: the sampled terminator is replaced by the exit marker's first token.
            if exit_seq and ci < len(conv_ids) and conv_ids[ci] == exit_seq[0]:
                rows.append(TokenSource(conv_ids[ci], VOCAB,
                                        "value terminator, rewritten into the exit marker"))
                ci += 1
                di += 1
                continue
            return [], f"decision token {token} does not appear in what was written"
        skipped_kind = VOCAB if di == 0 else FORCED
        skipped_note = "sampled before the enter marker" if di == 0 else "written by the router"
        for tid in conv_ids[ci:at]:
            rows.append(TokenSource(tid, skipped_kind, skipped_note))
        rows.append(TokenSource(conv_ids[at], VOCAB))
        ci = at + 1
        di += 1

    for tid in conv_ids[ci:]:
        note = "turn end" if tid in turn_end else "exit marker completed by the router"
        rows.append(TokenSource(tid, FORCED, note))
    return rows, None


def _find(ids: list[int], target: int, start: int) -> int | None:
    for i in range(start, len(ids)):
        if ids[i] == target:
            return i
    return None


def _surface_ids(action_id: int, decision: dict[str, Any], id_to_seq: dict, formatter) -> list[int]:
    """The context tokens an extended action id materialises into.

    `id_to_seq` is what the schema compiled, so it is preferred; the diagnostics layer stringifies
    its keys and values, hence the `str()`/`int()`. Re-tokenising `chosen_surface_form` is the
    fallback for dumps that predate the field -- it agrees on every schema in the repo, but it is a
    second derivation, so it is not the first choice.
    """
    seq = id_to_seq.get(str(action_id)) or id_to_seq.get(action_id)
    if seq:
        return [int(x) for x in seq]
    surface = decision.get("chosen_surface_form")
    if surface and formatter.tok is not None:
        return list(formatter.tok.encode(surface, add_special_tokens=False))
    return []


def action_id_names(events) -> dict[int, str]:
    """local action id -> name, built from this run's own `dyad_decisions`.

    Never from the action yaml: yaml is a second source of truth and may already have been edited
    since the run finished. What the dump recorded is what was actually used.
    """
    names: dict[int, str] = {}
    for rec in events.records:
        if rec.get("event") != "generation_result":
            continue
        for dec in rec.get("dyad_decisions") or []:
            ids, _ = strip_truncation(dec.get("allowed_action_ids"))
            strs, _ = strip_truncation(dec.get("allowed_strs"))
            for aid, text in zip(ids, strs):
                if text is not None:
                    names[int(aid)] = str(text)
    return names


def sample(trajs: list[Traj], how: str, n: int) -> list[Traj]:
    """`first` = dump order. `diverse` = one per (termination_reason, won) stratum first.

    VALIDATION §4 requires ">=3 trajectories with different termination reasons" and forbids only
    sampling successful ones; `first` alone reliably returns three near-identical trajectories.
    """
    if how != "diverse":
        return trajs[:n]
    picked: list[Traj] = []
    seen: set[tuple[Any, Any]] = set()
    for traj in trajs:
        key = (traj.meta.get("termination_reason"), bool(traj.meta.get("won") or traj.meta.get("env_won")))
        if key in seen:
            continue
        seen.add(key)
        picked.append(traj)
        if len(picked) >= n:
            return picked
    for traj in trajs:
        if traj not in picked:
            picked.append(traj)
        if len(picked) >= n:
            break
    return picked


# Decode token text verbatim. Put special-token flags and repr-formatted whitespace
# in aligned-view columns rather than inserting annotations into decoded text.
# Never decode with skip_special_tokens=True.


RAW, ESCAPED = "raw", "escaped"

WARN = "⚠️"
# Extended action ids are the one thing that cannot be decoded: they are outside the tokenizer's
# vocabulary, so there is no text for them to be. The label is their only possible form, not a
# decoration on top of one.
ACT_L, ACT_R = "⟦", "⟧"


class TokenFormatter:
    """Holds everything needed to turn a token id into a string, and nothing else."""

    def __init__(self, tokenizer=None, special_ids: set[int] | None = None,
                 action_names: dict[int, str] | None = None, org_vocab: int = 1 << 62,
                 mode: str = RAW):
        self.tok = tokenizer
        self.org_vocab = int(org_vocab)
        self.mode = mode
        self.action_names = dict(action_names or {})
        self.unknown_action_ids: set[int] = set()
        if special_ids is not None:
            self.special_ids = set(special_ids)
        elif tokenizer is not None:
            self.special_ids = ({int(i) for i in (tokenizer.all_special_ids or [])}
                                | {int(v) for v in (tokenizer.get_added_vocab() or {}).values()})
        else:
            self.special_ids = set()

    # ---- single token -------------------------------------------------------------------

    def is_special(self, tid: int) -> bool:
        return tid in self.special_ids

    def is_action(self, tid: int) -> bool:
        return tid >= self.org_vocab

    def action_label(self, tid: int) -> str:
        local = tid - self.org_vocab
        name = self.action_names.get(local)
        if name is None:
            self.unknown_action_ids.add(local)
        return f"{ACT_L}ACT:{local}={name if name is not None else '?'}{ACT_R}"

    def token_text(self, tid: int) -> str:
        """Raw decoded text of one token (no visualisation). Extended ids are not in the vocab."""
        if self.is_action(tid):
            return self.action_label(tid)
        if self.tok is None:
            return f"<{tid}>"
        try:
            return self.tok.decode([tid], skip_special_tokens=False)
        except Exception:  # noqa: BLE001
            return f"<{tid}>"

    def token_cell(self, tid: int) -> str:
        """The aligned view's token column: always `escaped`, so nothing can hide in it."""
        return repr(self.token_text(tid))

    # ---- sequences ----------------------------------------------------------------------

    def sequence(self, ids: list[int], mode: str | None = None) -> str:
        mode = mode or self.mode
        if mode == ESCAPED:
            return " ".join(self.token_cell(t) for t in ids)
        return self._raw(ids)

    def _raw(self, ids: list[int]) -> str:
        """Verbatim decode; the `redecode == full_text` self-check must run on this and only this."""
        if self.tok is None:
            return "".join(f"<{t}>" for t in ids)
        if not any(self.is_action(t) for t in ids):
            return self.tok.decode(ids, skip_special_tokens=False)
        parts, buf = [], []
        for tid in ids:
            if self.is_action(tid):
                if buf:
                    parts.append(self.tok.decode(buf, skip_special_tokens=False))
                    buf = []
                parts.append(self.action_label(tid))
            else:
                buf.append(tid)
        if buf:
            parts.append(self.tok.decode(buf, skip_special_tokens=False))
        return "".join(parts)

    def _decode(self, ids: list[int]) -> str:
        if self.tok is None:
            return "".join(f"<{t}>" for t in ids)
        try:
            return self.tok.decode(ids, skip_special_tokens=False)
        except Exception:  # noqa: BLE001
            return "".join(f"<{t}>" for t in ids)

    def decision_sequence(self, ids: list[int], mask: list[int]) -> str:
        """The Dyad sampled-decision trace. Action positions come from the recorded decision_mask
        rather than from `tid >= org_vocab`: the mask is what rollout actually did."""
        parts, buf = [], []
        for i, tid in enumerate(ids):
            is_act = bool(mask[i]) if i < len(mask) else self.is_action(tid)
            if is_act:
                if buf:
                    parts.append(self._decode(buf))
                    buf = []
                parts.append(self.action_label(tid))
            else:
                buf.append(tid)
        if buf:
            parts.append(self._decode(buf))
        return "".join(parts)


# ---------------------------------------------------------------------------- header


def header(traj: Traj, formatter: TokenFormatter) -> list[str]:
    """Printed at the top of every view. `align=` is mandatory (W8 §B)."""
    meta = traj.meta
    won = meta.get("won") if meta.get("won") is not None else meta.get("env_won")
    lines = [
        "=" * 100,
        f"[{traj.mode}] req={traj.request_id[:12]} dump_seq={meta.get('dump_seq')} pid={meta.get('pid')} "
        f"validate={meta.get('validate')}",
        f"  align={traj.align.status}"
        + (f" ({traj.align.note})" if traj.align.note else "")
        + f"  tokens={len(traj.token_ids)} mask={len(traj.resp_mask)} "
          f"num_response_tokens={meta.get('num_response_tokens')} prompt_len={meta.get('prompt_len')}",
        f"  turns: GEN segments={traj.n_gen_segments} assistant_turns={meta.get('assistant_turns')} "
        f"match={traj.gen_turns_match}"
        + ("" if traj.gen_turns_match is not False else f"  {WARN} turn column shows ?"),
        f"  reward={meta.get('sum_turn_scores')} won={won} termination={meta.get('termination_reason')!r} "
        f"length_truncated={meta.get('length_truncated')} fallbacks={meta.get('fallbacks')}",
    ]
    if traj.span_agreement:
        lines.append(f"  recorded turn_spans vs mask-derived segments: {traj.span_agreement}")
    if traj.bind_gap:
        lines.append(f"  {WARN} env_step ({traj.bind_gap:+d}) vs <env>_env_pool.env_step_bind: counts "
                     f"disagree, so the two-source action check is skipped for this trajectory "
                     f"(pairing by index would report every later turn as rewritten)")
    if traj.align.status == MISALIGNED:
        lines.append(f"  {WARN} response_mask does not align with full_token_ids -- segment and turn "
                     f"columns are '?'; nothing was truncated or padded to make it fit.")
    if formatter.unknown_action_ids:
        lines.append(f"  {WARN} action ids with no name in this run's dyad_decisions: "
                     f"{sorted(formatter.unknown_action_ids)}")
    return lines


# ---------------------------------------------------------------------------- views


def view_conversation(traj: Traj, formatter: TokenFormatter, mode: str | None = None) -> list[str]:
    lines = ["-" * 100, f"[conversation] format={mode or formatter.mode}"]
    lines.append(formatter.sequence(traj.token_ids, mode))
    if traj.meta.get("length_truncated"):
        lines.append(f"[TRUNCATED] response reached response_length={traj.meta.get('response_length_limit')}")
    return lines


def view_aligned(traj: Traj, formatter: TokenFormatter) -> list[str]:
    """One line per token. The token column is always `escaped` -- see TokenFormatter.token_cell."""
    lines = ["-" * 100, "[aligned] one line per token; token column is repr(), so newlines and "
             "trailing spaces are literal",
             f"{'idx':>6} {'token_id':>9} {'mask':>4} {'seg':<7} {'turn':>4} sp  token"]
    seg_of: dict[int, Any] = {}
    for seg in traj.segments:
        for i in range(seg.start, seg.end):
            seg_of[i] = seg
    misaligned = traj.align.resp_start is None
    start = traj.align.resp_start
    for i, tid in enumerate(traj.token_ids):
        seg = seg_of.get(i)
        if misaligned or seg is None:
            seg_name, turn_col, mask_col = "?", "?", "?"
        else:
            seg_name = seg.kind
            turn_col = "-" if seg.turn is None and seg.kind == PROMPT else (
                "?" if seg.turn is None else str(seg.turn))
            mask_col = "-" if seg.kind == PROMPT else str(traj.align.mask[i - start])
        sp = "*" if formatter.is_special(tid) else (
            "A" if formatter.is_action(tid) else " ")
        lines.append(f"{i:>6} {tid:>9} {mask_col:>4} {seg_name:<7} {turn_col:>4} {sp}   "
                     f"{formatter.token_cell(tid)}")
    if traj.align.pad_positions:
        lines.append(f"(+{traj.align.pad_positions} padding mask positions, no tokens)")
    return lines


def view_turns(traj: Traj, formatter: TokenFormatter) -> list[str]:
    """Per turn: generated text -> action-head decisions -> action sent -> observation -> score.

    The decision column is deliberately separate and deliberately unaligned with the conversation:
    contract §3 says force-written tokens never enter `raw_token_ids`, so no positional mapping
    exists offline. Fabricating one would be worse than having no view at all.
    """
    lines = ["-" * 100, "[turns] the decision column is NOT positionally alignable with the "
             "conversation sequence (contract §3)"]
    if not traj.turns:
        lines.append("  (no turn records: this trajectory never interacted with an env)")
    for turn in traj.turns:
        span = f" span=[{turn.span[0]},{turn.span[1]}) tokens={turn.span[1] - turn.span[0]}" if turn.span else ""
        lines.append(f"  -- turn {turn.index}{span} turn_score={turn.turn_score}")
        if turn.gen_text is not None:
            for row in turn.gen_text.split("\n"):
                lines.append(f"     | {row}")
        for dec in turn.decisions:
            lines.append("     " + decision_line(dec))
        if turn.action_sent is not None or turn.action_in is not None:
            flag = f"  {WARN} action rewritten in transit" if turn.action_mismatch else ""
            lines.append(f"     -> env action_sent={turn.action_sent!r} reward={turn.reward} "
                         f"done={turn.done}{'  invalid_action!' if turn.invalid_action else ''}{flag}")
            if turn.action_in is not None:
                same = "==" if not turn.action_mismatch else "!="
                lines.append(f"        env_step_bind.action_in={turn.action_in!r}  ({same} action_sent)")
        if turn.observation is not None:
            trunc = "  [TRUNCATED]" if turn.obs_truncated else ""
            for row in str(turn.observation).split("\n"):
                lines.append(f"        obs| {row}")
            if trunc:
                lines.append(f"        obs{trunc}")
    lines.append(f"  -- end: termination={traj.meta.get('termination_reason')!r} "
                 f"sum_turn_scores={traj.meta.get('sum_turn_scores')}")
    if traj.decisions:
        n_action = sum(1 for i, _ in enumerate(traj.decisions)
                       if (traj.decision_mask[i] if i < len(traj.decision_mask) else 0))
        lines.append(f"  -- sampled decision sequence: {len(traj.decisions)} decisions "
                     f"(action head={n_action}, vocabulary={len(traj.decisions) - n_action})")
        lines.append("     " + formatter.decision_sequence(traj.decisions, traj.decision_mask))
    return lines


# ---------------------------------------------------------------------------- steps view


CHAT_USER = "<|im_start|>user"
CHAT_ASSISTANT = "<|im_start|>assistant"


def split_system_prompt(text: str) -> tuple[str, str]:
    """`(system block, first user message)`. Falls back to `("", text)` when the split is not found.

    The system prompt is byte-identical across every rollout of a run and is the single largest
    thing in the file -- 550 of the 570 prompt tokens on gsm8k. Printing it once at the top and
    showing only the question per rollout is what makes this file readable end to end; it is not a
    summarisation, because the omitted text is printed verbatim, just in one place.
    """
    at = text.find(CHAT_USER)
    if at < 0:
        return "", text
    return text[:at], text[at:]


def split_turn_header(text: str) -> tuple[str, str]:
    """Move a trailing `<|im_start|>assistant` off the observation and onto the turn it opens.

    The generation prompt sits at the end of the PROMPT/ENV segment because that is where the
    tokenizer put it, but it is the header of the *next* assistant turn, not part of what the
    environment said. Leaving it on the observation makes every `[obs]` end in a line that belongs
    to the block underneath it.
    """
    at = text.rfind(CHAT_ASSISTANT)
    if at < 0 or text[at + len(CHAT_ASSISTANT):].strip():
        return text, ""
    return text[:at], text[at:]


def view_steps(traj: Traj, formatter: TokenFormatter, index: int, total: int) -> list[str]:
    """One block per step: the observation that led into it, the assistant turn, and the provenance.

    A step is one assistant generation. Its `[obs]` is the environment reply that preceded it (the
    question itself for step 1), and its `[assistant]` is the verbatim decode of that turn's GEN
    segment -- taken from `full_token_ids`, not from the per-turn `token_ids`, because the latter is
    a dumped list and the diagnostics cap truncates it at 64 tokens (measured: 41 of 102 turns).
    """
    meta = traj.meta
    won = meta.get("won") if meta.get("won") is not None else meta.get("env_won")
    lines = [
        "=" * 100,
        f"rollout {index + 1}/{total}  req={traj.request_id[:12]}  mode={traj.mode}  "
        f"steps={traj.n_gen_segments}  reward={meta.get('sum_turn_scores')}  won={won}  "
        f"termination={meta.get('termination_reason')!r}",
        "=" * 100,
    ]
    if traj.align.status != OK:
        lines.append(f"{WARN} align={traj.align.status} ({traj.align.note}); the step split below is "
                     f"derived from response_mask and cannot be trusted for this rollout")
        return lines
    # Only when the two sources disagree or could not be compared. A MATCH is the expected case and
    # printing it 64 times says nothing.
    if traj.span_agreement and not traj.span_agreement.startswith("MATCH"):
        lines.append(f"{WARN} recorded turn_spans vs mask-derived segments: {traj.span_agreement}")

    pending_obs: str | None = None
    pending_header = ""
    step = 0
    for seg in traj.segments:
        text = formatter.sequence(traj.token_ids[seg.start:seg.end])
        if seg.kind in (PROMPT, ENV):
            if seg.kind == PROMPT:
                text = split_system_prompt(text)[1]
            pending_obs, pending_header = split_turn_header(text)
            continue
        turn = traj.turns[step] if step < len(traj.turns) else Turn(index=step)
        lines.append("")
        lines.append(f"## step {step + 1}")
        lines.append("[obs]")
        lines.append(pending_obs if pending_obs is not None else "(none)")
        lines.append("[assistant]")
        lines.append(pending_header + text)
        lines += _dyad_block(traj, turn, traj.token_ids[seg.start:seg.end], formatter)
        pending_obs, pending_header = None, ""
        step += 1
    return lines


def _dyad_block(traj: Traj, turn: Turn, conv_ids: list[int], formatter: TokenFormatter) -> list[str]:
    """The `[dyad]` verdict for one step, plus the per-token table when Dyad ran."""
    if traj.mode != "dyad":
        return []
    if not turn.triggered_dyad:
        return [f"[dyad] NOT triggered - the enter marker never fired, so the action head was never "
                f"consulted; all {len(conv_ids)} tokens came from the base vocabulary"]

    rows, reason = token_provenance(conv_ids, turn, formatter)
    if reason is not None:
        lines = [f"[dyad] triggered - {len(turn.decisions)} action-head decision(s)",
                 f"       {WARN} per-token provenance unavailable: {reason}"]
        for dec in turn.decisions:
            lines.append("       " + decision_line(dec))
        return lines

    counts = {VOCAB: 0, HEAD: 0, FORCED: 0}
    for row in rows:
        counts[row.kind] += 1
    lines = [f"[dyad] triggered - action head {counts[HEAD]} · vocab {counts[VOCAB]} · "
             f"forced {counts[FORCED]} token(s)",
             f"       {'idx':>4} {'token_id':>9}  {'decided by':<11} {'token':<18} why"]
    previous = None
    for i, row in enumerate(rows):
        # The `why` column repeats for long runs of identical rows (46 tokens of thinking text all
        # say "sampled before the enter marker"). Printed only where it changes, it reads as a
        # boundary marker instead of as wallpaper -- and the boundaries are the whole point here.
        why = "" if (row.kind, row.note) == previous else row.note
        previous = (row.kind, row.note)
        lines.append(f"       {i:>4} {row.token_id:>9}  {row.kind:<11} "
                     f"{formatter.token_cell(row.token_id):<18} {why}".rstrip())
    if turn.action_sent is not None:
        flag = f"  {WARN} rewritten in transit (env received {turn.action_in!r})" if turn.action_mismatch else ""
        lines.append(f"[env]  action={turn.action_sent!r}  reward={turn.reward}  done={turn.done}"
                     f"{'  invalid_action!' if turn.invalid_action else ''}{flag}")
    return lines


def decision_line(dec: dict[str, Any]) -> str:
    """One action-head decision. Kept identical in content to the retired
    decode_dyad_trajectories.py so nothing that tool could show is lost."""
    in_allowed = dec.get("chosen_in_allowed", False)
    shown, dropped = strip_truncation(dec.get("allowed_strs"))
    # `num_candidates` is the true width; the list may be short. Saying so beats printing a list
    # that silently disagrees with the number next to it.
    more = f" (+{dropped} not dumped)" if dropped else ""
    return (f"* pos={dec.get('pos')}  candidates{dec.get('num_candidates')}={shown}{more}  "
            f"chosen={dec.get('chosen_str')!r}  surface_form={dec.get('chosen_surface_form')!r}  "
            f"param value={dec.get('argument_value_text')!r}"
            + ("" if in_allowed else f"  [chosen not in admissible set!]  {WARN}"))


VIEWS = {"steps": None, "conversation": view_conversation, "aligned": view_aligned, "turns": view_turns}


# ---------------------------------------------------------------------------- run-level


def is_anomalous(traj: Traj, expected_size: int | None = None) -> bool:
    """Anything worth looking at first: bad alignment, turn-count disagreement, a rewritten action,
    an out-of-mask choice, an unexpected admissible-set size, or a vocabulary fallback."""
    if traj.align.status in (MISALIGNED, ABSENT) or traj.gen_turns_match is False:
        return True
    if traj.meta.get("fallbacks"):
        return True
    for turn in traj.turns:
        if turn.action_mismatch or turn.invalid_action:
            return True
        for dec in turn.decisions:
            if not dec.get("chosen_in_allowed", False):
                return True
            if expected_size is not None and dec.get("num_candidates") != expected_size:
                return True
    return False


def run_summary(trajs: list[Traj], events, formatter: TokenFormatter) -> list[str]:
    """Run-level counters, including the two cross-checks nothing else in the repo performs."""
    cand: dict[Any, int] = {}
    bad_allowed = total_dec = 0
    for traj in trajs:
        for turn in traj.turns:
            for dec in turn.decisions:
                total_dec += 1
                cand[dec.get("num_candidates")] = cand.get(dec.get("num_candidates"), 0) + 1
                if not dec.get("chosen_in_allowed", False):
                    bad_allowed += 1
    both = [t for tr in trajs for t in tr.turns if t.action_sent is not None and t.action_in is not None]
    mism = [t for t in both if t.action_mismatch]
    unpairable = [t for t in trajs if t.bind_gap]
    align_dist: dict[str, int] = {}
    for traj in trajs:
        align_dist[traj.align.status] = align_dist.get(traj.align.status, 0) + 1
    term: dict[Any, int] = {}
    for traj in trajs:
        key = traj.meta.get("termination_reason")
        term[key] = term.get(key, 0) + 1

    lines = [
        "=" * 100,
        f"run = {events.debug_dir}",
        f"components = {dict(sorted(events.components.items()))}",
        f"env pool component = {events.env_pool_component!r} (probed, never hardcoded)",
        f"dumped (pid,dump_seq) pairs = {events.dump_counts()}",
        f"trajectories = {len(trajs)}  modes = "
        f"{ {m: sum(1 for t in trajs if t.mode == m) for m in sorted({t.mode for t in trajs})} }",
        f"align status = {align_dist}",
        f"termination_reason = {term}",
        f"gen-segments == assistant_turns: "
        f"{sum(1 for t in trajs if t.gen_turns_match)} / "
        f"{sum(1 for t in trajs if t.gen_turns_match is not None)} checkable",
        f"action decisions = {total_dec}  admissible-size distribution = {dict(sorted(cand.items(), key=lambda kv: str(kv[0])))}  "
        f"chosen not in admissible set = {bad_allowed}/{total_dec}",
        f"action_sent == env_step_bind.action_in: violations {len(mism)}/{len(both)}"
        + (f"   {WARN} {len(unpairable)} trajectories skipped (env_step vs bind counts disagree by "
           f"{sorted({t.bind_gap for t in unpairable})})" if unpairable else ""),
    ]
    meta = events.run_meta()
    if meta:
        lines.append(f"run_meta: tokenizer={meta.get('tokenizer_path')!r} vocab_size={meta.get('vocab_size')} "
                     f"special_ids={len(formatter.special_ids)} diag_level={meta.get('diag_level')!r}")
    else:
        lines.append("run_meta: absent (dump predates the run_meta event; tokenizer taken from --model/$MODEL_PATH)")
    return lines


# =============================================================================
# decode_trajectories  (the CLI entry point)
# =============================================================================

PASS_MARK = "\033[92mPASS\033[0m"
FAIL_MARK = "\033[91mFAIL\033[0m"


def _p(cond: bool, tag: str, msg: str, out=sys.stdout) -> bool:
    print(f"  [{PASS_MARK if cond else FAIL_MARK}] {tag}: {msg}", file=out)
    return cond


def _load_tokenizer(candidates: list[tuple[str, str]]):
    """Try each (source, path) in order; return (tokenizer, source, tried) and never raise.

    Ordered rather than single-shot because the first candidate is routinely dead. With
    N_GPUS_PER_NODE>=4 the launch scripts stage the checkpoint into
    $LOCAL_RUNTIME_ROOT/models/<tag>-<snapshot>-<pid> and delete it when the run exits, so the
    `tokenizer_path` the run recorded is a directory that no longer exists. transformers then falls
    back to reading it as a Hub repo id and dies with

        OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/tmp/...'

    which says nothing about staging, nothing about which run, and nothing about what to pass
    instead. `tokenizer_source` (the pre-staging path, exported by the launch scripts) is tried
    first for that reason.
    """
    from transformers import AutoTokenizer

    tried: list[str] = []
    for source, path in candidates:
        if not path:
            continue
        tried.append(f"{source}={path}")
        if not os.path.isdir(path):
            continue
        try:
            return AutoTokenizer.from_pretrained(path), source, tried
        except Exception as exc:  # noqa: BLE001  a bad candidate must not end the whole decode
            tried[-1] += f" ({type(exc).__name__})"
    return None, None, tried


def resolve_org_vocab(meta: dict, tok) -> int:
    """Where Dyad's extended action ids begin, for this run.

    Anything at or above this is shown as an action label instead of being decoded, so getting it
    wrong silently turns real tokens into `⟦ACT:...⟧` and breaks the ids -> text round trip.

    Order, and why:

      1. `run_meta.org_vocab_size` -- the run's own truth. Dyad records it from
         `action_config["num_embeddings_size"]` (dyad_tool_agent_loop.py), which *is* the base the
         extended ids were built on.
      2. `$DYAD_ORG_VOCAB` -- an explicit operator override for a dump that predates (1).
      3. `len(tokenizer)` -- derived from the model in the run. An id at or above it cannot be a
         real token, so it must be an extended one; an id below it is a real token and must be
         decoded. grpo_react have no extended ids at all, and this correctly classifies
         every id as a token.

    There used to be a fourth step: a hardcoded 151936. That is Qwen2.5/Qwen3's embedding size and
    nothing else's. On Qwen3.5-4B (vocab 248320) every real special token -- 248044..248076 -- is
    above it, so all of them showed as `⟦ACT:96108..96140=?⟧` (that is `248044 - 151936`) and the
    round trip failed on 64/64 trajectories. A model-specific constant in the analysis layer fails
    exactly when someone changes the model, which is the one time nobody suspects the decoder.
    """
    recorded = meta.get("org_vocab_size")
    if recorded:
        return int(recorded)
    env = os.environ.get("DYAD_ORG_VOCAB")
    if env:
        return int(env)
    if tok is not None:
        return len(tok)
    # No tokenizer and no record: nothing can be decoded anyway, so classify nothing as an action
    # rather than guess a boundary.
    return 1 << 62


def build_token_formatter(events, model: str | None, format_mode: str) -> TokenFormatter:
    """Tokenizer + special-id set + action-id names, preferring what the run itself recorded."""
    meta = events.run_meta()
    tok, source, tried = _load_tokenizer([
        ("--model/$MODEL_PATH", model or os.environ.get("MODEL_PATH") or ""),
        # The pre-staging path, which survives the run. See _load_tokenizer.
        ("run_meta.tokenizer_source", str(meta.get("tokenizer_source") or "")),
        ("run_meta.tokenizer_path", str(meta.get("tokenizer_path") or "")),
    ])
    if tok is None:
        # Not fatal: the dump carries `full_text`, so the conversation view still works and
        # self_check falls back to comparing against it. But the aligned view cannot label special
        # tokens without a tokenizer, so say exactly that rather than let it look like a data bug.
        print(f"[decode] no usable tokenizer (tried: {'; '.join(tried) or 'nothing'}). "
              "Special-token labelling and re-decode are unavailable; pass --model <hf_snapshot>. "
              "A dead run_meta.tokenizer_path usually means the run staged the checkpoint to local "
              "disk and cleaned it up (MODEL_STAGE_TO_LOCAL=1, the default at N_GPUS_PER_NODE>=4).")
    elif source != "--model/$MODEL_PATH":
        print(f"[decode] tokenizer from {source}")
    special = set(parse_ids(meta["special_ids"])) if meta.get("special_ids") else None
    return TokenFormatter(tok, special, action_id_names(events), resolve_org_vocab(meta, tok), mode=format_mode)


def self_check(traj: Traj, formatter: TokenFormatter, index: int, out) -> bool:
    """`redecode == full_text`, always on the `raw` format -- any visualisation would make the
    comparison meaningless. A failure means the recorded ids and the recorded text disagree, which
    is a data bug, not a display one (contract §5)."""
    if formatter.tok is None:
        return _p(bool(traj.full_text), f"D[{index}]",
                  "no tokenizer; falling back to the recorded full_text", out)
    text = formatter.sequence(traj.token_ids, RAW)
    if traj.full_text is None:
        return _p("<|im_start|>" in text, f"D[{index}]",
                  "no recorded full_text; decode contains <|im_start|> (conversation structure intact)", out)
    return _p(text == traj.full_text, f"D[{index}]", "re-decoded full_token_ids == recorded full_text", out)


def _relpath(path: str) -> str:
    """Path relative to cwd when that is shorter, else as given. Never raises on another drive."""
    try:
        rel = os.path.relpath(path)
    except ValueError:
        return path
    return rel if len(rel) < len(path) else path


def emit(trajs, events, formatter, views, out) -> bool:
    """Write the requested views to one stream. There is no second output file.

    Everything printed here is teed into `<run>/analysis/decode_trajectories.md` by `save_report`, so one run
    produces exactly one readable artefact. It used to produce three (`decode/decode_all.md`,
    `decode/decode_aligned.md` and a `decode_trajectories.md` holding nothing but the paths of the other two),
    which meant the file named after the tool was the one file with no content in it.
    """
    ok = True
    if views == ["steps"]:
        for i, traj in enumerate(trajs):
            for line in view_steps(traj, formatter, i, len(trajs)):
                print(line, file=out)
            ok &= self_check(traj, formatter, i, out)
            print(file=out)
        return ok

    for line in run_summary(trajs, events, formatter):
        print(line, file=out)
    for i, traj in enumerate(trajs):
        for line in header(traj, formatter):
            print(line, file=out)
        for name in views:
            for line in VIEWS[name](traj, formatter):
                print(line, file=out)
        ok &= self_check(traj, formatter, i, out)
        print(file=out)
    if formatter.unknown_action_ids:
        print(f"{WARN} {len(formatter.unknown_action_ids)} action id(s) had no name in this "
              f"run's dyad_decisions: {sorted(formatter.unknown_action_ids)}", file=out)
    return ok


def emit_environment_steps(data, args) -> int:
    """Readable native transitions and saved independent generations, not token re-decode."""
    from _env_checks import is_probe, step_metadata

    print("# Shared environment-step evidence")
    print("Each saved generation is one decision. Message turns are 2 * episode decisions; "
          "they are not the decision cap.")
    print("[UNDECIDABLE] This view does not verify full token IDs, response masks or token/action-head "
          "alignment, even when an audit retains IDs. This is recorded text/native evidence, "
          "not a token round-trip check.")
    if args.view == "aligned" or args.only_anomalies:
        print("[UNDECIDABLE] Requested token alignment/anomaly filtering cannot be performed "
              "on native step evidence; use --view steps or conversation.")
        return 1
    rows = data["generations"] or data["records"]
    if not rows and data.get("evaluation_summaries"):
        rows = []
        print("Native terminal summaries retain sampled action audits. Submitted text does not "
              "prove tool execution or official scoring; inspect each episode's result separately.")
        for summary in data["evaluation_summaries"]:
            for episode in summary["episodes"]:
                print(f"episode={episode.get('episode_id')} status={episode.get('status')} "
                      f"metric_valid={episode.get('metric_valid')} "
                      f"official_scored={episode.get('official_scored')}")
                for audit in episode.get("action_audit", []):
                    text = audit.get("raw_text", "[not recorded]")
                    if not isinstance(text, str):
                        raise ValueError("Native action audit raw_text must be textual")
                    if not args.out and len(text) > 1000:
                        text = text[:1000] + f"\n[truncated; {len(text)} characters total; use --out for full text]"
                    prompt = (json.dumps(audit["messages"], ensure_ascii=False)
                              if "messages" in audit else "[not recorded]")
                    if not args.out and len(prompt) > 1000:
                        prompt = (prompt[:1000]
                                  + f"\n[truncated; {len(prompt)} characters total; use --out for full text]")
                    rows.append({"_source": summary["_source"], "uid": episode.get("episode_id"),
                                 "output": text, "input": prompt})
    selected = rows
    if args.request_id:
        selected = [r for r in selected if str(
            (step_metadata(r) or {}).get("trajectory_id", r.get("uid", r.get("request_id", "")))
        ).startswith(args.request_id)]
    if not args.out:
        selected = selected[:args.show]
    print(f"\n## Saved generation rows ({len(rows)} total, {len(selected)} shown)")
    print("Pool sessions and saved validation rows are separate evidence streams. "
          "Without explicit shared IDs they are not joined or assigned to each other's split.")
    for i, row in enumerate(selected):
        meta = step_metadata(row) or {}
        print(f"\n### Decision row {i + 1} source={row['_source']} "
              f"id={meta.get('trajectory_id', row.get('uid', row.get('request_id')))} "
              f"step_index={meta.get('step_index', 'not recorded')}")
        if meta:
            print(f"episode_length={meta.get('episode_length')} "
                  f"env_reward={meta.get('env_reward')} episode_reward={meta.get('episode_reward')} "
                  f"executed={meta.get('executed')} action_valid={meta.get('action_valid')} "
                  f"termination_reason={meta.get('termination_reason')}")
        print("Input (recorded)")
        print(row.get("input", row.get("raw_prompt", "[not recorded]")))
        print("Output (recorded)")
        print(row.get("output", row.get("response", "[not recorded]")))
    sessions = {}
    for event in data["events"]:
        if event.get("event") == "env_step_bind" and not is_probe(event.get("session_id")):
            key = (event["_source"], str(event.get("session_id")))
            sessions.setdefault(key, []).append(event)
    print(f"\n## Native environment sessions ({len(sessions)} total; probes excluded)")
    print("These are executed environment calls, not necessarily all policy decisions. "
          "Native rewards may differ from scaled/shaped training rewards.")
    selected_sessions = [(key, steps) for key, steps in sessions.items()
                         if not args.request_id or key[1].startswith(args.request_id)]
    if not args.out:
        selected_sessions = selected_sessions[:args.show]
    for (source, session), steps in selected_sessions:
        print(f"\n### Session {session} source={source} native_steps={len(steps)}")
        for i, step in enumerate(steps):
            print(f"\nNative call {i + 1}: action={step.get('action_in')!r} "
                  f"reward={step.get('reward')} done={step.get('done')}")
            print(step.get("observation_out", "[observation not recorded]"))
    if not rows and not sessions:
        print("[FAIL] No saved decision or real native transition evidence.")
        return 1
    print("\nRecorded-evidence display complete; masks, gradients and optimizer success not verified.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("debug_dir", nargs="?", default=None)
    ap.add_argument("--model", default=None, help="tokenizer path (defaults to the run's run_meta, then $MODEL_PATH)")
    ap.add_argument("--view", default="steps", choices=["steps", "conversation", "aligned", "turns"])
    ap.add_argument("--format", default=RAW, choices=[RAW, ESCAPED],
                    help="raw: verbatim decode, nothing inserted; escaped: repr() per token. "
                         "Per-token properties live in the aligned view's columns, not in the text")
    ap.add_argument("--sample", default="first", choices=["first", "diverse"],
                    help="diverse: one per (termination_reason, won) stratum -- serves VALIDATION §4")
    ap.add_argument("--show", type=int, default=10, help="print at most N trajectories")
    ap.add_argument("--request-id", default=None, help="only this request_id (a prefix is accepted)")
    ap.add_argument("--only-anomalies", action="store_true",
                    help="only trajectories with bad alignment / turn mismatch / rewritten action / out-of-mask choice / fallback")
    ap.add_argument("--out", action="store_true",
                    help="every trajectory instead of --show N; the report is always written to "
                         "<debug_dir>/analysis/decode_trajectories.md either way")
    args = ap.parse_args()

    debug_dir = resolve_dir(args.debug_dir)
    with save_report("decode_trajectories", debug_dir):
        events = load(debug_dir)
        trajs = build_all(events)
        if not trajs:
            from _env_checks import step_run_data

            data = step_run_data(debug_dir)
            if data["shared"]:
                return emit_environment_steps(data, args)
            _p(False, "D0", f"{debug_dir} has no trajectory_summary carrying full_token_ids "
                            f"(needs DYAD_DIAG_ENABLED=1 DYAD_DIAG_LEVEL=full)")
            return 1
        formatter = build_token_formatter(events, args.model, args.format)
        views = [args.view]

        selected = trajs
        if args.request_id:
            selected = [t for t in selected if t.request_id.startswith(args.request_id)]
        if args.only_anomalies:
            selected = [t for t in selected if is_anomalous(t, _expected_admissible_size(events))]
        if not args.out:
            selected = sample(selected, args.sample, args.show)
        if not selected:
            print("(no matching trajectory)")
            return 0

        if views == ["steps"]:
            for line in steps_preamble(selected, events, formatter):
                print(line)
        return 0 if emit(selected, events, formatter, views, sys.stdout) else 1


def steps_preamble(trajs, events, formatter: TokenFormatter) -> list[str]:
    """The legend plus the one thing every rollout shares: the system prompt.

    Deliberately short. The run-level counter block that used to open this file (component census,
    dump_seq pairs, align distribution, admissible-size distribution, tokenizer path) is what
    `verify_dyad.py` reports, and reporting it in two places let the two drift.
    """
    heads = {split_system_prompt(formatter.sequence(t.token_ids[:t.align.resp_start or 0]))[0]
             for t in trajs if t.align.resp_start}
    lines = [
        f"# decode · {os.path.basename(str(events.debug_dir).rstrip(os.sep))}",
        "",
        f"{len(trajs)} rollout(s). Each step is one assistant generation: the observation that led "
        f"into it, the turn verbatim, and who decided each generated token.",
        "",
        "  vocab        the policy sampled this token from the base vocabulary",
        "  action head  the Dyad action head chose the action; these are its surface form",
        "  forced       the router wrote it (marker completion / exit marker / turn end)",
        "",
    ]
    if len(heads) == 1:
        lines += ["-" * 100,
                  "system prompt (identical in every rollout, so it is printed here and nowhere else)",
                  "-" * 100,
                  heads.pop(), ""]
    else:
        lines += [f"{WARN} {len(heads)} distinct system prompts in this run; each rollout prints its own.",
                  ""]
    return lines


def _expected_admissible_size(events) -> int | None:
    """Only gsm8k has a fixed admissible-set size (calculate/answer); elsewhere the dynamic mask
    genuinely varies, so an "unexpected size" flag would be pure noise."""
    return 2 if str(events.run_meta().get("mode") or "").startswith("calc") else None


if __name__ == "__main__":
    raise SystemExit(main())
