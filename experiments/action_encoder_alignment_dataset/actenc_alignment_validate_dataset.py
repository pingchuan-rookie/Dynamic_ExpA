#!/usr/bin/env python3
"""Gate: the shipped Alignment dataset is what its inputs say it should be.

    .venvs/expa-verl/bin/python experiments/action_encoder_alignment_dataset/actenc_alignment_validate_dataset.py

**Nothing here reads a summary and believes it.** Every count is re-derived from
the rows themselves, and every encoder input is checked against the row's own
`action_definitions` rather than against the builder's report. A gate that reads the builder's own
summary only proves the builder is self-consistent, which it always is.

The checks, and the failure each one is for:

    D1  train/val/test exist with the row counts `n_per_size` and half-holdout imply
    D2  every row carries the schema's keys, with the types the trainer indexes by
    D3  all split case IDs are disjoint; val/test labels never appear in train candidates
    D4  the label is always inside `action_set` (the loss raises on this, but a run
        that dies on step 3000 is a wasted afternoon)
    D5  |C_t| == mcp_size in 4..10, with distinct candidates
    D6  `action_set` is the sole ordered candidate list; every sample validates
    D7  every encoder input contains the definition of the action it describes
    D8  both action description formats exist for every case, and they differ (a case whose two forms are
        byte-identical is one form counted twice)
    D9  the label's position inside `action_set` is not informative -- a positional
        shortcut would score far above chance while every metric looked normal
    D10 corrupted labels, singleton catalogues, legacy fields and missing prompts are rejected
    D11 final files reproduce from the case plan, generated text, catalogues and split seed
    D12 the manifest identifies the candidate/prompt versions and hashes the exact dataset products
    D13 policy inputs stop before the selected action at the format-specific decision prefix
    D14 sample documentation matches one complete stored record, without rewritten prompts

`DYAD_ALIGNMENT_DATA` points the whole thing at another tree, which is how this runs against a
throwaway dataset without touching the shipped one.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from agent_system.policies.dyad.data.actenc_alignment_parquet import (  # noqa: E402
    SCHEMA_VERSION,
    SPLIT_POLICY,
    SPLITS,
    read_dataset,
    select_split,
)
from experiments.action_encoder_alignment_dataset import actenc_alignment_generation_config as cfg  # noqa: E402

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_failures: list[str] = []


def check(tag: str, ok: bool, detail: str) -> bool:
    print(f"[alignment-dataset] {tag} {detail} -> {PASS if ok else FAIL}")
    if not ok:
        _failures.append(f"{tag} {detail}")
    return ok


def sample_is_valid(row: dict[str, Any]) -> bool:
    from agent_system.policies.dyad.data.actenc_alignment_dataset import validate_sample

    try:
        validate_sample(row)
    except (ValueError, TypeError):
        return False
    return True


def main() -> int:
    _failures.clear()
    try:
        rows = read_dataset(cfg.dataset_path())
    except (OSError, ValueError, TypeError) as exc:
        check("D1/D2", False, f"cannot read Alignment Parquet: {exc}")
        return 1
    splits = {s: select_split(rows, s) for s in SPLITS}
    final_names = {p.name for p in cfg.final_dir().iterdir()}
    check("D0", {p.name for p in cfg.data_root().iterdir()} == {"final", "intermediate"}
          and "dataset.parquet" in final_names
          and final_names <= {"dataset.parquet", "SAMPLE_RECORDS.md"},
          "two top-level directories and one final dataset, with optional sample documentation")
    # ---- D1 sizes -------------------------------------------------------------------------
    sizes = {s: len(rows) for s, rows in splits.items()}
    per_size = {s: collections.Counter(r["mcp_size"] for r in rows) for s, rows in splits.items()}
    balanced = all(len(set(c.values())) == 1 for c in per_size.values())
    generation = cfg.load_generation()
    n = int(generation["n_per_size"])
    cells = len(cfg.DOMAIN_ORDER) * len(cfg.MCP_SIZES) * len(cfg.FORMS)
    expected_sizes = {"train": cells * n,
                      "val": cells * int(generation["unseen"]["n_per_size"]) // 2,
                      "test": cells * int(generation["unseen"]["n_per_size"]) // 2}
    check("D1", balanced and sizes == expected_sizes
          and all(set(c) == set(cfg.MCP_SIZES) for c in per_size.values()),
          f"row counts {sizes} against {expected_sizes}; all configured size buckets are balanced")

    # ---- D2 schema ------------------------------------------------------------------------
    required = {"case_id", "domain", "mcp_size", "action_set_form", "action_set",
                "action_encoder_prompts", "policy_lm_prompt", "label",
                "entry_marker"}
    missing = {s: sorted(required - set(rows[0])) for s, rows in splits.items()}
    typed = all(isinstance(r["action_set"], list) and isinstance(r["label"], str)
                and isinstance(r["action_encoder_prompts"], dict)
                for rows in splits.values() for r in rows)
    check("D2", not any(missing.values()) and typed,
          f"every row has {len(required)} keys with the indexed types"
          + (f"; missing {missing}" if any(missing.values()) else ""))

    # ---- D3 the two generalisation axes ---------------------------------------------------
    cases = {s: {r["case_id"] for r in rows} for s, rows in splits.items()}
    labels = {s: {r["label"] for r in rows} for s, rows in splits.items()}
    train_candidates = {a for r in splits["train"] for a in r["action_set"]}
    from itertools import combinations
    check("D3", all(not (cases[a] & cases[b]) for a, b in combinations(SPLITS, 2))
          and all(not (train_candidates & labels[s]) for s in ("val", "test"))
          and labels["val"] == labels["test"],
          "all split case IDs are disjoint; val/test share labels absent from training candidates")
    strata = {s: collections.Counter((r["domain"], r["label"], r["mcp_size"])
                                     for r in split_rows if r["action_set_form"] == cfg.FORM_MCP)
              for s, split_rows in splits.items() if s != "train"}
    check("D3b", strata["val"] == strata["test"], "val/test cases balance exactly per domain/label/size")

    # ---- D4 label in C_t ------------------------------------------------------------------
    bad_label = [(s, r["case_id"]) for s, rows in splits.items() for r in rows
                 if r["label"] not in r["action_set"]]
    check("D4", not bad_label, "the label is inside action_set everywhere"
          + (f"; {bad_label[:3]} are not" if bad_label else ""))

    # ---- D5 every displayed candidate is distinct, with the configured size ---------------
    thin = [(s, r["case_id"], len(r["action_set"])) for s, rows in splits.items()
            for r in rows if not (
                len(r["action_set"]) == len(set(r["action_set"])) == r["mcp_size"]
                and r["mcp_size"] in cfg.MCP_SIZES)]
    check("D5", not thin, "every candidate set has mcp_size distinct entries in 4..10"
          + (f"; {thin[:3]}" if thin else ""))

    # ---- D6 one ordered candidate list, validated before training -------------------------
    invalid = [(s, r["case_id"]) for s, rows in splits.items() for r in rows
               if not sample_is_valid(r)]
    check("D6", not invalid,
          "action_set is the sole candidate list and every row passes loader validation"
          + (f"; {invalid[:3]} are invalid" if invalid else ""))

    # ---- D7 the encoder input carries this action's own definition ------------------------
    #
    # The one check that would catch a builder that drifted from its own inputs. Compared against
    # the row's own `action_definitions` rather than the catalogue files, so it does not depend on
    # those still being on disk in the state the row was built from.
    #
    # Containment only, **not** "appears after cat(e)". `prompt.index` returns the *first*
    # occurrence, and cat(e) already contains every action's definition -- so when a happens to be
    # the catalogue's first entry the index is 0 and a position test calls a perfectly good row
    # broken. That cost 1156 of 16008 rows a false failure on the first run of this gate.
    rebuilt_ok, rebuilt_total, first_bad = 0, 0, None
    for s, rows in splits.items():
        for r in rows:
            for action, prompt in r["action_encoder_prompts"].items():
                rebuilt_total += 1
                definition = (r.get("action_definitions") or {}).get(action, "")
                if definition and definition in prompt:
                    rebuilt_ok += 1
                elif first_bad is None:
                    first_bad = (s, r["case_id"], action)
    exact_coverage = all(
        set(r["action_encoder_prompts"]) == set(r.get("action_definitions", {}))
        == set(r["action_set"])
        for rows in splits.values() for r in rows
    )
    check("D7", exact_coverage and rebuilt_ok == rebuilt_total,
          f"{rebuilt_ok}/{rebuilt_total} encoder prompts contain their own def(a)"
          + (f"; first bad {first_bad}" if first_bad else ""))

    # ---- D8 both action description formats, and they differ -------------------------------------------
    by_case: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for s, rows in splits.items():
        for r in rows:
            by_case[f"{s}:{r['case_id']}"][r["action_set_form"]] = r
    both = [k for k, v in by_case.items() if len(v) == 2]
    identical = [k for k in both
                 if by_case[k][cfg.FORM_MCP]["action_encoder_prompts"]
                 == by_case[k][cfg.FORM_NL]["action_encoder_prompts"]]
    check("D8", len(both) == len(by_case) and not identical,
          f"{len(both)}/{len(by_case)} cases carry both action description formats, "
          f"{len(identical)} of them byte-identical (want 0)")

    # ---- D9 position carries no signal -----------------------------------------------------
    #
    # If the label sat at a fixed index, a model that learned only "pick slot k" would beat chance
    # by a wide margin and every metric in the summary would look healthy.
    worst = None
    for s, rows in splits.items():
        idx = [r["action_set"].index(r["label"]) for r in rows]
        first_rate = sum(1 for i in idx if i == 0) / len(idx)
        expected = sum(1 / len(r["action_set"]) for r in rows) / len(rows)
        drift = abs(first_rate - expected)
        if worst is None or drift > worst[1]:
            worst = (s, drift, first_rate, expected)
    # Allow six percentage points of sampling variation in label position.
    check("D9", worst[1] < 0.06,
          f"worst split {worst[0]}: label-at-index-0 rate {worst[2]:.3f} against "
          f"chance {worst[3]:.3f} (drift {worst[1]:.3f} < 0.06)")

    # The fixed JSON prefix is input syntax, never a filled-in target action.
    from experiments.action_encoder_alignment_dataset.actenc_alignment_prompt_rendering import (
        POLICY_PROMPT_VERSION,
        action_decision_prefix,
    )
    boundary_ok = all(
        r["policy_lm_prompt"].rsplit("\n", 1)[-1] == action_decision_prefix(r["action_set_form"], r["entry_marker"])
        for split_rows in splits.values() for r in split_rows
    )
    check("D13", boundary_ok, "every policy input stops at its format-specific action decision boundary")

    # ---- D10 counter-example ---------------------------------------------------------------
    #
    # Without this, D4 and D5 pass on an empty dataset, on a dataset the loader silently
    # truncated, and on a check that lost its loop body in a refactor.
    victim = dict(splits["train"][0])
    victim["label"] = " not an action "
    victim2 = dict(splits["train"][0])
    victim2["action_set"] = victim2["action_set"][:1]
    victim3 = dict(splits["train"][0])
    victim3["admissible_actions"] = list(victim3["action_set"])
    victim4 = dict(splits["train"][0])
    victim4["action_encoder_prompts"] = dict(victim4["action_encoder_prompts"])
    victim4["action_encoder_prompts"].pop(victim4["action_set"][-1])
    check("D10", all(not sample_is_valid(r) for r in (victim, victim2, victim3, victim4)),
          "corrupted labels, singleton catalogues, legacy fields and missing prompts are rejected")

    # Rebuild in memory only: a self-consistent edited final file must still match its plan,
    # generated text, catalogues and deterministic split. Never instantiate an LLM here.
    import yaml

    from experiments.action_encoder_alignment_dataset import actenc_alignment_generate_dataset as build

    try:
        from experiments.action_encoder_alignment_dataset.actenc_alignment_holdout_split import (
            split_metadata,
            split_unseen_holdout,
        )
        manifest = yaml.safe_load(cfg.manifest_path().read_text(encoding="utf-8"))
        split_seed = manifest["holdout_split"]["seed"]
        mcp, nl = cfg.load_both_catalogues()
        rules_by_form, markers, examples = cfg.policy_prompt_inputs(generation)
        expected_rows = {}
        for unseen in (False, True):
            suffix = "_unseen" if unseen else ""
            plan = cfg.read_jsonl(cfg.intermediate_dir() / f"case_plan{suffix}.jsonl")
            texts = {r["case_id"]: r for r in cfg.read_jsonl(
                cfg.intermediate_dir() / f"context_reasoning{suffix}.jsonl")}
            expected_rows[unseen] = [
                build.build_sample(
                    case, texts[case["case_id"]], form, mcp, nl, rules_by_form[form], markers,
                    examples[form],
                    generation["encoder_prompt"]["instruction"],
                ) for case in plan for form in cfg.FORMS
            ]
        def keyed(items):
            return {(r["case_id"], r["action_set_form"]): r for r in items}
        expected = split_unseen_holdout(
            [{**r, "split": "train"} for r in expected_rows[False]]
            + [{**r, "split": "test"} for r in expected_rows[True]], seed=split_seed,
        )
        matches = all(keyed(splits[s]) == keyed(select_split(expected, s)) for s in SPLITS)
        check("D11", matches, "final records match offline reconstruction and deterministic holdout splits")
        if "holdout_migration" in manifest:
            from experiments.action_encoder_alignment_dataset.actenc_alignment_migrate_holdout import validate_migration
            validate_migration(cfg.data_root(), manifest, read_dataset(cfg.dataset_path()))
            check("D11m", True, "source hashes, train, row order and every non-split field are unchanged")
        products = [
            *(cfg.intermediate_dir() / name for name in (
                "case_plan.jsonl", "case_plan_unseen.jsonl",
                "context_reasoning.jsonl", "context_reasoning_unseen.jsonl")),
            cfg.dataset_path(),
        ]
        hashes = {str(p.relative_to(cfg.data_root())): build._sha256(p) for p in products}
        input_hashes_ok = all(
            build._sha256(cfg.input_path(name)) == digest
            for name, digest in manifest["inputs"].items()
        )
        check("D12", manifest.get("dataset_schema_version") == SCHEMA_VERSION
              and manifest.get("policy_prompt_version") == POLICY_PROMPT_VERSION
              and manifest.get("storage_format") == "parquet"
              and manifest.get("split_column") == "split"
              and manifest.get("split_policy") == SPLIT_POLICY
              and manifest.get("holdout_split") == split_metadata(split_seed)
              and bool(manifest.get("historical_usage"))
              and manifest.get("counts") == sizes and input_hashes_ok
              and manifest.get("candidate_policy") == "visible_actions_are_candidates"
              and manifest.get("products") == hashes,
              "manifest identifies the candidate policy and hashes every dataset product")
    except (OSError, KeyError, ValueError, TypeError, yaml.YAMLError) as exc:
        check("D11/D12", False, f"offline provenance validation failed: {exc}")

    from experiments.action_encoder_alignment_dataset.actenc_alignment_gen_sample_records import main as check_samples
    check("D14", check_samples(["--check"]) == 0, "sample documentation equals the complete stored record")

    print()
    if _failures:
        print(f"[alignment-dataset] FAILED {len(_failures)}:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("[alignment-dataset] the shipped dataset matches what its inputs imply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
