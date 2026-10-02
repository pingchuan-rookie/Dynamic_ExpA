#!/usr/bin/env python3
# Copyright 2025 ExpA_verl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Build the shared gsm8k dataset for GRPO-ReAct and Dyad."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from experiments.shared.dataset.utils import source_snapshot as snapshot

DATA_SOURCE = "openai/gsm8k"

# Keep the public question as source data. The shared evaluation session renders
# agent_system.environments.prompts.gsm8k from the actual calculator state.
# No observation or history is fabricated during dataset preparation.

# Calculator tool name (aligned with tool_name in the tool config yaml and TOOL_NAME in the parser).
TOOL_NAME = "calculator"

_ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


# ---------------------------------------------------------------------------
# Source data reading and normalisation
# ---------------------------------------------------------------------------


def extract_solution(answer_text: str) -> str:
    """Extract the final `#### N` answer from the gsm8k answer text (thousands separators removed)."""
    m = _ANS_RE.search(answer_text)
    if m is None:
        raise ValueError(f"no '#### <answer>' in: {answer_text!r}")
    return m.group(1).strip().replace(",", "")


def _normalize(row: dict) -> dict:
    """Normalise one source row into {question, answer, ground_truth}.

    Two column formats are supported:
      - original GSM8K: {question, answer}
      - veRL preprocessed: {prompt, reward_model{ground_truth}, extra_info{question, answer}}
    """
    source = row if "question" in row and "answer" in row else row.get("extra_info") or {}
    question, answer = source.get("question"), source.get("answer")
    if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer:
        raise ValueError("Complete source question/answer required; never recover from rendered prompt")
    ground_truth = extract_solution(answer)
    previous = (row.get("reward_model") or {}).get("ground_truth")
    if previous is not None and str(previous) != ground_truth:
        raise ValueError("Source answer and reward ground truth disagree")
    return {"question": question, "answer": answer, "ground_truth": ground_truth}


# ---------------------------------------------------------------------------
# Building rows / building splits
# ---------------------------------------------------------------------------


def build_row(row: dict, idx: int, split: str) -> dict:
    """Build one algorithm-neutral multi-turn calculator task."""
    norm = _normalize(row)
    question, answer, ground_truth = norm["question"], norm["answer"], norm["ground_truth"]

    tools_kwargs = {
        # The calculator's own copy of the answer. Not a duplicate that can be dropped: this is the
        # only channel a tool has -- see the module docstring.
        TOOL_NAME: {
            "create_kwargs": {"ground_truth": ground_truth},
        }
    }

    row = {
        "data_source": DATA_SOURCE,
        "index": idx,
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "index": idx,
            "question": question,
            "answer": answer,
            "need_tools_kwargs": True,
            "tools_kwargs": tools_kwargs,
        },
    }
    assert_no_dead_ground_truth_copy(row)
    return row


def assert_no_dead_ground_truth_copy(row: dict) -> None:
    """Refuse a ground-truth copy that nothing reads.

    `reward_model.ground_truth` and the calculator's `create_kwargs.ground_truth` are both live, so
    this checks the one field that was not: `interaction_kwargs`. Naming it explicitly rather than
    counting copies is deliberate -- "how many times may the answer appear" is not the property that
    matters, "does each appearance have a reader" is, and only the second one stays true when the
    tool plumbing changes.

    Worth a guard rather than a comment because a dead copy has no symptom. It survives every shape
    check and every metric, and only becomes visible when someone edits one copy and not the others.
    """
    extra = row.get("extra_info") or {}
    if "interaction_kwargs" in extra:
        raise ValueError(
            "extra_info.interaction_kwargs is back. It repeats reward_model.ground_truth and the "
            "user message, and nothing reads it: verl 0.9 deleted verl.interactions, so the branch "
            "that consumed it cannot be reached. See this module's docstring."
        )


SPLITS = {"train": 7473, "test": 1319}
SOURCE = {
    "dataset": "openai/gsm8k", "status": "recovered_from_local_preprocessed",
    "artifact": "data/gsm8k/source/{train,test}.parquet",
    "split_mapping": {"train": "train", "test": "test"},
    "usage": "evaluation_only",
    "limitations": ["Question and answer recovered verbatim from local preprocessed extra_info; complete local artifacts retained.",
                    "No authenticated official native artifact/revision or release-byte digest was available; local artifacts are not certified official snapshots."],
}


def source_rows(root: Path, split: str):
    import itertools
    path = f"source/local_preprocessed/{split}.parquet"
    digest = snapshot.sha256(root / path)
    source = snapshot.iter_rows(root / path)
    old = snapshot.iter_rows(root / f"source/local_selection/{split}.parquet")
    sentinel = object()
    for index, (raw, previous) in enumerate(itertools.zip_longest(source, old, fillvalue=sentinel)):
        if raw is sentinel or previous is sentinel:
            raise ValueError(f"GSM8K source/selection count mismatch: {split}")
        norm = _normalize(raw)
        if _normalize(previous) != norm or previous["index"] != index:
            raise ValueError(f"GSM8K source/selection identity mismatch: {split}[{index}]")
        if previous["extra_info"]["tools_kwargs"][TOOL_NAME]["create_kwargs"]["ground_truth"] != norm["ground_truth"]:
            raise ValueError("Calculator and source ground truth disagree")
        row = build_row(raw, index, split)
        identity = f"gsm8k/{split}/{index}"
        yield snapshot.mark_row(row, identity, {"path": path, "sha256": digest,
                                "row": index, "task_id": identity}, split)


def validate_source_dataset(root: Path) -> None:
    manifest = snapshot.validate_manifest(root, "gsm8k")
    if {k: v["rows"] for k, v in manifest["splits"].items()} != SPLITS:
        raise ValueError("GSM8K split membership contract changed")
    for split in SPLITS:
        snapshot.compare_rows(root, split, source_rows(root, split))


def rebuild_source_dataset(destination: Path, source_dir: Path, *, overwrite=False):
    def populate(stage, active):
        if not (stage / "source").exists():
            snapshot.preserve_selection(active, stage, SPLITS)
            for split in SPLITS:
                snapshot.copy_asset(source_dir / f"{split}.parquet", stage,
                                    f"source/local_preprocessed/{split}.parquet")
        for split in SPLITS:
            snapshot.write_rows(stage / f"{split}.parquet", source_rows(stage, split))
        snapshot.make_manifest(stage, "gsm8k", SPLITS, SOURCE)
    return snapshot.rebuild(destination, overwrite=overwrite, populate=populate, validate=validate_source_dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--gsm8k_dir",
        "--src",
        dest="gsm8k_dir",
        default=str(_ROOT / "data/gsm8k/source"),
        help="Local preprocessed GSM8K directory (train.parquet/test.parquet) for initial source packaging; "
             "subsequent rebuilds reuse packaged source assets.",
    )
    parser.add_argument(
        "--out_root",
        default=os.path.join(Path(os.path.abspath(__file__)).parents[3], "data"),
        help="output root directory (default: the repository data/).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Authorize transactional replacement with recovery snapshot")
    parser.add_argument("--check-only", "--validate-only", action="store_true", help="Validate all packaged source assets and derived rows, without rebuilding")
    args = parser.parse_args()
    destination = Path(args.out_root).expanduser().resolve() / "gsm8k/dataset"
    if args.check_only:
        snapshot.check_only(destination, validate_source_dataset)
        return
    rebuild_source_dataset(destination, Path(args.gsm8k_dir).expanduser(), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
