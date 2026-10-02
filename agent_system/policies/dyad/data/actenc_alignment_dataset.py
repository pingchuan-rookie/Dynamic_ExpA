"""Loading Alignment samples and grouping them into batches.

Deliberately not a `torch.utils.data.Dataset`. A Alignment sample carries a ragged field -- one
encoder input per admissible action, and their number varies from four to ten -- so the collation a
DataLoader would do is wrong: padding the action axis and masking it later is a second place for
the admissible set to be defined, and section 1 of AGENTS.md is about exactly one thing being able
to define it. The model is handed the samples and does its own tokenisation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator, Sequence

from agent_system.policies.dyad.data.actenc_alignment_parquet import read_dataset, select_split

REQUIRED_FIELDS = (
    "case_id", "action_set_form", "policy_lm_prompt", "action_encoder_prompts",
    "action_set", "label", "domain", "mcp_size",
)


def validate_sample(row: dict[str, Any], *, source: str = "Alignment sample") -> None:
    """Require every displayed action to be encoded and scored, in catalogue order.

    This is a Alignment-only contract, not a restriction on AgenticRL router candidate sets.
    Legacy fields are unsupported, never silently dropped or accepted.
    """
    if not isinstance(row, dict):
        raise ValueError(f"{source} must be an object")
    where = f"{source} ({row.get('case_id', '?')})"
    legacy = [field for field in ("admissible_actions", "visible_actions", "action_catalogue", "catalogue_form") if field in row]
    if legacy:
        raise ValueError(
            f"{where} has legacy fields {legacy}; Alignment uses only action_set. "
            "Legacy samples are unsupported; load a current schema-v5 train/test dataset."
        )
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"{where} is missing {missing}")
    catalogue = row["action_set"]
    if not isinstance(catalogue, list) or not all(
        isinstance(action, str) and action.strip() for action in catalogue
    ):
        raise ValueError(f"{where} action_set must be a list of non-empty action names")
    if len(set(catalogue)) != len(catalogue):
        raise ValueError(f"{where} has duplicate actions in action_set")
    if not 4 <= len(catalogue) <= 10:
        raise ValueError(f"{where} must display 4-10 actions, got {len(catalogue)}")
    if type(row["mcp_size"]) is not int or row["mcp_size"] != len(catalogue):
        raise ValueError(f"{where} mcp_size must equal the displayed catalogue size {len(catalogue)}")
    if not isinstance(row["label"], str) or row["label"] not in catalogue:
        raise ValueError(f"{where} labels {row['label']!r}, which is not in its displayed catalogue")
    prompts = row["action_encoder_prompts"]
    if not isinstance(prompts, dict):
        raise ValueError(f"{where} action_encoder_prompts must be an object")
    absent = [action for action in catalogue if action not in prompts]
    if absent:
        raise ValueError(f"{where} has no encoder prompt for {absent}")
    extra = [action for action in prompts if action not in catalogue]
    if extra:
        raise ValueError(f"{where} has extra encoder prompts outside the displayed catalogue: {extra}")
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts.values()):
        raise ValueError(f"{where} every encoder prompt must be a non-empty string")
    if not isinstance(row["policy_lm_prompt"], str) or not row["policy_lm_prompt"].strip():
        raise ValueError(f"{where} policy_lm_prompt must be a non-empty string")


def load_split(path: Path | str, *, split: str = "train", limit: int = 0) -> list[dict[str, Any]]:
    """Read Parquet and validate the full split before applying a smoke-run limit."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    rows = select_split(read_dataset(path), split)
    for i, row in enumerate(rows, start=1):
        validate_sample(row, source=f"{path}:{i}")
    return rows[:limit] if limit else rows


def batches(
    rows: Sequence[dict[str, Any]],
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
) -> Iterator[list[dict[str, Any]]]:
    order = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [rows[i] for i in order[start:start + batch_size]]


def chance_accuracy(rows: Sequence[dict[str, Any]]) -> float:
    """`mean(1 / |C_t|)`: what a uniform policy scores on this split.

    Not `1/7`. The admissible sets range from four to ten actions, so the uniform baseline is an
    average of seven different numbers, and comparing an accuracy against the wrong one of them is
    how a model that learned nothing gets called an improvement.
    """
    if not rows:
        return 0.0
    return sum(1.0 / len(row["action_set"]) for row in rows) / len(rows)


def chance_cross_entropy(rows: Sequence[dict[str, Any]]) -> float:
    """`mean(log |C_t|)`: the cross-entropy of that same uniform policy."""
    import math

    if not rows:
        return 0.0
    return sum(math.log(len(row["action_set"])) for row in rows) / len(rows)
