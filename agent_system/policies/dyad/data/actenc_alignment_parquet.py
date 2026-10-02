"""Explicit, lossless storage for the single Alignment Parquet dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ("train", "val", "test")
SPLIT_POLICY = "seen-train-unseen-val-test-v1"
LEGACY_SPLIT_POLICY = "seen-train-unseen-test-v1"
SPLIT_POLICIES = {SPLIT_POLICY: SPLITS, LEGACY_SPLIT_POLICY: ("train", "test")}
SCHEMA_VERSION = 5
STRING_FIELDS = (
    "split", "case_id", "domain", "entry_marker_rule", "entry_marker", "action_set_form",
    "label", "context", "reasoning", "action_family", "format_example_action", "policy_lm_prompt",
)
LIST_FIELDS = ("action_set",)
MAP_FIELDS = ("action_definitions", "action_encoder_prompts")
SCHEMA = pa.schema([
    *(pa.field(name, pa.string(), nullable=False) for name in STRING_FIELDS),
    pa.field("mcp_size", pa.int32(), nullable=False),
    *(pa.field(name, pa.list_(pa.field("element", pa.string(), nullable=False)), nullable=False)
      for name in LIST_FIELDS),
    *(pa.field(name, pa.map_(pa.string(), pa.field("value", pa.string(), nullable=False)), nullable=False)
      for name in MAP_FIELDS),
])


def _validate_rows(rows: list[dict[str, Any]], *, split_policy: str = SPLIT_POLICY) -> None:
    """Validate structure and isolation; legacy policy is only for explicit migration."""
    if split_policy not in SPLIT_POLICIES:
        raise ValueError(f"unsupported split policy {split_policy!r}")
    splits = SPLIT_POLICIES[split_policy]
    seen = set()
    case_splits: dict[str, str] = {}
    present = set()
    for i, row in enumerate(rows):
        if set(row) != set(SCHEMA.names):
            raise ValueError(
                f"row {i}: fields must equal the Alignment Parquet schema v{SCHEMA_VERSION}; "
                "legacy fields are unsupported; use a current schema-v5 train/val/test dataset"
            )
        for name in STRING_FIELDS:
            if not isinstance(row[name], str) or not row[name].strip():
                raise ValueError(f"row {i}: {name} must be a non-empty string")
        split = row["split"]
        if split not in splits:
            raise ValueError(f"row {i}: invalid split {split!r}; expected {'/'.join(splits)}")
        present.add(split)
        if row["action_set_form"] not in ("tool-specification", "natural-language"):
            raise ValueError(f"row {i}: invalid action_set_form")
        key = (row["case_id"], row["action_set_form"])
        if key in seen:
            raise ValueError(f"duplicate sample {key}")
        seen.add(key)
        previous = case_splits.setdefault(row["case_id"], split)
        if previous != split:
            raise ValueError(f"case {row['case_id']} crosses splits")
        if type(row["mcp_size"]) is not int:
            raise ValueError(f"row {i}: mcp_size must be an integer")
        for name in LIST_FIELDS:
            value = row[name]
            if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
                raise ValueError(f"row {i}: invalid {name}")
        for name in MAP_FIELDS:
            value = row[name]
            if not isinstance(value, dict) or any(
                not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
                for k, v in value.items()
            ):
                raise ValueError(f"row {i}: invalid {name}")
    if present != set(splits):
        raise ValueError(
            f"missing splits: {sorted(set(splits) - present)}; expected {split_policy}. "
            "Legacy train/test data requires explicit offline holdout migration into a new data root."
        )
    train_actions = {action for row in rows if row["split"] == "train" for action in row["action_set"]}
    for split in splits[1:]:
        labels = {row["label"] for row in rows if row["split"] == split}
        if overlap := train_actions & labels:
            raise ValueError(f"{split} labels overlap training candidates: {sorted(overlap)}")
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    for case_id, pair in by_case.items():
        if len(pair) != 2 or {r["action_set_form"] for r in pair} != {
            "tool-specification", "natural-language"
        }:
            raise ValueError(f"case {case_id}: requires one MCP/NL pair")
        for field in ("domain", "label", "mcp_size", "action_set", "context", "reasoning",
                      "entry_marker", "format_example_action"):
            if pair[0][field] != pair[1][field]:
                raise ValueError(f"case {case_id}: MCP/NL pair disagrees on {field}")


def read_dataset(
    path: Path | str, *, expected_split_policy: str = SPLIT_POLICY,
) -> list[dict[str, Any]]:
    """Read losslessly; legacy data requires an explicit policy, never a split alias."""
    path = Path(path)
    table = pq.ParquetFile(path).read()
    declared = (table.schema.metadata or {}).get(b"split_policy")
    if declared is not None and declared.decode("utf-8") != expected_split_policy:
        raise ValueError(f"{path}: split policy {declared!r}; expected {expected_split_policy}")
    if not table.schema.equals(SCHEMA, check_metadata=False):
        raise ValueError(
            f"{path}: incompatible Alignment Parquet schema; expected schema v{SCHEMA_VERSION} "
            "with action_set as the sole candidate list. "
            "Legacy schemas are unsupported; use a current schema-v5 train/val/test dataset."
        )
    rows = table.to_pylist()
    for i, row in enumerate(rows):
        for name in MAP_FIELDS:
            pairs = row[name]
            if pairs is None:
                raise ValueError(f"row {i}: null {name}")
            keys = [key for key, _ in pairs]
            if len(set(keys)) != len(keys):
                raise ValueError(f"row {i}: duplicate map keys in {name}")
            row[name] = dict(pairs)
    _validate_rows(rows, split_policy=expected_split_policy)
    return rows


def select_split(rows: Iterable[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected {SPLITS}")
    return [{key: value for key, value in row.items() if key != "split"}
            for row in rows if row["split"] == split]
