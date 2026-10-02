"""Deterministic case holdout from a shared unseen-action pool, without new text."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from agent_system.policies.dyad.data.actenc_alignment_parquet import LEGACY_SPLIT_POLICY, _validate_rows

SPLIT_ALGORITHM = "stratified-case-sha256-half-v1"
DEFAULT_SPLIT_SEED = 42


def split_metadata(seed: int) -> dict[str, Any]:
    if type(seed) is not int:
        raise ValueError("split seed must be an integer")
    return {
        "algorithm": SPLIT_ALGORITHM,
        "seed": seed,
        "strata": ["domain", "label", "mcp_size"],
        "ranking": "sha256(utf8(str(seed) + ':' + case_id)), case_id",
        "assignment": "first half val, second half test; preserve physical row order",
        "unit": "case_id; MCP/NL stay paired",
        "action_pool": "val and test share unseen-to-train labels; not action-disjoint",
    }


def split_unseen_holdout(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Change only split fields of legacy unseen cases; return rows in input order."""
    if type(seed) is not int:
        raise ValueError("split seed must be an integer")
    _validate_rows(rows, split_policy=LEGACY_SPLIT_POLICY)
    strata: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in rows:
        if row["split"] == "test":
            strata[row["domain"], row["label"], row["mcp_size"]].add(row["case_id"])
    assignment = {}
    for key, cases in sorted(strata.items()):
        if len(cases) < 2 or len(cases) % 2:
            raise ValueError(f"holdout stratum {key}: needs a positive even case count, got {len(cases)}")
        ordered = sorted(cases, key=lambda case_id: (
            hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest(), case_id,
        ))
        assignment.update({case_id: "val" if i < len(ordered) // 2 else "test"
                           for i, case_id in enumerate(ordered)})
    result = [{**row, "split": assignment.get(row["case_id"], row["split"])} for row in rows]
    _validate_rows(result)
    return result
