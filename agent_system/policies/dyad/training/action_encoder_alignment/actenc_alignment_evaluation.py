"""Alignment candidate accuracy, cross-entropy and demonstrated-action probability.

All metrics and chance baselines come from the same rows and candidate softmax.
The trainer selects checkpoints on val; only explicit final evaluation scores test.
Distributed replicas score disjoint slices and sum counts before taking means.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence


def file_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}

import torch
import torch.distributed as dist

from agent_system.policies.dyad.algorithms.action_selection import admissible_log_probs
from agent_system.policies.dyad.data.actenc_alignment_dataset import batches, chance_accuracy, chance_cross_entropy


def _world() -> tuple[int, int]:
    """`(rank, world_size)`, and `(0, 1)` when nothing distributed was ever initialised."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


@torch.no_grad()
def evaluate(model, rows: Sequence[dict[str, Any]], batch_size: int = 8) -> dict[str, float]:
    if not rows:
        return {}
    rank, world = _world()
    # A strided slice, not a contiguous block: the splits are ordered by case and a contiguous cut
    # would hand one replica the easy sizes and another the hard ones, so the per-replica numbers
    # would differ for a reason that has nothing to do with the model. The totals are summed
    # afterwards, so the split only has to be disjoint and complete -- which striding is.
    mine = rows[rank::world] if world > 1 else rows
    correct = 0
    nll_total = 0.0
    prob_total = 0.0
    for batch in batches(mine, batch_size):
        for sample, logits in zip(batch, model.score_batch(batch)):
            log_probs = admissible_log_probs(logits)
            label = sample["action_set"].index(sample["label"])
            correct += int(torch.argmax(log_probs).item() == label)
            nll_total += float(-log_probs[label].item())
            prob_total += float(log_probs[label].exp().item())
    seen = len(mine)
    if world > 1:
        totals = torch.tensor([correct, nll_total, prob_total, seen], dtype=torch.float64,
                              device=torch.device("cuda", torch.cuda.current_device()))
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        correct, nll_total, prob_total, seen = (
            float(totals[0]), float(totals[1]), float(totals[2]), int(totals[3]))
    n = len(rows)
    assert seen == n, f"replicas covered {seen} rows of {n}; the slices are not disjoint-complete"
    return {
        "n": n,
        "top1_accuracy": correct / n,
        "cross_entropy": nll_total / n,
        "demonstrated_action_probability": prob_total / n,
        "chance_accuracy": chance_accuracy(rows),
        "chance_cross_entropy": chance_cross_entropy(rows),
        "mean_candidates": sum(len(r["action_set"]) for r in rows) / n,
    }


def by_group(model, rows: Sequence[dict[str, Any]], key: str,
             batch_size: int = 8) -> dict[str, dict[str, float]]:
    """The same metrics sliced by one field, usually `action_set_form` or `mcp_size`.

    The tool-specification and natural-language halves are the axis the dataset was built around,
    and an aggregate number hides the case where the head learned one action description format and not the
    other -- which is the single most likely way for Alignment to look like it worked.
    """
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    return {str(value): evaluate(model, members, batch_size)
            for value, members in sorted(groups.items(), key=lambda kv: str(kv[0]))}
