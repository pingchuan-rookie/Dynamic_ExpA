"""The balanced MCP-subset scheduler (strategy document section 1.6).

The one thing this module exists to prevent is the generation order the document forbids:

    target action -> generate context/reasoning -> delete actions from the MCP

Under that order, the visible action set is chosen after the strong LLM has already written a
situation, so which distractors are present is a function of what the LLM happened to say. The
resulting dataset has a correlation between "what the context talks about" and "which actions were
left visible", and a model can score well on it by reading the catalogue rather than the reasoning.
The failure is invisible in every aggregate statistic.

So `(mcp_size, target_action, action_set)` is decided here, first, and the strong LLM is
handed a fixed, already-cropped catalogue.

The three balances this produces, all asserted by check_action_encoder_alignment_dataset.py:

    mcp size            each k in 4..10 gets exactly n_per_size cases
    target action       each target is the label exactly n_per_size / len(targets) times per k
    distractor exposure non-target actions appear as distractors about equally often, and each
                        target co-occurs with each other action about equally often

The first two are exact and easy. The third cannot be exact -- k-1 distractors drawn from a pool
whose size does not divide evenly -- so it is greedy: always draw from the least-used actions,
break ties by least co-occurrence with this target, break remaining ties at random.
"""

from __future__ import annotations

import random
from typing import Any, Optional, Sequence


def build_balanced_mcp_subsets(
    domain_actions: Sequence[str],
    n_per_size: int,
    min_actions: int = 4,
    max_actions: int = 10,
    seed: int = 0,
    *,
    targets: Optional[Sequence[str]] = None,
    distractor_pool: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    """Rows of `{mcp_size, target_action, action_set}`, jointly balanced.

    `domain_actions` is the domain's full draw pool. `targets` and `distractor_pool` default
    to it, which is the configuration the strategy document describes: ten actions per domain, any
    of which can be the label and any of which can be a distractor.

    They are separate arguments because the unseen split needs them to differ. There, the label is
    always one of four held-out actions while the distractors may come from all fourteen -- an
    unseen action surrounded only by other unseen actions would test something narrower than the
    metric asks for.

    The balance requirement is stated in the document as `n_per_size % 10 == 0`. Ten is the size of
    the target pool in that instance, so the constraint is derived from `len(targets)` rather than
    written as a literal; with four unseen targets the modulus is four.
    """
    pool = list(dict.fromkeys(domain_actions))
    if len(pool) != len(domain_actions):
        raise ValueError(f"domain_actions has duplicates: {list(domain_actions)}")
    target_list = list(targets) if targets is not None else list(pool)
    draw_pool = list(distractor_pool) if distractor_pool is not None else list(pool)

    unknown = [a for a in target_list if a not in pool]
    if unknown:
        raise ValueError(f"targets not in domain_actions: {unknown}")
    unknown = [a for a in draw_pool if a not in pool]
    if unknown:
        raise ValueError(f"distractor_pool not in domain_actions: {unknown}")

    if not target_list:
        raise ValueError("targets is empty; there would be nothing to label")
    if n_per_size % len(target_list) != 0:
        raise ValueError(
            f"n_per_size={n_per_size} is not a multiple of len(targets)={len(target_list)}, so the "
            "targets cannot appear equally often under each mcp size. The joint "
            "(mcp_size, target_action) distribution would be lopsided, and nothing downstream "
            "would report it."
        )
    if min_actions < 1 or max_actions < min_actions:
        raise ValueError(f"need 1 <= min_actions <= max_actions, got {min_actions}, {max_actions}")
    if max_actions > len(draw_pool):
        raise ValueError(
            f"max_actions={max_actions} exceeds the {len(draw_pool)} actions available to draw "
            "from; the largest subsets could not be filled"
        )

    rng = random.Random(seed)
    per_target = n_per_size // len(target_list)
    distractor_count: dict[str, int] = {a: 0 for a in draw_pool}
    cooccurrence: dict[str, dict[str, int]] = {t: {a: 0 for a in draw_pool} for t in target_list}

    rows: list[dict[str, Any]] = []
    for size in range(min_actions, max_actions + 1):
        slots = [t for t in target_list for _ in range(per_target)]
        rng.shuffle(slots)
        for target in slots:
            distractors = [a for a in draw_pool if a != target]
            if len(distractors) < size - 1:
                raise ValueError(
                    f"mcp_size={size} needs {size - 1} distractors but only {len(distractors)} "
                    f"actions exist beside {target!r}"
                )
            ranked = sorted(
                distractors,
                key=lambda a: (distractor_count[a], cooccurrence[target][a], rng.random()),
            )
            chosen = ranked[: size - 1]
            for action in chosen:
                distractor_count[action] += 1
                cooccurrence[target][action] += 1
            visible = [target, *chosen]
            rng.shuffle(visible)
            rows.append({
                "mcp_size": size,
                "target_action": target,
                "action_set": visible,
            })
    return rows


def exposure_spread(rows: Sequence[dict[str, Any]]) -> int:
    """max - min over "how often did this action appear as a non-target", across a domain's rows.

    Reported rather than asserted here: the acceptable spread depends on n_per_size and on whether
    the target pool equals the draw pool, and the gate is where that judgement belongs.
    """
    counts: dict[str, int] = {}
    for row in rows:
        for action in row["action_set"]:
            if action != row["target_action"]:
                counts[action] = counts.get(action, 0) + 1
    if not counts:
        return 0
    return max(counts.values()) - min(counts.values())
