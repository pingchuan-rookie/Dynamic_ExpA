"""Fixed task membership shared by evaluators and result verification."""
from __future__ import annotations

import random

PROTOCOL_ID = "env-eval-fixed-v1"
SUBSET_SEED = 42
ENVIRONMENTS = ("alfworld", "webshop", "t2bench", "swebench_verified")
TASK_COUNTS = {"alfworld": 134, "webshop": 100, "swebench_verified": 50}
SOURCE_COUNTS = {"alfworld": 134, "webshop": 500, "swebench_verified": 500}
SPLITS = {"alfworld": "valid_unseen", "webshop": "test", "swebench_verified": "test"}


def select_task_ids(benchmark, source_ids):
    """Sample from canonical IDs, independently of file order and rollout seed."""
    ids = sorted(source_ids)
    if len(ids) != len(set(ids)) or len(ids) != SOURCE_COUNTS[benchmark]:
        raise ValueError(f"{benchmark} requires {SOURCE_COUNTS[benchmark]} distinct source task IDs")
    if benchmark == "webshop" and ids != list(range(500)):
        raise ValueError("WebShop source must be the official test task IDs 0..499")
    count = TASK_COUNTS[benchmark]
    return ids if count == len(ids) else sorted(random.Random(SUBSET_SEED).sample(ids, count))


def selection_identity(benchmark, source_ids):
    selected = select_task_ids(benchmark, source_ids)
    return {"protocol": PROTOCOL_ID, "split": SPLITS[benchmark],
            "source_count": SOURCE_COUNTS[benchmark], "selected_count": len(selected),
            "selection_seed": None if benchmark == "alfworld" else SUBSET_SEED,
            "selected_task_ids": selected}
