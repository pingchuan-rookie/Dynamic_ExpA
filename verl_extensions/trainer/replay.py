# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Require complete task groups before consuming their queue evidence.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verl.trainer.ppo.v1.replay_buffer import KVBatchMeta


def replay_materialize_batch(
    self, partition_id: str, selected_prompt_uids: list[str], partition_snapshot: dict[str, dict]
) -> KVBatchMeta:
    # Preserve validation uid boundaries and reject missing or failed episodes before clearing queue entries.
    from verl.trainer.ppo.v1.replay_buffer import (
        KVBatchMeta,
        tq,
    )

    keys, tags = [], []
    selected = set(selected_prompt_uids)
    # Preserve validation uid boundaries and reject missing or failed episodes before clearing queue entries.
    materialized_uids = set()
    for key, tag in partition_snapshot.items():
        # DYAD-VALIDATION: Public validation identities may themselves contain underscores.
        # Preserve validation uid boundaries and reject missing or failed episodes before clearing queue entries.
        uid = key.rsplit("_", 2)[0] if partition_id == "val" else key.split("_")[0]
        if uid in selected:
            keys.append(key)
            tags.append(tag)
            # Preserve validation uid boundaries and reject missing or failed episodes before clearing queue
            # entries.
            materialized_uids.add(uid)

    # DYAD-VALIDATION: Never erase failure evidence or silently shrink the evaluation population.
    # This shared boundary protects both synchronous and asynchronous replay policies.
    if partition_id == "val":
        failed_uids = selected & self.failure_keys[partition_id]
        missing_uids = selected - materialized_uids
        if failed_uids or missing_uids:
            raise RuntimeError(
                f"Incomplete validation: expected_uids={len(selected)}, "
                f"materialized_uids={len(materialized_uids)}, failed_uids={len(failed_uids)}, "
                f"missing_uids={len(missing_uids)}; "
                f"failed={sorted(failed_uids)[:5]}, missing={sorted(missing_uids)[:5]}"
            )

    tq.kv_clear(partition_id=partition_id, keys=selected_prompt_uids)
    return KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)


def require_complete_training_groups(self, partition_id: str, selected_prompt_uids, partition_snapshot):
    # Refill failed or filtered groups using the configured synchronous generation budget.
    if partition_id != "val":
        selected_uids = set(selected_prompt_uids)
        materialized_uids = selected_uids & {key.split("_")[0] for key in partition_snapshot}
        failed_uids = selected_uids & self.failure_keys[partition_id]
        missing_uids = selected_uids - materialized_uids
        if failed_uids or missing_uids:
            message = (
                f"Incomplete synchronous training: expected_uids={len(selected_uids)}, "
                f"materialized_uids={len(materialized_uids)}, failed_uids={len(failed_uids)}, "
                f"missing_uids={len(missing_uids)}; "
                f"failed={sorted(failed_uids)[:5]}, missing={sorted(missing_uids)[:5]}."
            )
            if not self.sync_refill_failed_groups and failed_uids & missing_uids:
                message += (
                    " Enable trainer.v1.sampler.sync_refill_failed_groups to replace failed groups "
                    "with no trajectories; failed groups with outputs still invalidate the batch."
                )
            raise RuntimeError(message)
