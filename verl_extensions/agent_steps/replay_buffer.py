"""Official verl-agent dynamic sampling over complete synchronous V1 episodes.

Unlike V1 DAPO refill, every attempt repeats the same complete source prompt batch.
Each attempt receives fresh group identities; source_prompt_uid remains unchanged.
All variable-return groups are accumulated, including overshoot, and the final
allowed attempt is unfiltered, matching official filter_group_data(last_try=True).
Rollout failures, incomplete episodes and timed-out attempts abort rather than
being mistaken for zero-variance groups or a successful final-attempt fallback.
"""

# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Official verl-agent dynamic sampling over complete synchronous V1 episodes.
# Extension point: PPOTrainer step, advantage, actor-update, and FSDP reduction hooks
# DYAD-AGENT-STEPS: shared synchronous collection retries, independent of action interface.
from __future__ import annotations

import copy
import math
import time
import uuid
from collections import defaultdict

from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, tq
from verl.utils import tensordict_utils as tu


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class StepReplayBuffer(ReplayBuffer):
    """Whole-batch, on-policy retries for shared text/Dyad GRPO and GiGPO.

    The trainer calls ``submit_batch(batch, trainer._submit_batch_to_rollout)``
    instead of submitting its initial training batch directly.
    ``sample`` then owns retries without advancing the dataloader or weights.
    Validation continues to use the ordinary V1 replay implementation.
    """

    def __init__(self, *args, group_n, max_num_gen_batches, attempt_timeout_seconds=1800.0, **kwargs):
        self.group_n = _positive_int("group_n", group_n)
        if self.group_n < 2:
            raise ValueError("Dynamic step groups require group_n >= 2")
        self.max_num_gen_batches = _positive_int("max_num_gen_batches", max_num_gen_batches)
        if (
            isinstance(attempt_timeout_seconds, bool)
            or not isinstance(attempt_timeout_seconds, int | float)
            or not math.isfinite(attempt_timeout_seconds)
            or attempt_timeout_seconds <= 0
        ):
            raise ValueError("attempt_timeout_seconds must be finite and positive")
        self.attempt_timeout_seconds = float(attempt_timeout_seconds)
        super().__init__(*args, **kwargs)
        if self.trainer_mode != "sync":
            raise ValueError("Dynamic step sampling requires synchronous V1 training")
        if self.filter_groups_metric is not None or self.sync_refill_failed_groups:
            raise ValueError("Dynamic step sampling cannot use generic DAPO or failed-group refill")
        if not math.isfinite(self.poll_interval) or self.poll_interval < 0:
            raise ValueError("poll_interval must be finite and nonnegative")
        self._template = None
        self._submit_fn = None
        self._attempt = 0
        self._active_uids = []
        self._source_uids = []
        self._failed = False

    def submit_batch(self, batch, submit_fn):
        """Snapshot one complete gen_batch and dispatch the first attempt.

        Only source identities are retained between attempts, never rollout IDs.
        A deep copy protects the template from worker-side mutation of prompts.
        The dispatch callback must register pending prompt tags before rollout.
        """
        if self._failed:
            raise RuntimeError("Dynamic step sampling previously failed; restart the trainer")
        if self._template is not None:
            raise RuntimeError("Previous dynamic step batch has not been sampled")
        if not len(batch) or not callable(submit_fn):
            raise ValueError("Dynamic sampling requires a nonempty batch and a submit callback")
        sources = list(batch["uid"])
        if any(not isinstance(uid, str) or not uid for uid in sources) or len(set(sources)) != len(sources):
            raise ValueError("Source prompt uids must be unique nonempty strings")
        if "__rollout_n__" in batch:
            counts = list(batch["__rollout_n__"])
            if any(tu.unwrap_non_tensor_data(count) != self.group_n for count in counts):
                raise ValueError("Every dynamic prompt must use the configured group_n")
        self._template = copy.deepcopy(batch)
        self._source_uids = sources
        tu.assign_non_tensor_stack(self._template, "source_prompt_uid", sources)
        self._submit_fn = submit_fn
        self._attempt = 0
        try:
            return self._dispatch_attempt()
        except Exception:
            self._failed = True
            raise

    def _dispatch_attempt(self):
        if self._attempt >= self.max_num_gen_batches:
            raise RuntimeError("Dynamic step sampling exceeded max_num_gen_batches")
        self._attempt += 1
        batch = copy.deepcopy(self._template)
        self._active_uids = [uuid.uuid4().hex for _ in self._source_uids]
        tu.assign_non_tensor_stack(batch, "uid", self._active_uids)
        tu.assign_non_tensor_data(batch, "dynamic_sampling_attempt", self._attempt)
        self._deadline = time.monotonic() + self.attempt_timeout_seconds
        return self._submit_fn(batch)

    def _wait_for_attempt(self, global_steps):
        expected = set(self._active_uids)
        while True:
            self._sync_metadata_from_transfer_queue()
            failed = expected & self.failure_keys["train"]
            if failed:
                raise RuntimeError(f"Dynamic step rollout failure on attempt {self._attempt}: {sorted(failed)[:5]}")
            if expected <= self.finished_keys["train"]:
                versions = self.prompt_global_steps["train"]
                if any(versions[uid] != global_steps for uid in expected):
                    raise RuntimeError("Dynamic step sampling cannot mix policy versions")
                return
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                missing = expected - self.finished_keys["train"]
                raise TimeoutError(
                    f"Dynamic step attempt {self._attempt}/{self.max_num_gen_batches} timed out; "
                    f"unfinished_groups={len(missing)}"
                )
            time.sleep(min(self.poll_interval, remaining))

    def _validated_groups(self):
        """Read each episode score once, validating every original decision row."""
        active = set(self._active_uids)
        keys = []
        grouped = defaultdict(lambda: defaultdict(list))
        for key in self.partitions["train"]:
            parts = key.rsplit("_", 2)
            if len(parts) != 3 or parts[0] not in active:
                continue
            if not parts[1].isdigit() or not parts[2].isdigit():
                raise RuntimeError(f"Malformed dynamic step transport key: {key}")
            keys.append(key)
        if not keys:
            raise RuntimeError("Finished dynamic step groups have no episode rows")
        data = tq.kv_batch_get(
            keys=keys, partition_id="train", select_fields=["uid", "source_prompt_uid", "extra_fields"]
        )
        sources = dict(zip(self._active_uids, self._source_uids, strict=True))
        for key, uid, source, extra in zip(
            keys, data["uid"], data["source_prompt_uid"], data["extra_fields"], strict=True
        ):
            uid, source, extra = (tu.unwrap_non_tensor_data(value) for value in (uid, source, extra))
            group, session, step = key.rsplit("_", 2)
            record = extra.get("environment_step") if isinstance(extra, dict) else None
            if (
                uid != group
                or source != sources[group]
                or not isinstance(record, dict)
                or record.get("group_id") != group
                or record.get("trajectory_id") != f"{group}_{session}"
                or isinstance(record.get("step_index"), bool)
                or record.get("step_index") != int(step)
            ):
                raise RuntimeError(f"Dynamic step identity mismatch: {key}")
            tag = self.partitions["train"][key]
            if tag.get("status") != "success" or tag.get("is_padding") or tag.get("is_statistical_copy"):
                raise RuntimeError(f"Dynamic sampling requires original successful step rows: {key}")
            grouped[group][int(session)].append((int(step), key, record))
        result = []
        for uid in self._active_uids:
            sessions = grouped[uid]
            if set(sessions) != set(range(self.group_n)):
                raise RuntimeError(f"Incomplete dynamic step group {uid}: expected {self.group_n} sessions")
            scores, group_keys = [], []
            for session in range(self.group_n):
                rows = sorted(sessions[session])
                if [row[0] for row in rows] != list(range(len(rows))):
                    raise RuntimeError(f"Incomplete or duplicate steps in dynamic episode {uid}_{session}")
                try:
                    rewards = [float(row[2]["env_reward"]) for row in rows]
                    scores_for_steps = [float(row[2]["episode_reward"]) for row in rows]
                    total = math.fsum(rewards)
                    valid = all(math.isfinite(value) for value in rewards + scores_for_steps)
                    valid &= all(
                        not isinstance(row[2]["episode_length"], bool)
                        and row[2]["episode_length"] == len(rows)
                        and score == scores_for_steps[0]
                        and math.isclose(score, total, rel_tol=1e-6, abs_tol=1e-6)
                        for row, score in zip(rows, scores_for_steps, strict=True)
                    )
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError(f"Invalid dynamic episode score metadata: {uid}_{session}") from exc
                if not valid:
                    raise RuntimeError(f"Incomplete or inconsistent dynamic episode: {uid}_{session}")
                scores.append(scores_for_steps[0])
                group_keys.extend(row[1] for row in rows)
            # Official uses exact inequality, equivalent to nonzero std for finite
            # returns, without floating-point std underflow or cancellation.
            variable = any(score != scores[0] for score in scores[1:])
            result.append((uid, group_keys, variable))
        return result

    def sample(self, global_steps, partition_id, batch_size):
        if partition_id == "val":
            return super().sample(global_steps, partition_id, batch_size)
        if partition_id != "train":
            raise ValueError("Dynamic step sampling only supports train and val partitions")
        if self._failed:
            raise RuntimeError("Dynamic step sampling previously failed; restart the trainer")
        if self._template is None:
            raise RuntimeError("Call StepReplayBuffer.submit_batch before sampling training data")
        if batch_size != len(self._source_uids):
            raise ValueError("Dynamic sampling target must equal the entire source gen_batch size")
        kept_keys, kept_tags = [], []
        kept_groups = filtered_groups = fallback_groups = 0
        try:
            while kept_groups < batch_size:
                self._wait_for_attempt(global_steps)
                groups = self._validated_groups()
                last_try = self._attempt == self.max_num_gen_batches
                rejected = set()
                sources = dict(zip(self._active_uids, self._source_uids, strict=True))
                for uid, keys, variable in groups:
                    if not variable and not last_try:
                        rejected.add(uid)
                        filtered_groups += 1
                        continue
                    kept_groups += 1
                    fallback_groups += int(not variable)
                    kept_keys.extend(keys)
                    source = sources[uid]
                    for key in keys:
                        tag = copy.deepcopy(self.partitions["train"][key])
                        tag.update(source_prompt_uid=source, dynamic_sampling_attempt=self._attempt)
                        kept_tags.append(tag)
                self._clear_groups("train", rejected)
                # Retained rows stay in TQ for the ordinary V1 actor pipeline.
                tq.kv_clear(partition_id="train", keys=[uid for uid in self._active_uids if uid not in rejected])
                if kept_groups < batch_size:
                    self._dispatch_attempt()
        except Exception:
            # Preserve failure evidence; no partial batch can reach the optimizer.
            self._failed = True
            raise
        metrics = {
            "training/filter_groups/num_gen_batches": self._attempt,
            "training/filter_groups/generated_groups": self._attempt * batch_size,
            "training/filter_groups/filtered_groups": filtered_groups,
            "training/filter_groups/retained_groups": kept_groups,
            "training/filter_groups/final_unfiltered_groups": fallback_groups,
            "training/filter_groups/overshoot_groups": kept_groups - batch_size,
        }
        self._template = None
        self._submit_fn = None
        self._active_uids = []
        self._source_uids = []
        return KVBatchMeta(partition_id="train", keys=kept_keys, tags=kept_tags), metrics
