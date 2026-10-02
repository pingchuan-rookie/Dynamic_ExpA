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


"""Shared step training and evaluation within the official V1 stage order.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verl.trainer.ppo.v1.trainer_base import KVBatchMeta


def trainer_close_dataloaders(self):
    """Join owned loader workers before the task actor and its children exit."""
    # DYAD-LOADER-LIFECYCLE: StatefulDataLoader retains its iterator even for
    # nonpersistent workers. Early training exit leaves prefetched workers alive;
    # Ray teardown can kill them before multiprocessing's atexit join runs.
    from contextlib import ExitStack

    iterators = {}
    for loader_name in ("train_dataloader", "val_dataloader"):
        loader = getattr(self, loader_name, None)
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterators[id(iterator)] = iterator
    iterator = getattr(self, "train_dataloader_it", None)
    if iterator is not None:
        iterators[id(iterator)] = iterator
    with ExitStack() as stack:
        for iterator in iterators.values():
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                stack.callback(shutdown)
    self.train_dataloader_it = None
    for loader_name in ("train_dataloader", "val_dataloader"):
        loader = getattr(self, loader_name, None)
        if loader is not None:
            loader._iterator = None


def trainer_init_dataloader(self):
    """Initialize train and validate dataloader."""
    from verl.trainer.ppo.v1.trainer_base import (
        OmegaConf,
        StatefulDataLoader,
        collate_fn,
        create_rl_dataset,
        create_rl_sampler,
        logger,
        open_dict,
    )

    # DYAD: Inference-only evaluation does not read or sample training data.
    # Use the shared dataset/sampler contract and record the consumed generation-batch budget.
    val_only = self.config.trainer.get("val_only", False)
    self.train_dataset = None
    if not val_only:
        self.train_dataset = create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=True,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
    self.val_dataset = create_rl_dataset(
        self.config.data.val_files,
        self.config.data,
        self.tokenizer,
        self.processor,
        is_train=False,
        max_samples=self.config.data.get("val_max_samples", -1),
    )

    # Exact refill counts require single-prompt dataloader fetches.
    filter_groups = self.config.algorithm.get("filter_groups", None)
    dapo_enabled = bool(filter_groups is not None and filter_groups.get("enable", False))
    sync_refill_failed_groups = bool(self.config.trainer.v1.sampler.get("sync_refill_failed_groups", False))
    # DYAD-STEP: official dynamic retries retain a complete source gen_batch.
    # Use the shared dataset/sampler contract and record the consumed generation-batch budget.
    requires_exact_refill = (
        self.trainer_mode != "sync"
        or sync_refill_failed_groups
        or (dapo_enabled and not self._uses_dynamic_step_sampling())
    )
    if requires_exact_refill:
        user_gen_batch_size = self.config.data.get("gen_batch_size", None)
        if user_gen_batch_size not in (None, 1):
            logger.warning(f"data.gen_batch_size={user_gen_batch_size} is overridden to 1.")
        elif user_gen_batch_size is None:
            logger.info("data.gen_batch_size defaulted to 1.")
        with open_dict(self.config):
            self.config.data.gen_batch_size = 1

    # use gen_batch_size as the batch size for the dataloader if set, otherwise use train_batch_size
    gen_batch_size = self.config.data.get("gen_batch_size", None) or self.config.data.train_batch_size
    # Use the shared dataset/sampler contract and record the consumed generation-batch budget.
    self.train_dataloader = None
    if not val_only:
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=gen_batch_size,
            num_workers=self.config.data["dataloader_num_workers"],
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(self.config.data, self.train_dataset),
        )
    self.train_dataloader_it = None
    self.val_dataloader = StatefulDataLoader(
        dataset=self.val_dataset,
        batch_size=self.config.data.val_batch_size or len(self.val_dataset),
        num_workers=self.config.data["dataloader_num_workers"],
        shuffle=self.config.data.get("validation_shuffle", True),
        drop_last=False,
        collate_fn=collate_fn,
    )
    # Use the shared dataset/sampler contract and record the consumed generation-batch budget.
    logger.info(
        f"train and validate dataloader initialized, train dataset size: "
        f"{0 if val_only else len(self.train_dataset)}, val dataset size: {len(self.val_dataset)}"
    )

    self.steps_per_epoch = 0 if val_only else len(self.train_dataset) // self.config.data.train_batch_size

    # adjust total_training_steps
    total_training_steps = self.steps_per_epoch * self.config.trainer.total_epochs
    # Use the shared dataset/sampler contract and record the consumed generation-batch budget.
    if not val_only and self.config.trainer.total_training_steps is not None:
        total_training_steps = self.config.trainer.total_training_steps
    self.total_training_steps = total_training_steps
    logger.info(f"Total training steps: {self.total_training_steps}")

    # The LR scheduler steps once per local update, and each global step performs
    # ``parameter_sync_step`` local updates (see ``PPOTrainer.step``). The optimizer's
    # schedule horizon must therefore count optimizer updates.
    optim_total_training_steps = total_training_steps * self.parameter_sync_step
    try:
        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                self.config.actor_rollout_ref.actor.optim.total_training_steps = optim_total_training_steps
            if OmegaConf.select(self.config, "critic.optim"):
                self.config.critic.optim.total_training_steps = optim_total_training_steps
    except Exception as e:
        logger.warning(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")


def trainer_load_checkpoint(self):
    from verl.trainer.ppo.v1.trainer_base import (
        Role,
        _tq_supports_checkpoint,
        find_latest_ckpt_path,
        logger,
        os,
        tq,
    )

    self.global_steps = 0

    # 1. find latest checkpoint folder
    if self.config.trainer.resume_mode == "disable":
        return
    elif self.config.trainer.resume_mode == "auto":
        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            working_dir = os.getcwd()
            checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
        if global_step_folder is None:
            logger.info("Training from scratch")
            return
    elif self.config.trainer.resume_mode == "resume_path":
        assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
        assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
        global_step_folder = self.config.trainer.resume_from_path
        if not os.path.isabs(global_step_folder):
            working_dir = os.getcwd()
            global_step_folder = os.path.join(working_dir, global_step_folder)
    else:
        logger.exception(f"Unknown resume mode {self.config.trainer.resume_mode}")

    # DYAD-STEP: optimizer/dataloader state must never cross training protocols silently.
    # Validate the saved training and dataset protocol before restoring model and sampler state.
    if self.config.get("algorithm", {}).get("step_rollout", {}).get("enabled", False):
        from verl_extensions.agent_steps.protocol import validate_training_resume

        load_contents = self.config.actor_rollout_ref.actor.checkpoint.get("load_contents", [])
        weights_only = set(load_contents) == {"model"} and not self.config.trainer.get("val_only", False)
        validate_training_resume(self.config, global_step_folder, weights_only=weights_only)
        if weights_only:
            self.actor_rollout_wg.load_checkpoint(
                local_path=os.path.join(global_step_folder, "actor"),
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )
            logger.info(f"Imported actor weights from {global_step_folder}; starting a new training run")
            return

    # set global step
    self.global_steps = int(global_step_folder.split("global_step_")[-1])
    logger.info(f"Resuming from {global_step_folder}, setting global step to {self.global_steps}")

    # DYAD-RESUME: verify dataset/filter identity before loading weights.
    # Validate the saved training and dataset protocol before restoring model and sampler state.
    from verl_extensions.dataset.resume import load_dataloader_checkpoint

    dataloader_state = None
    if not self.config.trainer.get("val_only", False):
        dataloader_state = load_dataloader_checkpoint(
            self.train_dataloader, os.path.join(global_step_folder, "data.pt")
        )

    # 2. load actor checkpoint
    self.actor_rollout_wg.load_checkpoint(
        local_path=os.path.join(global_step_folder, "actor"),
        del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
    )

    # 3. load critic checkpoint
    if self.use_critic:
        self.critic_wg.load_checkpoint(
            local_path=os.path.join(global_step_folder, str(Role.Critic)),
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

    # 4. load dataloader checkpoint
    # Validate the saved training and dataset protocol before restoring model and sampler state.
    if dataloader_state is not None:
        self.train_dataloader.load_state_dict(dataloader_state)

    # 5. restore TransferQueue state (async modes). Re-issuing the restored in-flight prompts is
    # deferred to fit() to use the agent_loop_manager.
    if self.trainer_mode != "sync" and _tq_supports_checkpoint():
        tq_ckpt_path = os.path.join(global_step_folder, "transfer_queue")
        if os.path.exists(tq_ckpt_path):
            logger.info(f"Loading TransferQueue state from {tq_ckpt_path}")
            tq.load_checkpoint(tq_ckpt_path)


def trainer_validate(self) -> dict[str, float]:
    # Lists to collect samples for the table
    # DYAD-EVAL: DIVE also records planned_episodes for provenance, but uses
    # ordinary step rewards. Only explicit official wire fields select this path.
    # Retain complete environment episodes, source identities, and benchmark scoring denominators.
    from verl.trainer.ppo.v1.trainer_base import (
        defaultdict,
        np,
        tq,
        tu,
        uuid,
    )

    episode_field = getattr(self.val_dataset, "episode_field", None)
    tau_evaluation = episode_field in {"tau_episode", "swebench_episode"}
    tau_results = []
    if tau_evaluation and self.config.actor_rollout_ref.rollout.val_kwargs.n != 1:
        raise ValueError("Tau trials are expanded by the dataset; val_kwargs.n must be 1")
    sample_uids = []
    sample_inputs = []
    sample_outputs = []
    sample_gts = []
    sample_scores = []
    sample_turns = []
    data_sources = []
    reward_extra_infos_dict: dict[str, list] = defaultdict(list)
    dump_all_inputs: list[str] = []
    dump_all_outputs: list[str] = []
    dump_all_keys: list[str] = []
    # DYAD-STEP: retain decision evidence without feeding dictionaries into metric reducers.
    # Retain complete environment episodes, source identities, and benchmark scoring denominators.
    shared_steps = self.config.get("algorithm", {}).get("step_rollout", {}).get("enabled", False)
    dump_all_step_records: list[dict] = []
    session_to_sample_idx: dict[str, int] = {}
    # DYAD-STEP: lengths of every sampled decision, not only each session's final output.
    # Retain complete environment episodes, source identities, and benchmark scoring denominators.
    response_lengths: list[int] = []

    for batch_dict in self.val_dataloader:
        # 1. put batch to agent loop manager
        # Retain complete environment episodes, source identities, and benchmark scoring denominators.
        if "uid" not in batch_dict:
            batch_dict["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object
            )
        # DYAD-VALIDATION: Snapshot the submitted population before conversion or worker mutation.
        expected_uids = list(batch_dict["uid"])
        if len(set(expected_uids)) != len(expected_uids):
            raise RuntimeError("Incomplete validation: submitted prompt uids must be unique within a batch")
        rollout_counts = batch_dict.get(
            "__rollout_n__", [self.config.actor_rollout_ref.rollout.val_kwargs.n] * len(expected_uids)
        )
        expected_sessions = set()
        for uid, count in zip(expected_uids, rollout_counts, strict=True):
            if int(count) != count or count <= 0:
                raise RuntimeError(f"Incomplete validation: invalid rollout count {count!r} for uid={uid}")
            expected_sessions.update((uid, str(session_id)) for session_id in range(int(count)))

        # DYAD: Keep independent public metadata across conversion and generation.
        if tau_evaluation:
            from copy import deepcopy

            tau_plan = {
                ep["episode_id"]: deepcopy({k: v for k, v in ep.items() if k != "session_config"})
                # DYAD-SWE: explicit dataset wire key, not a tau identity alias.
                for ep in batch_dict[episode_field]
            }
        batch = tu.get_tensordict(batch_dict)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        tu.assign_non_tensor_data(batch, "validate", True)
        # Register each prompt (GRPO group) in TransferQueue as a tag-only status marker.
        # global_steps is required by ReplayBuffer's metadata sync / staleness ordering.
        tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in range(len(batch))]
        tq.kv_batch_put(keys=list(batch["uid"]), partition_id="val", tags=tags)
        self.agent_loop_manager.generate_sequences(batch)

        # 2. sample batch from replay buffer: one prompt (GRPO group) per submitted row.
        batch, _ = self.replay_buffer.sample(global_steps=self.global_steps, partition_id="val", batch_size=len(batch))

        # DYAD-VALIDATION: A custom sampler or partial session must not alter metric denominators.
        # Count sessions, not outputs: multi-output sessions are legitimate and scored below by final output.
        # Retain complete environment episodes, source identities, and benchmark scoring denominators.
        actual_sessions = set()
        for key in batch.keys:
            parts = key.rsplit("_", 2)
            if len(parts) != 3 or not parts[2].isdigit():
                raise RuntimeError(f"Incomplete validation: malformed trajectory key {key!r}")
            actual_sessions.add((parts[0], parts[1]))
        if actual_sessions != expected_sessions:
            raise RuntimeError(
                f"Incomplete validation: expected_uids={len(expected_uids)}, "
                f"actual_uids={len({uid for uid, _ in actual_sessions})}, "
                f"expected_sessions={len(expected_sessions)}, actual_sessions={len(actual_sessions)}; "
                f"missing={sorted(expected_sessions - actual_sessions)[:5]}, "
                f"unexpected={sorted(actual_sessions - expected_sessions)[:5]}"
            )

        # 3. [OPTIONAL] compute reward score with colocated reward model
        if self.reward_loop_manager.reward_loop_worker_handles is None:
            self.checkpoint_manager.sleep_replicas()
            batch = self._compute_reward_colocate(batch)
            self.checkpoint_manager.update_weights()

        # 4. collect necessary data for logging
        # For multi-output agent loops, only use the final output per session for metrics.
        # Keys have format {uid}_{session_id}_{index}; keep only the highest index per session.
        session_max: dict[str, tuple[int, int]] = {}  # session_key -> (max_index, position)
        for pos, key in enumerate(batch.keys):
            parts = key.rsplit("_", 2)
            if len(parts) == 3:
                session_key = f"{parts[0]}_{parts[1]}"
                index = int(parts[2])
                if session_key not in session_max or index > session_max[session_key][0]:
                    session_max[session_key] = (index, pos)
            else:
                session_max[key] = (0, pos)
        sorted_sessions = sorted(session_max.items(), key=lambda x: x[1][1])
        final_indices = [pos for _, (_, pos) in sorted_sessions]
        final_keys = [batch.keys[i] for i in final_indices]
        base_offset = len(sample_scores)
        session_to_sample_idx.update(
            {session_key: base_offset + j for j, (session_key, _) in enumerate(sorted_sessions)}
        )

        # Retain complete environment episodes, source identities, and benchmark scoring denominators.
        text_fields = ["prompts", "responses"]
        if shared_steps and self.config.trainer.get("validation_data_dir"):
            text_fields.append("extra_fields")
        text_data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=text_fields)
        if "extra_fields" in text_data:
            dump_all_step_records.extend(extra["environment_step"] for extra in text_data.pop("extra_fields").tolist())
        # DYAD-STEP: measure before padding; unbind covers jagged and strided nested layouts.
        response_lengths.extend(row.numel() for row in text_data["responses"].unbind(0))
        text_data["prompts"] = text_data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
        text_data["responses"] = text_data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
        all_inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["prompts"]]
        all_outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["responses"]]

        fields = ["uid", "rm_scores", "num_turns", "reward_model", "data_source", "extra_fields"]
        data = tq.kv_batch_get(keys=final_keys, partition_id=batch.partition_id, select_fields=fields)
        # DYAD-VALIDATION: Metrics group by payload uid, which must agree with the validated keys.
        # get() preserves NonTensorStack; indexed access unwraps it to a LinkedList without tolist().
        # Retain complete environment episodes, source identities, and benchmark scoring denominators.
        payload_uids = data.get("uid").tolist()
        if payload_uids != [key.rsplit("_", 2)[0] for key in final_keys]:
            raise RuntimeError("Incomplete validation: trajectory payload uids do not match requested keys")

        if tau_evaluation:
            for uid, extra in zip(payload_uids, data.get("extra_fields").tolist(), strict=True):
                result = extra.get("episode_result") if isinstance(extra, dict) else None
                if not isinstance(result, dict):
                    raise RuntimeError("Tau rollout did not return episode_result")
                episode = tau_plan[uid]
                # DYAD-SWE: a worker result cannot rewrite the planned task/seed identity.
                if getattr(self.val_dataset, "episode_field", None) == "swebench_episode":
                    from agent_system.environments.backends.swebench.metrics import IDENTITY_FIELDS

                    if any(result.get(key) != episode[key] for key in IDENTITY_FIELDS):
                        raise RuntimeError("SWE-bench result identity differs from the planned episode")
                record = {**episode, **result}
                if extra.get("dyad_action_context") is not None:
                    record["dyad_action_context"] = extra["dyad_action_context"]
                tau_results.append(record)
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
            continue

        sample_uids.extend(data.pop("uid").tolist())
        sample_outputs.extend(all_outputs[i] for i in final_indices)
        sample_inputs.extend(all_inputs[i] for i in final_indices)
        # DYAD-VALIDATION: Both jagged and strided TQ layouts must yield one score per trajectory.
        # Retain complete environment episodes, source identities, and benchmark scoring denominators.
        rm_scores = data["rm_scores"]
        scores = (
            [row.sum().item() for row in rm_scores.unbind(0)] if rm_scores.is_nested else rm_scores.sum(dim=1).tolist()
        )
        # DYAD-GIGPO: an empty final response has no token slot for its episode score.
        if self.config.get("algorithm", {}).get("step_rollout", {}).get("enabled", False):
            scores = [float(extra["environment_step"]["episode_reward"]) for extra in data.get("extra_fields").tolist()]
        sample_scores.extend(scores)
        sample_turns.extend(data.pop("num_turns").tolist())
        reward_extra_infos_dict["reward"].extend(scores)

        extra_fields_list = data.pop("extra_fields", None)
        if extra_fields_list is not None:
            n_prior = len(reward_extra_infos_dict["reward"]) - len(extra_fields_list.tolist())
            for extra_field in extra_fields_list.tolist():
                reward_extra_info = extra_field.get("reward_extra_info", {}) if isinstance(extra_field, dict) else {}
                for key in reward_extra_infos_dict:
                    if key != "reward" and key not in reward_extra_info:
                        reward_extra_infos_dict[key].append(None)
                for key, value in reward_extra_info.items():
                    if key not in reward_extra_infos_dict:
                        reward_extra_infos_dict[key] = [None] * n_prior
                    reward_extra_infos_dict[key].append(value)
                n_prior += 1

        reward_model = data.pop("reward_model", None)
        if reward_model is not None:
            sample_gts.extend([item.get("ground_truth", None) for item in reward_model.tolist()])
        else:
            sample_gts.extend([None] * len(final_indices))

        data_source = data.pop("data_source", None)
        if data_source is not None:
            data_sources.extend(data_source.tolist())
        else:
            data_sources.extend(["unknown"] * len(final_indices))

        dump_all_inputs.extend(all_inputs)
        dump_all_outputs.extend(all_outputs)
        dump_all_keys.extend(batch.keys)

        # 5. cleanup transfer queue
        tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

    # logger to wandb
    # Retain complete environment episodes, source identities, and benchmark scoring denominators.
    if tau_evaluation:
        # DYAD-SWE: benchmark-owned official metrics retain the full denominator.
        if getattr(self.val_dataset, "episode_field", None) == "swebench_episode":
            from agent_system.environments.backends.swebench.metrics import finish_swebench_validation

            return finish_swebench_validation(self, tau_results)
        from agent_system.environments.backends.tau.metrics import finish_tau_validation

        return finish_tau_validation(self, tau_results)

    self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

    # dump to local dir
    val_data_dir = self.config.trainer.get("validation_data_dir", None)
    if val_data_dir:
        # Sort according to uid (so that generations in the same rollout are together)
        sort_keys = []
        for key in dump_all_keys:
            parts = key.rsplit("_", 2)
            sort_keys.append((parts[0], int(parts[1]), int(parts[2])) if len(parts) == 3 else (key, 0, 0))
        sorted_indices = sorted(range(len(dump_all_keys)), key=lambda i: sort_keys[i])
        dump_all_inputs = [dump_all_inputs[i] for i in sorted_indices]
        dump_all_outputs = [dump_all_outputs[i] for i in sorted_indices]
        dump_all_keys = [dump_all_keys[i] for i in sorted_indices]

        # For ground truths, scores and reward extra infos, find the values in the
        # lists for the final samples of each session
        dump_all_sessions = [
            f"{parts[0]}_{parts[1]}" if len(parts) == 3 else key
            for key in dump_all_keys
            for parts in [key.rsplit("_", 2)]
        ]
        session_final_indices = [session_to_sample_idx[session] for session in dump_all_sessions]
        # Retain complete environment episodes, source identities, and benchmark scoring denominators.
        self._dump_generations(
            inputs=dump_all_inputs,
            outputs=dump_all_outputs,
            gts=[sample_gts[i] for i in session_final_indices],
            scores=[sample_scores[i] for i in session_final_indices],
            reward_extra_infos_dict={
                k: [v[i] for i in session_final_indices] for k, v in reward_extra_infos_dict.items()
            }
            | {"uid": dump_all_keys}
            | ({"environment_step": [dump_all_step_records[i] for i in sorted_indices]} if shared_steps else {}),
            dump_path=val_data_dir,
        )

    metric_dict = self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)
    # DYAD-STEP: only this path; tau keeps one final row per episode, not every decision.
    metric_dict.update(self._val_response_length_metrics(response_lengths))
    return metric_dict


def trainer_val_response_length_metrics(self, response_lengths) -> dict[str, float]:
    # DYAD-STEP: clip against the configured per-decision budget, as training metrics do.
    from verl.trainer.ppo.v1.trainer_base import (
        np,
    )

    if not response_lengths:
        return {}
    lengths = np.asarray(response_lengths)
    budget = self.config.actor_rollout_ref.rollout.response_length
    return {
        "val-aux/response_length/mean": float(lengths.mean()),
        "val-aux/response_length/max": int(lengths.max()),
        "val-aux/response_length/min": int(lengths.min()),
        "val-aux/response_length/clip_ratio": float((lengths >= budget).mean()),
    }


def trainer_log_rollout_data(self, batch: KVBatchMeta, timing_raw: dict, rollout_data_dir: str):
    """Fetch rollout data from TransferQueue and dump sorted by uid."""
    from verl.trainer.ppo.v1.trainer_base import (
        marked_timer,
        tq,
    )

    with marked_timer("dump_rollout_generations", timing_raw, color="green"):
        # DYAD-STEP: dump environment decisions once, not statistical copies or dummies.
        # Export original environment decisions without treating statistical copies as interactions.
        batch = batch.select_keys(
            [
                key
                for key, tag in zip(batch.keys, batch.tags, strict=True)
                if not tag.get("is_padding", False) and not tag.get("is_statistical_copy", False)
            ]
        )
        if not len(batch):
            return
        fields = ["uid", "prompts", "responses", "rm_scores", "reward_model"]
        # Export original environment decisions without treating statistical copies as interactions.
        shared_steps = self.config.get("algorithm", {}).get("step_rollout", {}).get("enabled", False)
        if shared_steps:
            fields.append("extra_fields")
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        # Export original environment decisions without treating statistical copies as interactions.
        step_records = (
            [extra["environment_step"] for extra in data.pop("extra_fields").tolist()] if shared_steps else None
        )
        data["prompts"] = data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
        data["responses"] = data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)

        uids = data.pop("uid").tolist()
        inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in data["prompts"]]
        outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in data["responses"]]
        # Export original environment decisions without treating statistical copies as interactions.
        scores = (
            [record["episode_reward"] for record in step_records]
            if shared_steps
            else [row.sum().item() for row in data["rm_scores"].unbind(0)]
        )

        reward_model = data.pop("reward_model", None)
        if reward_model is not None:
            gts = [item.get("ground_truth", None) for item in reward_model.tolist()]
        else:
            gts = [None] * len(uids)

        # Sort by uid key ({sample}_{rollout}_{output})
        sort_keys = []
        for key in batch.keys:
            parts = key.rsplit("_", 2)
            if len(parts) == 3:
                sort_keys.append((parts[0], int(parts[1]), int(parts[2])))
            else:
                sort_keys.append((key, 0, 0))
        sorted_indices = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])

        inputs = [inputs[i] for i in sorted_indices]
        outputs = [outputs[i] for i in sorted_indices]
        gts = [gts[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]

        reward_extra_infos_dict = {"uid": [batch.keys[i] for i in sorted_indices]}
        # Export original environment decisions without treating statistical copies as interactions.
        if shared_steps:
            reward_extra_infos_dict["environment_step"] = [step_records[i] for i in sorted_indices]

        self._dump_generations(
            inputs=inputs,
            outputs=outputs,
            gts=gts,
            scores=scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=rollout_data_dir,
        )


def trainer_balance_batch(self, batch: KVBatchMeta, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
    """Reorder the data on single controller such that each dp rank gets similar total tokens."""
    from verl.trainer.ppo.v1.trainer_base import (
        calculate_workload,
        get_seqlen_balanced_partitions,
        log_seqlen_unbalance,
        torch,
        upsample_batch_to_divisible_size,
    )

    # Preserve reference step ordering and balance using the shared full-sequence-length contract.
    dp_size = self._get_actor_dp_size()

    # Upsampling the batch with padding sequences
    batch_multiple = self._get_required_batch_multiple(dp_size)
    batch = upsample_batch_to_divisible_size(batch, batch_multiple, self.tokenizer.eos_token_id)
    global_seqlen_lst = torch.tensor([tag["seq_len"] for tag in batch.tags], dtype=torch.int64)
    # DYAD-STEP: reference advantage ordering and actor partitions use raw lengths.
    # Preserve reference step ordering and balance using the shared full-sequence-length contract.
    shared_steps = self.config.get("algorithm", {}).get("step_rollout", {}).get("enabled", False)
    workload_lst = global_seqlen_lst.tolist() if shared_steps else calculate_workload(global_seqlen_lst)

    # reorder based on index. The data will be automatically equally partitioned by dispatch function
    if shared_steps and not self.config.trainer.get("balance_batch", True):
        rank_size = len(batch) // dp_size
        global_partition_lst = [list(range(start, start + rank_size)) for start in range(0, len(batch), rank_size)]
    else:
        global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
    batch.reorder([j for partition in global_partition_lst for j in partition])
    global_balance_stats = log_seqlen_unbalance(
        seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
    )
    metrics.update(global_balance_stats)
    return batch


def trainer_compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
    """Compute the advantage of the batch."""
    from verl.trainer.ppo.v1.trainer_base import (
        DataProto,
        TensorDict,
        apply_kl_penalty,
        compute_advantage_for_multi_trajectories,
        compute_rollout_correction_and_add_to_batch,
        np,
        response_to_nested,
        tq,
    )

    fields = ["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob", "values"]
    # DYAD-GIGPO: TQ stores step metadata, but it must not enter tensor padding.
    # Carry step identities and masks into the shared GRPO/GiGPO adapter; retain the official non-step path.
    gigpo = self.config.algorithm.adv_estimator == "gigpo"
    shared_steps = self.config.algorithm.get("step_rollout", {}).get("enabled", False)
    # DYAD-STEP: reject retired layouts before reading rollout metadata.
    if gigpo and not shared_steps:
        raise ValueError("GiGPO requires shared step protocol v2")
    if shared_steps:
        fields.append("extra_fields")
    if shared_steps and self.config.actor_rollout_ref.actor.strategy == "dyad":
        fields.append("seq_mask")
    data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
    # Carry step identities and masks into the shared GRPO/GiGPO adapter; retain the official non-step path.
    step_records = None
    if shared_steps:
        metadata_key = "environment_step"
        extra_fields = data.pop("extra_fields").tolist()
        step_records = np.empty(len(batch), dtype=object)
        for i, (extra, tag) in enumerate(zip(extra_fields, batch.tags, strict=True)):
            if tag.get("is_padding", False):
                step_records[i] = None
            elif not isinstance(extra, dict) or metadata_key not in extra:
                raise ValueError(f"GiGPO requires {metadata_key} from every real rollout")
            else:
                step_records[i] = extra[metadata_key]

    response_mask = data["response_mask"]
    data = DataProto(batch=data.to_padded_tensor())
    # Carry step identities and masks into the shared GRPO/GiGPO adapter; retain the official non-step path.
    if shared_steps:
        data.non_tensor_batch[metadata_key] = step_records
        data.non_tensor_batch["is_padding"] = np.array([tag.get("is_padding", False) for tag in batch.tags], dtype=bool)
    if shared_steps:
        data.non_tensor_batch["step_source_key"] = np.array(
            [tag.get("step_source_key", key) for key, tag in zip(batch.keys, batch.tags, strict=True)], dtype=object
        )
        data.non_tensor_batch["step_occurrence"] = np.array(
            [tag.get("step_occurrence", -1) for tag in batch.tags], dtype=np.int64
        )
    data.batch["token_level_scores"] = data.batch["rm_scores"]
    data.non_tensor_batch["uid"] = np.array(data.batch.pop("uid").tolist(), dtype=object)

    # 1. apply kl penalty to rewards
    if self.config.algorithm.use_kl_in_reward:
        data, kl_metrics = apply_kl_penalty(
            data, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
        )
        metrics.update(kl_metrics)
    else:
        data.batch["token_level_rewards"] = data.batch["token_level_scores"]

    # 2. Compute rollout correction: IS weights, rejection sampling, and metrics
    # Only runs in decoupled mode (computes once per batch using stable π_old)
    # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
    rollout_correction = (
        rollout_corr_config is not None and "rollout_log_probs" in data.batch and not bypass_recomputing_logprobs
    )
    if rollout_correction:
        data, is_metrics = compute_rollout_correction_and_add_to_batch(data, rollout_corr_config)
        metrics.update(is_metrics)

    # 3. compute advantages
    data = compute_advantage_for_multi_trajectories(
        data,
        batch_keys=batch.keys,
        adv_estimator=self.config.algorithm.adv_estimator,
        gamma=self.config.algorithm.gamma,
        lam=self.config.algorithm.lam,
        num_repeat=self.config.actor_rollout_ref.rollout.n,
        norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
        config=self.config.algorithm,
    )
    # DYAD-GIGPO: aggregate diagnostics before converting back to nested tensors.
    # Carry step identities and masks into the shared GRPO/GiGPO adapter; retain the official non-step path.
    metrics.update(data.meta_info.pop("gigpo_metrics", {}))
    metrics.update(data.meta_info.pop("step_metrics", {}))

    # 4. write nested advantages and returns back to TransferQueue
    fields = ["advantages", "returns"]
    if self.config.algorithm.use_kl_in_reward:
        fields.append("token_level_rewards")
    if rollout_correction:
        fields.append("response_mask")
        if "rollout_is_weights" in data.batch:
            fields.append("rollout_is_weights")

    output = {}
    for field in fields:
        output[field] = response_to_nested(data.batch[field], response_mask)
    output = TensorDict(output, batch_size=len(batch))

    batch = tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)

    return batch


def trainer_update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
    """Update the actor network."""
    from verl.trainer.ppo.v1.trainer_base import (
        TensorDict,
        is_distillation_enabled,
        reduce_metrics,
        rename_dict,
        torch,
        tq,
    )

    ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
    # DYAD-GIGPO-STEP: step budgets count real decisions, not task groups.
    # Dispatch real step mini-batches within the existing actor-update phase and retain one rollout anchor.
    if not self._actor_uses_step_minibatches():
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
    calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
        self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
    )
    distillation_use_topk = (
        self.distillation_config.distillation_loss.loss_settings.use_topk
        if is_distillation_enabled(self.config.get("distillation"))
        else False
    )
    distillation_only = False  # distillation_only flag means we can skip policy loss and reduce mem footprint
    if is_distillation_enabled(self.config.get("distillation")):
        distillation_loss_cfg = self.distillation_config.distillation_loss
        distillation_only = (
            distillation_use_topk
            and not distillation_loss_cfg.use_task_rewards
            and not distillation_loss_cfg.use_policy_gradient
        )
    extra_info = {
        "calculate_entropy": calculate_entropy,
        "distillation_use_topk": distillation_use_topk,
        "distillation_only": distillation_only,
        "global_batch_size": ppo_mini_batch_size,
        "mini_batch_size": ppo_mini_batch_size,
        "epochs": self.config.actor_rollout_ref.actor.ppo_epochs,
        "seed": self.config.actor_rollout_ref.actor.data_loader_seed,
        "dataloader_kwargs": {"shuffle": self.config.actor_rollout_ref.actor.shuffle},
        "temperature": self.config.actor_rollout_ref.rollout.temperature,
    }
    batch.extra_info.update(extra_info)

    # DYAD-STEP: the engine selects the versioned objective reduction explicitly.
    # Dispatch real step mini-batches within the existing actor-update phase and retain one rollout anchor.
    step_settings = self.config.get("algorithm", {}).get("step_rollout", {})
    if step_settings.get("enabled", False):
        extra_info["environment_step_loss_reduction"] = step_settings.get("loss_reduction", "reference_microbatch")
        # DYAD-STEP: preserve the configured budget when tail dispatch overwrites actual sizes.
        extra_info["environment_step_nominal_mini_batch_size"] = ppo_mini_batch_size
        extra_info["environment_step_response_length"] = self.config.actor_rollout_ref.rollout.response_length
        batch.extra_info.update(extra_info)
        # Real resampled occurrences participate; transport-only rows do not.
        batch_info = dict(batch.extra_info)
        batch = tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(
                {
                    "environment_step_is_padding": torch.tensor(
                        [tag.get("is_padding", False) for tag in batch.tags], dtype=torch.bool
                    ),
                },
                batch_size=len(batch),
            ),
        )
        # The returned snapshot includes the new field in worker fetches.
        batch.extra_info.update(batch_info)

    # DYAD-GIGPO-STEP: do not let globally balanced padding split real minis.
    # Old/ref log-probs were computed once before this call; weight sync stays
    # at the existing rollout boundary after every mini has completed.
    if self._actor_uses_step_minibatches():
        from verl_extensions.agent_steps.minibatch import update_step_minibatches

        output = {"metrics": update_step_minibatches(self, batch, extra_info)}
    else:
        output: TensorDict = self.actor_rollout_wg.update_actor(batch)
    output = rename_dict(output["metrics"], "actor/")
    output["perf/mfu/actor"] = output.pop("actor/mfu")
    actor_metrics = reduce_metrics(output)
    metrics.update(actor_metrics)

    return batch
