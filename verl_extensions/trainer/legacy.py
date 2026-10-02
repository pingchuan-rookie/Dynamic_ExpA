# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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


"""Dataset restoration and environment evaluation for the legacy trainer.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verl.trainer.ppo.ray_trainer import Optional, Sampler


def legacy_create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
    """
    Creates the train and validation dataloaders.
    """
    from verl.trainer.ppo.ray_trainer import (
        OmegaConf,
        StatefulDataLoader,
        create_rl_dataset,
        create_rl_sampler,
        open_dict,
    )

    # DYAD: val-only runs have no training dataset or loader.
    val_only = self.config.trainer.get("val_only", False)
    if train_dataset is None and not val_only:
        train_dataset = create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
    if val_dataset is None:
        val_dataset = create_rl_dataset(
            self.config.data.val_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            max_samples=self.config.data.get("val_max_samples", -1),
        )
    self.train_dataset, self.val_dataset = train_dataset, val_dataset

    if train_sampler is None and not val_only:
        train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
    if collate_fn is None:
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

        collate_fn = default_collate_fn

    num_workers = self.config.data["dataloader_num_workers"]

    self.train_dataloader = None
    if not val_only:
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", None) or self.config.data.train_batch_size,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

    val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
    if val_batch_size is None:
        val_batch_size = len(self.val_dataset)

    self.val_dataloader = StatefulDataLoader(
        dataset=self.val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=self.config.data.get("validation_shuffle", True),
        drop_last=False,
        collate_fn=collate_fn,
    )

    assert val_only or len(self.train_dataloader) >= 1, "Train dataloader is empty!"
    assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

    print(
        f"Size of train dataloader: {0 if val_only else len(self.train_dataloader)}, Size of val dataloader: "
        f"{len(self.val_dataloader)}"
    )

    total_training_steps = 0 if val_only else len(self.train_dataloader) * self.config.trainer.total_epochs

    if not val_only and self.config.trainer.total_training_steps is not None:
        total_training_steps = self.config.trainer.total_training_steps

    self.total_training_steps = total_training_steps
    print(f"Total training steps: {self.total_training_steps}")

    try:
        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            if OmegaConf.select(self.config, "critic.optim"):
                self.config.critic.optim.total_training_steps = total_training_steps
    except Exception as e:
        print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")


def legacy_validate(self, merged: bool = False):
    # DYAD-EVAL: Planned task provenance alone (e.g. DIVE) is not an
    # official evaluation wire contract.
    from verl.trainer.ppo.ray_trainer import (
        DataProto,
        defaultdict,
        extract_reward,
        np,
        pad_dataproto_to_divisor,
        unpad_dataproto,
        uuid,
    )

    episode_field = getattr(self.val_dataset, "episode_field", None)
    tau_evaluation = episode_field in {"tau_episode", "swebench_episode"}
    tau_results = []
    if tau_evaluation and self.config.actor_rollout_ref.rollout.val_kwargs.n != 1:
        raise ValueError("Tau trials are expanded by the dataset; val_kwargs.n must be 1")
    data_source_lst = []
    reward_extra_infos_dict: dict[str, list] = defaultdict(list)

    # Lists to collect samples for the table
    sample_inputs = []
    sample_outputs = []
    sample_gts = []
    sample_scores = []
    sample_turns = []
    sample_uids = []

    for test_data in self.val_dataloader:
        test_batch = DataProto.from_single_dict(test_data)

        if "uid" not in test_batch.non_tensor_batch:
            test_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
            )

        # repeat test batch
        test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

        ground_truths = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch]
        sample_gts.extend(ground_truths)

        # DYAD: Preserve public identities before DataProto.pop and rollout mutate the batch.
        if tau_evaluation:
            from copy import deepcopy

            tau_plan = [
                deepcopy({k: v for k, v in episode.items() if k != "session_config"})
                for episode in test_batch.non_tensor_batch[episode_field]
            ]
        test_gen_batch = self._get_gen_batch(test_batch)
        test_gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
            "global_steps": self.global_steps,
        }
        print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

        # pad to be divisible by dp_size
        size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
        test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
        if tau_evaluation:
            from agent_system.environments.backends.tau.metrics import mark_tau_padding

            mark_tau_padding(test_gen_batch_padded, pad_size)
        test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

        if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
            # for colocate reward models, we need to sleep rollout model
            # to spare GPU memory for reward model
            self.checkpoint_manager.sleep_replicas()
            batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
            test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
            # wake up rollout model
            # replace with wake_up method once supported
            self.checkpoint_manager.update_weights(self.global_steps)

        # unpad
        test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
        if tau_evaluation:
            episode_results = test_output_gen_batch.non_tensor_batch.get("episode_result")
            if episode_results is None:
                raise RuntimeError("Tau rollout did not return episode_result")
            contexts = test_output_gen_batch.non_tensor_batch.get("dyad_action_context", [None] * len(tau_plan))
            for episode, result, context in zip(tau_plan, episode_results, contexts, strict=True):
                record = {**episode, **result}
                if context is not None:
                    record["dyad_action_context"] = context
                tau_results.append(record)
            continue

        print("validation generation end")

        # Store generated outputs
        output_ids = test_output_gen_batch.batch["responses"]
        output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        sample_outputs.extend(output_texts)

        test_batch = test_batch.union(test_output_gen_batch)
        test_batch.meta_info["validate"] = True

        # Store original inputs
        input_ids = test_batch.batch["prompts"]
        # TODO: Can we keep special tokens except for padding tokens?
        input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        sample_inputs.extend(input_texts)
        sample_uids.extend(test_batch.non_tensor_batch["uid"])

        # evaluate using reward_function
        reward_tensor, reward_extra_info = extract_reward(test_batch)

        scores = reward_tensor.sum(-1).cpu().tolist()
        sample_scores.extend(scores)

        reward_extra_infos_dict["reward"].extend(scores)
        for key, values in reward_extra_info.items():
            if key not in reward_extra_infos_dict:
                reward_extra_infos_dict[key] = []
            if isinstance(values, np.ndarray):
                reward_extra_infos_dict[key].extend(values.tolist())
            else:
                reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

        # collect num_turns of each prompt
        if "__num_turns__" in test_batch.non_tensor_batch:
            sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

        data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

    if tau_evaluation:
        # DYAD-SWE: Preserve the official scorer associated with the wire field.
        if episode_field == "swebench_episode":
            from agent_system.environments.backends.swebench.metrics import finish_swebench_validation

            return finish_swebench_validation(self, tau_results)
        from agent_system.environments.backends.tau.metrics import finish_tau_validation

        return finish_tau_validation(self, tau_results)

    self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

    # dump generations
    val_data_dir = self.config.trainer.get("validation_data_dir", None)
    if val_data_dir:
        self._dump_generations(
            inputs=sample_inputs,
            outputs=sample_outputs,
            gts=sample_gts,
            scores=sample_scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=val_data_dir,
        )

    for key_info, lst in reward_extra_infos_dict.items():
        assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

    if merged:
        print("_merge_validation_results validate result will be merged")
        return {
            "data_sources": data_source_lst,
            "sample_uids": sample_uids,
            "sample_turns": sample_turns,
            "reward_extra_infos_dict": reward_extra_infos_dict,
        }
    data_sources = np.concatenate(data_source_lst, axis=0)
    return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)


def legacy_load_checkpoint(self):
    from verl.trainer.ppo.ray_trainer import (
        Role,
        find_latest_ckpt_path,
        os,
    )

    if self.config.trainer.resume_mode == "disable":
        return 0

    # load from hdfs
    if self.config.trainer.default_hdfs_dir is not None:
        raise NotImplementedError("load from hdfs is not implemented yet")
    else:
        checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
        if not os.path.isabs(checkpoint_folder):
            working_dir = os.getcwd()
            checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

    # find global_step_folder
    if self.config.trainer.resume_mode == "auto":
        if global_step_folder is None:
            print("Training from scratch")
            return 0
    else:
        if self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
    print(f"Load from checkpoint folder: {global_step_folder}")
    # set global step
    self.global_steps = int(global_step_folder.split("global_step_")[-1])

    print(f"Setting global step to {self.global_steps}")
    print(f"Resuming from {global_step_folder}")

    # DYAD-RESUME: validate even at epoch boundaries, before loading weights.
    from verl_extensions.dataset.resume import load_dataloader_checkpoint

    dataloader_state = None
    if not self.config.trainer.get("val_only", False):
        dataloader_state = load_dataloader_checkpoint(
            self.train_dataloader, os.path.join(global_step_folder, "data.pt")
        )

    actor_path = os.path.join(global_step_folder, "actor")
    critic_path = os.path.join(global_step_folder, str(Role.Critic))
    # load actor
    self.actor_rollout_wg.load_checkpoint(
        actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
    )
    # load critic
    if self.use_critic:
        self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

    # DYAD-RESUME: retain sampler RNG at epoch boundaries too.
    # Saved exhausted iterators advance to the next epoch without replaying it.
    if dataloader_state is not None:
        self.train_dataloader.load_state_dict(dataloader_state)
