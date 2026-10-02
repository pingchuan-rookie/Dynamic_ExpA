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

# TODO: move this file to verl.experimental.agent_loop after V1 is stable
"""TransferQueue adapter for AgentLoopManager and AgentLoopWorker"""
# DYAD-UPSTREAM: verl v0.9.0
# Source commit: 483b8a009ba3a97563edee3a19887e4862b8094a
# The official GRPO flow remains the backbone; ADD/REPLACE/REMOVE blocks record local changes.

import asyncio
import logging

# >>> DYAD-ADD(module)
# Use asynchronous completion tracking for bounded environment sessions.
import operator as operator

# <<< DYAD-ADD(module)
import os
from typing import Any

import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData as NonTensorData
from tensordict import NonTensorStack as NonTensorStack
from tensordict import TensorDict

from verl.experimental.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
)
from verl.experimental.agent_loop import (
    get_trajectory_info as get_trajectory_info,
)
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


async def _settle_session_tasks(tasks: list[asyncio.Task[Any]]) -> list[BaseException]:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if isinstance(result, BaseException)]


@ray.remote
class AgentLoopWorkerTQ(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()
        # DYAD-ENV-CAPACITY: budget per dispatched shard, not the global batch on every worker.
        # >>> DYAD-ADD(AgentLoopWorkerTQ.__init__)
        # Track per-worker environment capacity and outstanding sampling sessions.
        self._capacity_lock = asyncio.Lock()
        self._inflight_sessions: dict[int, int] = {}
        # <<< DYAD-ADD(AgentLoopWorkerTQ.__init__)

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Spawn agent loop for each sample in the batch without waiting for the results."""
        # >>> DYAD-REPLACE(AgentLoopWorkerTQ.generate_sequences)
        # verl v0.9.0 / 483b8a009ba3a97563edee3a19887e4862b8094a.
        # Reserve shared environment capacity before dispatching rollout sessions.
        # Original upstream implementation (retained for comparison):
        # validate = batch["validate"] if "validate" in batch else False
        # batch.pop("validate", None)
        # config = self.config.actor_rollout_ref.rollout
        # sampling_params = dict(
        #     temperature=config.temperature,
        #     top_p=config.top_p,
        #     top_k=config.top_k,
        #     repetition_penalty=1.0,
        #     logprobs=config.calculate_log_probs,
        # )
        #
        # # override sampling params for validation
        # if validate:
        #     sampling_params["top_p"] = config.val_kwargs.top_p
        #     sampling_params["top_k"] = config.val_kwargs.top_k
        #     sampling_params["temperature"] = config.val_kwargs.temperature
        #
        # # by default, we assume it's a single turn agent
        # if "agent_name" not in batch:
        #     default_agent_loop = config.agent.default_agent_loop
        #     batch["agent_name"] = NonTensorData(default_agent_loop)
        #
        # trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)
        #
        # # create background tasks for each sample in the batch
        # for i in range(len(batch)):
        #     # TODO(wuxibin): add trace support
        #     trace_this_sample = False
        #     prompt = {}
        #     for k, v in batch.items():
        #         if isinstance(v, torch.Tensor):
        #             prompt[k] = v[i]
        #         elif isinstance(v, NonTensorStack):
        #             prompt[k] = v[i].data
        #         elif isinstance(v, NonTensorData):
        #             prompt[k] = v.data
        #         else:
        #             logger.exception(f"Unsupported type {type(v)} for key {k}")
        #
        #     # “fire-and-forget” background tasks
        #     task = asyncio.create_task(
        #         self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
        #     )
        #     self.background_tasks.add(task)
        #     task.add_done_callback(self.background_tasks.discard)
        from verl_extensions.rollout.transfer_queue import tq_generate_sequences

        return await tq_generate_sequences(self, batch)
        # <<< DYAD-REPLACE(AgentLoopWorkerTQ.generate_sequences)

    def _prompt_finished(self, task: asyncio.Task, sessions: int, pool_ids: tuple[int, ...]) -> None:
        # >>> DYAD-ADD(AgentLoopWorkerTQ._prompt_finished)
        # verl v0.9.0 / 483b8a009ba3a97563edee3a19887e4862b8094a.
        # Reserve shared environment capacity before dispatching rollout sessions.
        from verl_extensions.rollout.transfer_queue import tq_prompt_finished

        return tq_prompt_finished(self, task, sessions, pool_ids)
        # <<< DYAD-ADD(AgentLoopWorkerTQ._prompt_finished)
            # <<< DYAD-REPLACE(AgentLoopWorkerTQ.generate_sequences)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        """Spawn multiple agent loops in parallel according to rollout.n or rollout.val_kwargs.n."""
        uid, partition_id = prompt["uid"], "train" if not trajectory["validate"] else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        tasks = []
        try:
            # NOTE: user can dynamically adjust n for each sample here, e.g according to task difficulty.
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
            do_sample = prompt.pop("__do_sample__", True)

            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)

            tasks = []
            for i in range(n):
                task = asyncio.create_task(
                    self._run_agent_loop(
                        run_sampling_params, trajectory=trajectory, trace=trace, session_id=i, **prompt
                    )
                )
                tasks.append(task)

            # Publish a terminal status only after every session settles, so no sibling can write after
            # ReplayBuffer clears a failed group.
            session_errors = await _settle_session_tasks(tasks)
            if session_errors:
                for error in session_errors:
                    logger.error(
                        f"Error in _run_prompt for uid={uid}",
                        exc_info=(type(error), error, error.__traceback__),
                    )
                status = "failure"
            else:
                status = "finished"
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})
        except Exception as e:
            logger.exception(f"Error in _run_prompt: {e}")
            if tasks:
                await _settle_session_tasks(tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put agent loop outputs into TransferQueue."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        # TODO: Support output:list[AgentLoopOutput]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields["reward_extra_info"]

        # NOTE: agent loop may has multiple outputs, put each output into TransferQueue.
        # key format: {uid}_{session_id}_{index}
        # - uid: raw prompt uid from dataset
        # - session_id: session id for rollout.n sampling
        # - index: index of agent loop output
        keys, fields, tags = [], [], []
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            keys.append(f"{uid}_{session_id}_{i}")
            # DYAD-STEP-TQ: dataset kwargs must not overwrite the sampled policy trace,
            # task context or weight-version metadata carried by this actual decision.
            # >>> DYAD-REPLACE(merge_sampled_metadata)
            # verl v0.9.0 / 483b8a009ba3a97563edee3a19887e4862b8094a.
            # Keep sampled decisions and policy versions authoritative over dataset metadata.
            # Original upstream code:
            # field = output.as_dict()
            # field.update(kwargs)
            from verl_extensions.rollout.transfer_queue import merge_sampled_metadata

            field = merge_sampled_metadata(output, kwargs)
            # <<< DYAD-REPLACE(merge_sampled_metadata)
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            # TODO: uniform response_mask and loss_mask
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                    # These tags are used for off-policy staleness control, if a trajectory
                    # spans too many global steps, we need to filter it out.
                    # global_steps: which global steps this sample is from dataloader
                    "global_steps": kwargs["global_steps"],
                    # min_global_steps: start generation model weights version of this trajectory
                    "min_global_steps": field["extra_fields"].get("min_global_steps"),
                    # max_global_steps: end generation model weights version of this trajectory
                    "max_global_steps": field["extra_fields"].get("max_global_steps"),
                }
            )
            # DYAD-STEP: carry source order across asynchronous session completion.
            # >>> DYAD-ADD(AgentLoopWorkerTQ._agent_loop_postprocess)
            # Preserve sampled decisions, contexts, and weight versions when merging dataset metadata.
            if "step_prompt_order" in kwargs:
                tags[-1].update(
                    step_prompt_order=kwargs["step_prompt_order"],
                    dynamic_sampling_attempt=kwargs.get("dynamic_sampling_attempt", 0),
                )
            # <<< DYAD-ADD(AgentLoopWorkerTQ._agent_loop_postprocess)

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )


class AgentLoopManagerTQ(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = AgentLoopWorkerTQ
        super().__init__(*args, **kwargs)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    def generate_sequences(self, prompts: TensorDict) -> None:
        """
        Dispatch input batch to agent loop workers without blocking. Workers should put agent loop outputs
        into TransferQueue once an agent loop finished.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
            ]
        )
