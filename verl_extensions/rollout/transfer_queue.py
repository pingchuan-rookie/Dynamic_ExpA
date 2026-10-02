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


"""Reserve shared environment capacity before dispatching rollout sessions.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verl.trainer.ppo.v1.agent_loop_tq import TensorDict, asyncio


async def tq_generate_sequences(self, batch: TensorDict) -> None:
    """Spawn agent loop for each sample in the batch without waiting for the results."""
    from verl.trainer.ppo.v1.agent_loop_tq import (
        NonTensorData,
        NonTensorStack,
        asyncio,
        get_trajectory_info,
        logger,
        operator,
        os,
        torch,
    )

    validate = batch["validate"] if "validate" in batch else False
    batch.pop("validate", None)
    config = self.config.actor_rollout_ref.rollout
    sampling_params = dict(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repetition_penalty=1.0,
        logprobs=config.calculate_log_probs,
    )

    # override sampling params for validation
    if validate:
        sampling_params["top_p"] = config.val_kwargs.top_p
        sampling_params["top_k"] = config.val_kwargs.top_k
        sampling_params["temperature"] = config.val_kwargs.temperature

    # by default, we assume it's a single turn agent
    if "agent_name" not in batch:
        default_agent_loop = config.agent.default_agent_loop
        batch["agent_name"] = NonTensorData(default_agent_loop)

    trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)

    # Admit sessions using the assigned shard budget and publish only completed prompt groups.
    prompts = []
    for i in range(len(batch)):
        # Pass trace=False at the capacity-controlled task creation site instead of retaining a separate local flag.
        prompt = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                prompt[k] = v[i]
            elif isinstance(v, NonTensorStack):
                prompt[k] = v[i].data
            elif isinstance(v, NonTensorData):
                prompt[k] = v.data
            else:
                logger.exception(f"Unsupported type {type(v)} for key {k}")
        # Admit sessions using the assigned shard budget and publish only completed prompt groups.
        prompts.append(prompt)

    # DYAD-ENV-CAPACITY: validation can be larger than training and have per-task repeats.
    # Reserve before spawning anything, so a failed reservation cannot leave half a batch live.
    repeats = [
        operator.index(prompt.get("__rollout_n__", config.val_kwargs.n if validate else config.n)) for prompt in prompts
    ]
    if any(n < 1 for n in repeats):
        raise ValueError("Environment rollout repeats must be positive")
    tools = self.tools
    if validate and os.environ.get("DYAD_VAL_TOOL_CONFIG"):
        from verl.tools.tool_registry import initialize_tools_from_config

        tools = initialize_tools_from_config(os.environ["DYAD_VAL_TOOL_CONFIG"])
    pools = {
        id(tool.pool): tool.pool for tool in tools if getattr(tool, "DEFAULT_ENV_TYPE", None) in {"alfworld", "codegym"}
    }
    async with self._capacity_lock:
        for pool_id, pool in pools.items():
            required = self._inflight_sessions.get(pool_id, 0) + sum(repeats)
            if required:
                await pool.ensure_capacity(required)
        for i, (prompt, n) in enumerate(zip(prompts, repeats, strict=True)):
            for pool_id in pools:
                self._inflight_sessions[pool_id] = self._inflight_sessions.get(pool_id, 0) + n
            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=False)
            )
            self.background_tasks.add(task)
            task.add_done_callback(
                lambda done, count=n, pool_ids=tuple(pools): self._prompt_finished(done, count, pool_ids)
            )


def tq_prompt_finished(self, task: asyncio.Task, sessions: int, pool_ids: tuple[int, ...]) -> None:
    self.background_tasks.discard(task)
    for pool_id in pool_ids:
        remaining = self._inflight_sessions[pool_id] - sessions
        if remaining:
            self._inflight_sessions[pool_id] = remaining
        else:
            self._inflight_sessions.pop(pool_id)


def merge_sampled_metadata(output, kwargs: dict):
    # Preserve sampled decisions, contexts, and weight versions when merging dataset metadata.
    sampled = output.as_dict()
    # DYAD-STEP-TQ: reject missing sampled versions before publishing a success row.
    # Dataset metadata cannot establish which weights generated this decision.
    sampled_extra = sampled.get("extra_fields", {})
    if "environment_step" in sampled_extra:
        versions = [sampled_extra.get(key) for key in ("min_global_steps", "max_global_steps")]
        if any(type(version) is not int or version < 0 for version in versions):
            raise ValueError("Environment step output requires authoritative sampled policy versions")
        if versions[0] > versions[1]:
            raise ValueError("Environment step sampled policy versions are out of order")
    field = dict(kwargs)
    field.update(sampled)
    field["extra_fields"] = {**kwargs.get("extra_fields", {}), **sampled.get("extra_fields", {})}
    return field
