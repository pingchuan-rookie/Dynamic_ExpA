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


"""Validate shared step training and evaluation protocols at the config boundary.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations


def validate_project_config(config, use_critic: bool):
    # DYAD-SHARED-STEP: one validated architecture for text/Dyad and both estimators.
    from verl_extensions.agent_steps.protocol import shared_step_enabled, validate_step_config

    shared_step = shared_step_enabled(config)
    validate_step_config(config, use_critic=use_critic)
    # DYAD-STEP: evaluation uses v2 decisions without enabling training.
    step = config.algorithm.get("step_rollout", {})
    step_evaluation = config.trainer.val_only and step.get("evaluation_only", False)
    rollout = config.actor_rollout_ref.rollout
    if step_evaluation:
        if (
            step.get("protocol_version") != 2
            or rollout.agent.default_agent_loop != "environment_step_agent"
            or not config.trainer.get("use_v1", True)
            or config.trainer.v1.trainer_mode != "sync"
        ):
            raise ValueError("Step evaluation requires protocol v2, environment_step_agent and V1 sync")
    if config.algorithm.adv_estimator == "gigpo" and not (shared_step or step_evaluation):
        raise ValueError("GiGPO requires shared step protocol v2; old interaction protocols were removed")
    if rollout.agent.default_agent_loop in {"gigpo_step_agent", "dyad_tool_agent"}:
        raise ValueError("Old interaction protocols were removed; use environment_step_agent")
    if "rollout_layout" in (config.algorithm.get("gigpo") or {}):
        raise ValueError("algorithm.gigpo.rollout_layout was removed; use shared step protocol v2")
    actor = config.actor_rollout_ref.actor
    step_minibatches = actor.get("ppo_mini_batch_size_unit", "trajectory") == "step"
    if step_minibatches and not shared_step:
        raise ValueError("Step mini-batches require shared step protocol v2")
    return step_minibatches
