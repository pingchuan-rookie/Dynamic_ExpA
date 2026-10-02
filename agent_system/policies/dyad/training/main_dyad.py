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
"""Dyad entry point for the shared synchronous V1 runtime."""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Validate the Dyad protocol, then delegate to the official V1 task runner.
# Extension point: verl.trainer.main_ppo.run_ppo / TaskRunnerV1
import hydra

from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


@hydra.main(config_path="pkg://verl.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_ppo(config)


def run_ppo(config, task_runner_class=None):
    from verl.trainer.main_ppo import TaskRunnerV1, run_ppo as run_shared_ppo

    step = config.algorithm.get("step_rollout", {})
    evaluation = config.trainer.get("val_only", False) and step.get("evaluation_only", False)
    if not (step.get("enabled", False) or evaluation) or step.get("protocol_version") != 2:
        raise ValueError("Old interaction protocols were removed; use shared step protocol v2")
    if not config.trainer.use_v1 or config.trainer.v1.trainer_mode != "sync":
        raise ValueError("Shared environment-step execution requires the synchronous V1 trainer")
    validate_config(config=config, use_reference_policy=need_reference_policy(config),
                    use_critic=need_critic(config))
    return run_shared_ppo(config, task_runner_class=task_runner_class or TaskRunnerV1)


if __name__ == "__main__":
    main()
