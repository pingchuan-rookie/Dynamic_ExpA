# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
# Copyright 2026 Dyad contributors.
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
"""WebShop prompts adapted from verl-agent's environments/prompts/webshop.py."""
from __future__ import annotations


# Adapt the verl-agent templates with a benchmark-neutral task description.
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in an online shopping environment.\x20
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:\x20
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.\x20
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""
WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in an online shopping environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:\x20
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.\x20
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


def build_webshop_prompt(task, observation, actions, history, history_length):
    """Build the task ReAct prompt, independent of native model thinking mode."""
    recent = history[-history_length:] if history_length else []
    template = WEBSHOP_TEMPLATE if recent else WEBSHOP_TEMPLATE_NO_HIS
    context = "\n".join(
        f"[Observation {index}: '{before}', Action {index}: '{action}']"
        for index, (before, action) in enumerate(recent, start=len(history) - len(recent) + 1)
    )
    values = dict(task_description=task, current_observation=observation,
                  available_actions="\n".join(f"'{action}'," for action in actions),
                  step_count=len(history), history_length=len(recent), action_history=context,
                  current_step=len(history) + 1)
    result = template.format(**values)
    if len(result) > 13000 and recent:
        return build_webshop_prompt(task, observation, actions, [], 0)
    return result
