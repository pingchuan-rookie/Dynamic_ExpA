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
"""ALFWorld prompts adapted from verl-agent's environments/prompts/alfworld.py."""
from __future__ import annotations


# Adapt the verl-agent templates with a benchmark-neutral task description.
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in an interactive household environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.\x20
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in an interactive household environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.\x20
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


def build_alfworld_prompt(
    task: str, observation: str, available_actions: list[str], history: list[tuple[str, str]],
    history_length: int = 2, *, no_thinking: bool = False,
) -> str:
    """Build the task ReAct prompt, independent of native model thinking mode.

    ``no_thinking`` is retained only for the frozen legacy action-only collector.
    Current shared rollouts always use the reference task protocol.
    """
    template = ALFWORLD_TEMPLATE if history else ALFWORLD_TEMPLATE_NO_HIS
    if no_thinking:
        template = template[:template.index("You should first reason step-by-step")]
        template += "Choose one admissible action and output only <action>your action</action>. Do not output reasoning or explanation.\n"
    admissible = "\n ".join(f"'{action}'" for action in available_actions if action != "help")
    if not history:
        return template.format(current_observation=observation, admissible_actions=admissible)
    recent = history[-history_length:]
    start = len(history) - len(recent) + 1
    context = "\n".join(
        f"[Observation {index}: '{before}', Action {index}: '{action}']"
        for index, (before, action) in enumerate(recent, start=start)
    )
    return template.format(
        task_description=task, step_count=len(history), history_length=len(recent), action_history=context,
        current_step=len(history) + 1, current_observation=observation, admissible_actions=admissible,
    )
