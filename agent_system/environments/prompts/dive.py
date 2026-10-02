"""GiGPO-style DIVE decision prompts retaining the native tool-call protocol."""
from __future__ import annotations

import json


_DIVE_RESPONSE_INSTRUCTIONS = """Reason about the next step within <think> </think> tags.
Then, directly after </think>, call the appropriate tools using the native tool-call format and their declared arguments, and wait for their results.
When ready, write your final answer directly after </think> instead of tool calls, in the format requested by the query.
A response without tool calls ends the task; do not output reasoning alone or text outside the <think> block other than tool calls or the final answer.
"""

DIVE_TEMPLATE_NO_HIS = """You are an expert agent solving a task using the available tools.
Your task is to: {task_description}
Your current observation is: {current_observation}
Your available tools are: {available_actions}
Use these tool schemas to construct calls; the interface specifies the native tool-call syntax.

""" + _DIVE_RESPONSE_INSTRUCTIONS

DIVE_TEMPLATE = """You are an expert agent solving a task using the available tools.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your available tools are: {available_actions}
Use these tool schemas to construct calls; the interface specifies the native tool-call syntax.

""" + _DIVE_RESPONSE_INSTRUCTIONS


def build_dive_messages(
    query: str, tools: list[dict] | None = None, *,
    observation: str = "No tool results yet.",
    history: list[tuple[str, str]] | tuple = (), history_length: int = 2,
) -> list[dict[str, str]]:
    """Render only public task state; schemas remain native chat-template tools.

    History entries pair the observation before a decision with its sampled native
    response. They are textual context, not standalone tool replies. The current
    observation always contains the latest feedback, even when history is disabled.
    """
    if type(history_length) is not int or history_length < 0:
        raise ValueError("DIVE history_length must be a nonnegative integer")
    recent = history[-history_length:] if history_length else []
    context = "\n".join(
        f"[Observation {index}: '{before}', Action {index}: '{action}']"
        for index, (before, action) in enumerate(recent, start=len(history) - len(recent) + 1)
    )
    template = DIVE_TEMPLATE if recent else DIVE_TEMPLATE_NO_HIS
    content = template.format(
        task_description=query, current_observation=observation,
        available_actions=json.dumps(tools or [], ensure_ascii=False),
        step_count=len(history), history_length=len(recent), action_history=context,
        current_step=len(history) + 1,
    )
    return [{"role": "user", "content": content}]
