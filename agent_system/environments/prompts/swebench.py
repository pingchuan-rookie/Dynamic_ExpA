"""GiGPO-style SWE-bench decisions preserving native tools and patch submission."""
from __future__ import annotations


_SWEBENCH_RESPONSE_INSTRUCTIONS = """Now it's your turn to take the next step.
You should first reason step-by-step about the reported issue, the current evidence, and what to inspect, change, or test next.
This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, make your tool calls directly after </think>, using the native tool-call format and their declared argument schemas.
After making tool calls, stop and wait for their results before continuing. Output no text outside the <think> block and the tool calls.
Use repository contents and local tool results as evidence; do not invent file contents or test outcomes.
When ready, call finish to submit the actual working-tree diff, including new files, for separate evaluation.
If you make multiple tool calls in one response, finish must be the last call.
A plain-text answer or reasoning alone does not submit the patch and is not a valid action.
"""

SWEBENCH_TEMPLATE_NO_HIS = """You are an expert agent fixing a software issue in an offline repository.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s).
You are now at step {current_step} and your current observation is: {current_observation}
Your available tools are: {available_actions}
Their argument schemas and native tool-call format are provided by the interface.
Follow the repository and submission instructions above.

""" + _SWEBENCH_RESPONSE_INSTRUCTIONS

SWEBENCH_TEMPLATE = """You are an expert agent fixing a software issue in an offline repository.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your available tools are: {available_actions}
Their argument schemas and native tool-call format are provided by the interface.
Follow the repository and submission instructions above.

""" + _SWEBENCH_RESPONSE_INSTRUCTIONS


def build_swebench_messages(
    task_description: str, tools: list[dict], *,
    observation: str = "No tool results yet.",
    history: list[tuple[str, str]] | tuple = (), history_length: int = 2,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Render public issue state without replaying the full runtime transcript.

    The caller projects the repository identity and issue into task_description.
    Each history entry pairs the observation before a decision with its actual
    sampled native response. Latest feedback is independent of the history bound.
    Tool schemas remain native chat-template inputs; calls use native syntax after reasoning.
    """
    if type(history_length) is not int or history_length < 0:
        raise ValueError("SWE-bench history_length must be a nonnegative integer")
    if system_prompt is None:
        # The environment package owns the authoritative scaffold. Its initial-message entry
        # passes this value explicitly, avoiding a module-initialization cycle.
        from agent_system.environments.env_package.swebench.tools import SYSTEM_PROMPT
        system_prompt = SYSTEM_PROMPT
    recent = history[-history_length:] if history_length else []
    context = "\n".join(
        f"[Observation {index}: '{before}', Action {index}: '{action}']"
        for index, (before, action) in enumerate(recent, start=len(history) - len(recent) + 1)
    )
    template = SWEBENCH_TEMPLATE if recent else SWEBENCH_TEMPLATE_NO_HIS
    content = template.format(
        task_description=task_description, current_observation=observation,
        available_actions=", ".join(tool["function"]["name"] for tool in tools),
        step_count=len(history), current_step=len(history) + 1,
        history_length=len(recent), action_history=context,
    )
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": content}]
