"""Public-task calculator prompts shared by text and Dyad evaluation decisions."""
from __future__ import annotations


GSM8K_TEMPLATE_NO_HIS = """
You are an expert agent solving a math problem with a calculator.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s).
You are now at step {current_step} and your current observation is: {current_observation}
Your available actions are: calculate and answer.

Reason about the next step within <think> </think> tags, then output exactly one action.
Use <action>calculate {{expression}}=</action> for arithmetic and wait for the result.
Use <action>answer {{number}}</action> to submit the final numeric answer and end the task.
"""

GSM8K_TEMPLATE = """
You are an expert agent solving a math problem with a calculator.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your available actions are: calculate and answer.

Reason about the next step within <think> </think> tags, then output exactly one action.
Use <action>calculate {{expression}}=</action> for arithmetic and wait for the result.
Use <action>answer {{number}}</action> to submit the final numeric answer and end the task.
"""


def build_gsm8k_messages(
    task_description: str, observation: str, history: list[tuple[str, str]], history_length: int,
) -> list[dict]:
    """Render only the public question and actual calculator decision history.

    Reference solutions and reset/scoring settings are deliberately not inputs.
    History contains each prior observation and its corresponding sampled action;
    hiding history never resets the absolute decision count.
    """
    if not isinstance(task_description, str) or not task_description.strip():
        raise ValueError("GSM8K public task must be nonempty text")
    if not isinstance(observation, str):
        raise ValueError("GSM8K calculator observation must be text")
    if type(history_length) is not int or history_length < 0:
        raise ValueError("GSM8K history length must be a nonnegative integer")
    recent = history[-history_length:] if history_length else []
    context = "\n".join(
        f"[Observation {index}: '{before}', Action {index}: '{action}']"
        for index, (before, action) in enumerate(recent, start=len(history) - len(recent) + 1)
    )
    template = GSM8K_TEMPLATE if recent else GSM8K_TEMPLATE_NO_HIS
    prompt = template.format(
        task_description=task_description, current_observation=observation,
        step_count=len(history), current_step=len(history) + 1,
        history_length=len(recent), action_history=context,
    )
    return [{"role": "user", "content": prompt}]
