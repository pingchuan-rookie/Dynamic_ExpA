"""Task-preserving CodeGym prompts shared by dataset preparation and rollout."""
from __future__ import annotations

import copy
import re


# Exact compatibility text for prepared action-only datasets, not an active template.
CODEGYM_RESPONSE_INSTRUCTIONS = """========================  CodeGym interaction protocol  ========================
Use only the functions declared for this task to accomplish its goal.
Choose the next call from the current observation and task requirements; do not invent function names, arguments, observations, or results.
Respect the declared argument names, types, and preconditions.
Follow the task's own submission and completion rules; a special inspection or submission function is available only if the task declares it.
After each call, stop and wait for environment feedback before choosing the next call.
If a call fails, use the returned feedback to correct it rather than repeating it unchanged.
At every step, output exactly one function call as a JSON list containing one object, enclosed by these markers:
<|FunctionCallBegin|>[{"name": "<declared function name>", "parameters": {"<argument name>": "<value>"}}]<|FunctionCallEnd|>
Replace the placeholders with the actual function name and arguments; use an empty object for a function with no arguments.
Output only the marker-wrapped JSON, without prose, reasoning, code fences, or executable code."""

# Exact compatibility text for already prepared datasets, not an active template.
_LEGACY_RESPONSE_INSTRUCTIONS = """========================  HOW TO RESPOND (READ CAREFULLY)  ========================
You are NOT allowed to write prose, explanations, reasoning, or code.
At EVERY step you must output EXACTLY ONE function call and nothing else,
using this exact format (a JSON list with one object, wrapped by the markers):

<|FunctionCallBegin|>[{"name": "<FunctionName>", "parameters": {<args>}}]<|FunctionCallEnd|>

Rules:
- Always START by calling Observe to read the current state:
  <|FunctionCallBegin|>[{"name": "Observe", "parameters": {}}]<|FunctionCallEnd|>
- After each call, STOP and wait for the "Observation:" message, then issue the next call.
- When you are confident, submit via the Done function with your final answer.
- Output ONLY the marker-wrapped JSON. No extra words before or after.
=================================================================================="""


_CODEGYM_RESPONSE_INSTRUCTIONS = """Reason about the next step within <think> </think> tags.
Then output exactly one function call using its declared arguments:
<|FunctionCallBegin|>[{{"name": "<function name>", "parameters": {{"<argument name>": "<value>"}}}}]<|FunctionCallEnd|>
Use {{}} for no arguments; do not write executable code.
Wait for feedback after each call and follow the task's submission rules.
"""

CODEGYM_TEMPLATE_NO_HIS = """You are an expert agent operating in an interactive function-calling environment.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s).
You are now at step {current_step} and your current observation is: {current_observation}
Your available functions are: [{available_actions}].

""" + _CODEGYM_RESPONSE_INSTRUCTIONS

CODEGYM_TEMPLATE = """You are an expert agent operating in an interactive function-calling environment.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your available functions are: [{available_actions}].

""" + _CODEGYM_RESPONSE_INSTRUCTIONS

def build_codegym_messages(messages: list[dict]) -> list[dict]:
    """Copy public task source and remove only known, locally owned prompt additions.

    Dataset preparation retains the original public task and function declarations.
    Live decisions are rendered separately by ``build_codegym_step_messages``.
    """
    result = copy.deepcopy(list(messages))
    for system in result:
        if system.get("role") != "system":
            continue
        text = system.get("content", "")
        if not isinstance(text, str):
            raise ValueError("CodeGym system instructions must be text")
        while True:
            for instructions in (CODEGYM_RESPONSE_INSTRUCTIONS, _LEGACY_RESPONSE_INSTRUCTIONS):
                if text == instructions:
                    text = ""
                    break
                if text.endswith("\n\n" + instructions):
                    text = text[:-(len(instructions) + 2)]
                    break
            else:
                break
        system["content"] = text
    return result


def build_codegym_step_messages(
    initial_messages: list[dict], observation: str, history: list[tuple[str, str]],
    history_length: int,
) -> list[dict]:
    """Render a fresh task ReAct decision from public source and live observations.

    The first source user message is the public task, including any official header.
    Source conversation history is not replayed: only the bounded live history is
    rendered. No question-marker heuristics, environment initialization data,
    implementation source, reference answers, or oracle trajectories are needed.
    """
    if history_length < 0:
        raise ValueError("CodeGym history length must be nonnegative")
    messages = build_codegym_messages(initial_messages)
    task = next((message.get("content", "") for message in messages if message.get("role") == "user"), None)
    if not isinstance(task, str) or not task.strip():
        raise ValueError("CodeGym public task instructions must be nonempty text")
    systems = [message for message in messages if message.get("role") == "system"]
    source = "\n".join(message.get("content", "") for message in systems)
    names = tuple(dict.fromkeys(re.findall(r"(?m)^\s*(?:async\s+)?def\s+(\w+)\s*\(", source)))
    if not names:
        raise ValueError("CodeGym task has no public function declarations")
    recent = history[-history_length:] if history_length else []
    context = "\n".join(
        f"[Observation {index}: '{before}', Action {index}: '{action}']"
        for index, (before, action) in enumerate(recent, start=len(history) - len(recent) + 1)
    )
    template = CODEGYM_TEMPLATE if recent else CODEGYM_TEMPLATE_NO_HIS
    prompt = template.format(
        task_description=task, current_observation=observation,
        available_actions=", ".join(names), action_history=context,
        step_count=len(history), current_step=len(history) + 1, history_length=len(recent),
    )
    return [*systems, {"role": "user", "content": prompt}]
