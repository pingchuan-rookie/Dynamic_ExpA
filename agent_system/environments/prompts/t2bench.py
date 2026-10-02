"""GiGPO-style decision prompts for evaluation on official tau2 domains."""
from __future__ import annotations

import json


T2BENCH_INSTRUCTION = '''
# Instruction
Follow the policy above and the declared tool schemas.
Ask the customer for missing information or required confirmation; do not invent facts or tool results.
'''

_T2BENCH_RESPONSE_INSTRUCTIONS = '''Reason about the next step within <think> </think> tags.
Then output exactly one action as valid JSON enclosed within <action> </action> tags:
<action>
{{"name": "<action name>", "arguments": {{"<argument name>": "<value>"}}}}
</action>
To message the customer, use name="respond" and arguments={{"content": "<message>"}}; this does not end the conversation.
Stop after </action> and wait for feedback; output no text outside the <think> and <action> blocks or Markdown fences.
'''

T2BENCH_TEMPLATE_NO_HIS = '''You are an expert agent helping a customer in an interactive service environment.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s).
You are now at step {current_step} and your current observation is: {current_observation}
Your available actions are: [{available_actions}].

''' + _T2BENCH_RESPONSE_INSTRUCTIONS

T2BENCH_TEMPLATE = '''You are an expert agent helping a customer in an interactive service environment.
Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your available actions are: [{available_actions}].

''' + _T2BENCH_RESPONSE_INSTRUCTIONS


def build_t2bench_prompt(domain_policy: str, tools: list[dict]) -> str:
    """Keep the official policy and schemas intact in the system message."""
    return (domain_policy + '\n#Available tools\n'
            + json.dumps(tools, ensure_ascii=False) + T2BENCH_INSTRUCTION)


def build_t2bench_messages(
    domain_policy: str, tools: list[dict], messages: list[dict], history_length: int = 2,
    *, task_description: str,
) -> list[dict[str, str]]:
    """Render public observations and actions, without replaying unbounded chat.

    The caller supplies the first public customer request as the task description.
    Messages are text-normalized user/tool observations and assistant actions.
    Consecutive observations are grouped into one decision, including multi-tool feedback.
    """
    if type(history_length) is not int or history_length < 0:
        raise ValueError('t2bench history_length must be a nonnegative integer')
    history, pending = [], []
    for message in messages:
        if message['role'] == 'assistant':
            # The official orchestrator can seed an assistant greeting before the
            # first public observation; it is not a sampled agent decision.
            if pending or history:
                history.append(('\n'.join(pending), message.get('content') or ''))
            pending = []
        elif message['role'] == 'user':
            content = message.get('content') or ''
            if content == 'Format error.' and not pending and history:
                # A rejected local action does not replace the live observation.
                before = history[-1][0]
                if before.endswith('\nFormat error.'):
                    before = before[:-len('\nFormat error.')]
                pending.append(before)
            pending.append(content)
        else:
            raise ValueError('t2bench decision history requires normalized user/assistant messages')
    recent = history[-history_length:] if history_length else []
    context = '\n'.join(
        f"[Observation {index}: '{before}', Action {index}: '{action}']"
        for index, (before, action) in enumerate(recent, start=len(history) - len(recent) + 1)
    )
    template = T2BENCH_TEMPLATE if recent else T2BENCH_TEMPLATE_NO_HIS
    prompt = template.format(
        task_description=task_description,
        current_observation='\n'.join(pending) or 'No new customer message or tool result.',
        available_actions=', '.join([tool['function']['name'] for tool in tools] + ['respond']),
        step_count=len(history), history_length=len(recent), action_history=context,
        current_step=len(history) + 1,
    )
    return [
        {'role': 'system', 'content': build_t2bench_prompt(domain_policy, tools)},
        {'role': 'user', 'content': prompt},
    ]
