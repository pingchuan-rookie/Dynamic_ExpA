"""Historical action-only prompt adapter, not a native model thinking switch.

Retained for explicit compatibility imports; active generation must not select it
from Qwen3.5 identity or enable_thinking=False.
Only exact template fragments are changed, never generated assistant messages,
observations, arbitrary think blocks, or the task question following CodeGym's header.
"""
from __future__ import annotations

import copy


_ALFWORLD_LINES = (
    '<Think>I need to find a mug, so I will check the cabinet first.</Think>\n',
    '<Think>I need to find the mug first, so I should go to the countertop.</Think>\n',
    '<Think>brief reasoning</Think>\n',
)
_WEBSHOP_INSTRUCTION = (
    'On each turn, reason briefly inside <Think>...</Think>, then output exactly one command inside <Action>...</Action>.'
)
_CODEGYM_HEADER = 'Please answer the following question step by step according to the requirements below!'


def no_thinking_messages(messages: list[dict]) -> list[dict]:
    """Return a copy with known ALFWorld/WebShop/CodeGym thought instructions removed."""
    result = copy.deepcopy(messages)
    for message in result:
        text = message.get('content')
        if message.get('role') not in {'system', 'user'} or not isinstance(text, str):
            continue
        if ('Complete the task in a household environment.' in text
                or 'Here is one worked example of the interaction format (a different task):' in text):
            for line in _ALFWORLD_LINES:
                text = text.replace(line, '')
            text = text.replace('Respond in this format:', 'Output only the action, without reasoning or explanation:')
        if 'You are shopping in an online store to satisfy' in text:
            text = text.replace(_WEBSHOP_INSTRUCTION,
                                'On each turn, output only one command inside <Action>...</Action>, without reasoning or explanation.')
            text = text.replace('<Think>I should search for products matching the requested features.</Think>\n', '')
        if text.startswith(_CODEGYM_HEADER):
            header, separator, question = text.partition('\nQuestion:')
            header = header.replace(_CODEGYM_HEADER, 'Use the provided functions to answer the following question according to the requirements below!')
            header = header.replace('* Do not overthink; think briefly, then decide how to call the function.',
                                    '* Output only the function call, without reasoning or explanation.')
            text = header + separator + question
        message['content'] = text
    return result
