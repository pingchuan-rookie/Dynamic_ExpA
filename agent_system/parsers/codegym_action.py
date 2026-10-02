"""CodeGym's text envelope and official OnlineFcGymEnv JSON boundary."""
import json
import re


def prepare_codegym_action(text: str) -> tuple[str, str | None]:
    """Unwrap one call block without repairing fields or splitting a call list.

    The official client validates JSON, then strips one outer pair of list
    brackets and sends the remaining string to env.step. The environment owns
    name/parameter errors and completion. Missing or partial text envelopes have
    no official equivalent; send their unchanged text to the same JSON check.
    """
    blocks = re.findall(r"<\|FunctionCallBegin\|>(.*?)<\|FunctionCallEnd\|>", text, re.DOTALL)
    single_block = (len(blocks) == 1 and text.count('<|FunctionCallBegin|>') == 1
                    and text.count('<|FunctionCallEnd|>') == 1)
    action = (blocks[0] if single_block else text).strip()
    try:
        json.loads(action)
    except (ValueError, TypeError) as exc:
        return action, f"The action cannot be parsed in json format {action}, {exc}"
    if action.startswith("[") and action.endswith("]"):
        action = action[1:-1]
    return action, None
