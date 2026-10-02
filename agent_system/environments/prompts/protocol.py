"""Task prompt identities, independent of native model thinking settings."""


def task_prompt_protocol(environment: str) -> str:
    if environment in {"dive", "swebench_verified"}:
        # Native tool calls follow reasoning directly, without an outer <action> block.
        return "environment_react_v4"
    if environment in {"t2bench", "gsm8k"}:
        return "environment_react_v3"
    if environment == "codegym":
        return "environment_react_v2"
    return "environment_react_v1"
