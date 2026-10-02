from typing import Any


def build_policy_trace(action_payload: dict[str, Any], emitted_token_ids: list[int]) -> dict[str, Any]:
    action_config = action_payload["action_config"]
    # Unified router (router==unified): delegate to ActionRouter replay (same rules as rollout).
    # Must come before the codegym branch (new configs also carry mode==codegym and would be misrouted).
    if action_config.get("router") == "unified" or action_payload.get("unified"):
        from agent_system.policies.dyad.actions.action_router import build_unified_policy_trace

        return build_unified_policy_trace(
            action_config, action_payload["raw_token_ids"], emitted_token_ids)
    # Compatibility for callers holding an old flat CodeGym config. Normal rollout
    # compiles even old CodeGym source schemas into the unified branch above.
    if action_payload.get("codegym") or action_config.get("mode") == "codegym":
        from agent_system.policies.dyad.actions.flat_replay import build_flat_policy_trace
        return build_flat_policy_trace(
            action_config, action_payload["raw_token_ids"], emitted_token_ids)

    # Unknown schemas must fail rather than silently becoming an unmasked vocab trace.
    raise NotImplementedError(
        "build_policy_trace only supports action_config with router:unified or mode:codegym; "
        f"got router={action_config.get('router')!r} mode={action_config.get('mode')!r}"
    )



def build_vocab_policy_trace(
    emitted_token_ids: list[int],
    vocab_size: int,
    action_size: int,
) -> dict[str, Any]:
    """Build a plain-vocab trace when generation contains no expanded Dyad actions."""
    emitted_token_ids = list(emitted_token_ids)
    expanded_ids = [token_id for token_id in emitted_token_ids if token_id >= vocab_size]
    if expanded_ids:
        raise ValueError(
            "Dyad generation returned no action_content but emitted expanded action ids: "
            f"{expanded_ids[:8]}"
        )
    return {
        "emitted_token_ids": emitted_token_ids,
        "response_dyad": emitted_token_ids,
        "seq_mask": [True] * len(emitted_token_ids),
        "tool_mask": [False] * len(emitted_token_ids),
        "allowed_action_ids": [[] for _ in emitted_token_ids],
        "action_size": action_size,
    }


def validate_policy_trace(trace: dict[str, Any], emitted_token_ids: list[int]) -> None:
    if trace["emitted_token_ids"] != list(emitted_token_ids):
        raise ValueError(
            "Dyad policy trace is not aligned with emitted tokens: "
            f"trace={trace['emitted_token_ids']}, emitted={list(emitted_token_ids)}"
        )
    for key in ["response_dyad", "seq_mask", "tool_mask", "allowed_action_ids"]:
        if len(trace[key]) != len(emitted_token_ids):
            raise ValueError(f"Dyad policy trace field {key!r} is not aligned with emitted tokens.")
