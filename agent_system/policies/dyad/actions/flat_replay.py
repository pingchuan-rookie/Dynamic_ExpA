"""Replay legacy flat decisions with the sampling router."""
from __future__ import annotations

from agent_system.policies.dyad.actions.flat_router import FlatActionRouter


def build_flat_policy_trace(action_config, raw_token_ids, emitted_token_ids):
    """Training side: **replay** the rollout with FlatActionRouter to rebuild the mask at every emitted position.

    Matches the output format of policy_trace.build_policy_trace:
      seq_mask[i]           : whether the i-th emitted position is a "real policy decision"
      tool_mask[i]          : whether that decision is an extended-action decision (action id)
      allowed_action_ids[i] : extended action ids allowed at that decision point (already -vocab_size), empty for vocab decisions
      response_dyad[i]      : decision positions carry the raw decision id back (extended decisions write the extended id), the rest match emitted

    raw_token_ids: the "real sampled decision sequence" recorded by rollout (SAMPLE steps only, in order).
    emitted_token_ids: the plain tokens actually written into the context (forced tokens included).

    Because FlatActionRouter is fully deterministic, the mask produced by the replay is per-decision
    identical to the one used during rollout (the core invariant).
    """
    V = action_config["num_embeddings_size"]
    n = len(emitted_token_ids)
    seq_mask = [False] * n
    tool_mask = [False] * n
    allowed = [[] for _ in range(n)]
    response_dyad = list(emitted_token_ids)

    router = FlatActionRouter(action_config)
    raw_idx = 0
    pos = 0
    guard = 0
    while pos < n:
        guard += 1
        if guard > 10 * (n + 8):
            raise RuntimeError("codegym policy_trace replay loop guard")
        d = router.decision()
        if d["kind"] == "force":
            # forced token: not a decision position
            router.advance(d["forced_token"])
            pos += 1
            continue
        # SAMPLE: consume one real decision
        if raw_idx >= len(raw_token_ids):
            break
        raw = int(raw_token_ids[raw_idx]); raw_idx += 1
        is_tool = raw >= V
        seq_mask[pos] = True
        tool_mask[pos] = is_tool
        if is_tool:
            response_dyad[pos] = raw
        # the allowed set at this decision point
        if d["phase"] == "ARGUMENT_VALUE":
            # free value over the full vocabulary (terminating '\n' included): no expanded action admitted
            allowed[pos] = []
        else:
            allowed[pos] = [aid - V for aid in d["allowed_action_ids"]]
        router.advance(raw)
        pos += 1

    return {
        "emitted_token_ids": list(emitted_token_ids),
        "response_dyad": response_dyad,
        "seq_mask": seq_mask,
        "tool_mask": tool_mask,
        "allowed_action_ids": allowed,
        "action_size": action_config["total_size"],
    }

