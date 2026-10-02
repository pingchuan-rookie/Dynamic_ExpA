"""Recover malformed action turns without advancing an environment."""
from __future__ import annotations

from agent_system.utils.diagnostics import record_span


def _closing_suffix(loop, prompt_ids):
    """Complete the chat boundary, never the model's action payload."""
    suffix = getattr(loop, "_format_recovery_closing", None)
    if suffix is None:
        tokenizer = loop.tokenizer
        if not hasattr(tokenizer, "apply_chat_template"):
            return list(getattr(loop, "turn_separator", []))
        kwargs = dict(getattr(loop, "apply_chat_template_kwargs", {}))
        kwargs.update(tokenize=True, add_generation_prompt=False, return_dict=False)
        probes = [list(tokenizer.apply_chat_template(
            [{"role": "user", "content": "Continue."}, {"role": "assistant", "content": text}], **kwargs,
        )) for text in ("x", "y")]
        length = 0
        while length < min(map(len, probes)) and probes[0][-length - 1] == probes[1][-length - 1]:
            length += 1
        suffix = probes[0][-length:] if length else []
        loop._format_recovery_closing = suffix
    for overlap in range(min(len(prompt_ids), len(suffix)), 0, -1):
        if list(prompt_ids[-overlap:]) == suffix[:overlap]:
            return suffix[overlap:]
    return list(suffix)


async def recover_missing_action(loop, data, *, dyad=False):
    """Return None for legal text completion, otherwise whether generation can resume."""
    feedback = getattr(loop.tool_parser, "missing_action_feedback", None)
    if not feedback:
        return None
    data.extra_fields["format_error_count"] = data.extra_fields.get("format_error_count", 0) + 1
    for limit_name, count_name in (("max_assistant_turns", "assistant_turns"), ("max_user_turns", "user_turns")):
        limit = getattr(loop, limit_name, None)
        if limit and getattr(data, count_name) >= limit:
            data.termination_reason = limit_name
            return False
    if len(data.response_mask) >= loop.response_length:
        data.termination_reason = "response_length"
        return False

    continuous = getattr(loop, "enable_continuous_token", False)
    collect_logprobs = getattr(data, "collect_response_logprobs", False) or bool(data.response_logprobs)
    previous = list(data.messages)
    if not continuous:
        previous.append({"role": "assistant", "content": loop.tokenizer.decode(data.response_ids, skip_special_tokens=True)})
    delta = [{"role": "user", "content": feedback}]
    messages = previous + delta
    closing = _closing_suffix(loop, data.prompt_ids)
    if continuous:
        if dyad:
            raise ValueError("Dyad requires append-only token assembly")
        merged, mask, logprobs = await loop.ct_merge_non_assistant_msg(
            previous, messages, data.prompt_ids + closing, data.response_mask + [0] * len(closing),
            data.response_logprobs + [0.0] * len(closing) if collect_logprobs else None, tools=None,
        )
        if len(mask) <= len(data.response_mask):
            raise ValueError("Format feedback must consume a positive token budget")
        if len(mask) >= loop.response_length:
            data.termination_reason = "response_length"
            return False
        data.prompt_ids, data.response_mask = merged.token_ids, mask
        if logprobs is not None:
            data.response_logprobs = logprobs
    else:
        ids = await loop.apply_chat_template(delta, tools=None, remove_system_prompt=True)
        ids = closing + list(ids)
        if not ids:
            raise ValueError("Format feedback must consume a positive token budget")
        if len(data.response_mask) + len(ids) >= loop.response_length:
            data.termination_reason = "response_length"
            return False
        record_span(data.span_records, "ENV", len(data.response_mask), len(ids))
        data.prompt_ids += ids
        data.response_mask += [0] * len(ids)
        if collect_logprobs:
            data.response_logprobs += [0.0] * len(ids)
        if dyad:
            data.response_dyad += ids
            data.seq_mask += [False] * len(ids)
            data.tool_mask += [False] * len(ids)
            data.dyad_allowed_action_ids += [[] for _ in ids]
    data.messages = messages
    data.user_turns += 1
    data.extra_fields["format_retry_count"] = data.extra_fields.get("format_retry_count", 0) + 1
    return True
