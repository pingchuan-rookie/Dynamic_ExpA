"""Agent-turn environment hooks shared by the existing rollout loops."""
from __future__ import annotations

import copy
import inspect


def initialize_task_evidence(data):
    """Bind scored ALFWorld/WebShop attempts to their real dataset reset IDs."""
    if not getattr(data, "is_validate", False):
        return
    kwargs = getattr(data, "tools_kwargs", None) or getattr(data, "extra_info", {}).get("tools_kwargs", {})
    selections = [(name, kwargs[name]) for name in ("alfworld_action", "webshop_action") if name in kwargs]
    if not selections:
        return
    if len(selections) != 1:
        raise ValueError("An evaluated trajectory must select one environment task")
    name, settings = selections[0]
    create = settings.get("create_kwargs", {})
    task_id = create.get("task_id")
    if task_id is None or isinstance(task_id, bool) or not str(task_id):
        raise ValueError("Environment evaluation requires the dataset reset task_id")
    # A policy that never calls a valid tool is still an attempted task, not a
    # discarded sample. Infrastructure failures are accumulated separately below.
    data.extra_fields["environment_evidence"] = {
        "task_id": str(task_id), "benchmark": "alfworld" if name == "alfworld_action" else "webshop",
        "metric_valid": True, "service_error": False,
        "won": False, "task_score": 0.0,
    }


def record_task_response(data, metrics):
    evidence = data.extra_fields.get("environment_evidence")
    if evidence is None:
        return
    if not isinstance(metrics, dict) or any(metrics.get(key) for key in (
        "env_actor_error", "http_failure", "tool_failure", "timed_out", "service_error",
    )):
        evidence.update(metric_valid=False, service_error=True)
        return
    if "won" in metrics:
        evidence["won"] = evidence["won"] or bool(metrics["won"])
    if evidence["benchmark"] == "webshop" and "task_score" in metrics:
        evidence["task_score"] = metrics["task_score"]


def invalidate_task_evidence(data):
    evidence = data.extra_fields.get("environment_evidence")
    if evidence is not None:
        evidence.update(metric_valid=False, service_error=True)


async def _value(value):
    return await value if inspect.isawaitable(value) else value


def record_action_audit(loop, data, output, action_payload=None, policy_trace=None):
    """Retain generation evidence independently of optional diagnostic logging."""
    if getattr(data, "runtime_env_tool", None) is None:
        return
    ids = [int(token) for token in output.token_ids]
    entry = {
        "assistant_turn": data.assistant_turns,
        "requested_max_tokens": getattr(data, "runtime_generation_max_tokens", None),
        "schema_hash": data.runtime_env_context.get("schema_hash"),
        "server_request_id": getattr(output, "server_request_id", None),
        "stop_reason": getattr(output, "stop_reason", None),
        "finish_reason": getattr(output, "finish_reason", None),
        "engine_stop_reason": getattr(output, "engine_stop_reason", None),
        "token_ids": ids,
        "decoded_with_special_tokens": loop.tokenizer.decode(ids, skip_special_tokens=False),
        "raw_text": loop.tokenizer.decode(ids, skip_special_tokens=True),
        "tokenizer_eos_token_id": getattr(loop.tokenizer, "eos_token_id", None),
        "action_content": copy.deepcopy(action_payload),
        "raw_token_ids": copy.deepcopy(action_payload.get("raw_token_ids")) if action_payload else None,
        "policy_trace": copy.deepcopy(policy_trace),
        "selection": None,
        "submitted": False,
    }
    data.extra_fields.setdefault("action_audit", []).append(entry)
    data.runtime_action_audit = entry


def session_sampling_params(loop, data, sampling_params):
    """Apply the session's per-decision cap without changing trajectory capacity."""
    if getattr(data, "runtime_env_tool", None) is None:
        return sampling_params
    remaining = loop.response_length - len(data.response_mask)
    if remaining <= 0:
        data.termination_reason = "response_length"
        return None
    limits = [remaining]
    if getattr(data, "runtime_max_tokens", None) is not None:
        limits.append(data.runtime_max_tokens)
    params = dict(sampling_params)
    for key in ("max_tokens", "max_new_tokens"):
        if key in params:
            limits.append(params.pop(key))
    params["max_tokens"] = min(limits)
    data.runtime_generation_max_tokens = params["max_tokens"]
    return params


async def bootstrap_session(loop, data):
    tools = getattr(data, "_active_tools", loop.tools)
    candidates = [tool for tool in tools.values() if getattr(tool, "runtime_session", False)]
    if not candidates:
        return False
    if len(candidates) != 1:
        raise ValueError("A trajectory must own exactly one runtime environment session")
    tool = candidates[0]
    data.runtime_env_tool = tool
    loop.runtime_env_session = True
    kwargs = data.tools_kwargs or data.extra_info.get("tools_kwargs", {})
    create_kwargs = dict(kwargs.get(tool.name, {}).get("create_kwargs", {}))
    settings = create_kwargs.get("create_payload", {}).get("session_config", {})
    data.runtime_max_tokens = settings.get("max_tokens")
    if data.runtime_max_tokens is not None and (
        type(data.runtime_max_tokens) is not int or data.runtime_max_tokens < 1
    ):
        raise ValueError("Runtime per-decision max_tokens must be a positive integer")
    instance_id, _ = await tool.create(instance_id=data.request_id, **create_kwargs)
    data.extra_fields.setdefault("tool_instances", {})[tool.name] = instance_id
    data.runtime_env_id = instance_id
    context = await _value(tool.get_session_context(instance_id))
    data.runtime_env_context = context
    data.messages = copy.deepcopy(context["initial_messages"])
    data.extra_fields["raw_prompt"] = copy.deepcopy(data.messages)
    data.extra_fields["terminal_official"] = True
    if context.get("protocol") == "dive":
        data.extra_fields["runtime_protocol"] = "dive"
        from agent_system.parsers.dive import native_protocol
        data.runtime_native_protocol = native_protocol(loop.tokenizer)
        if getattr(data, "gigpo_enabled", False):
            data.gigpo_anchor = context["observation"]
            data.gigpo_initial = True
    data.runtime_env_done = bool(context.get("done"))
    if context.get("episode_result"):
        data.extra_fields["episode_result"] = context["episode_result"]
    return True


async def finalize_session(data):
    tool = getattr(data, "runtime_env_tool", None)
    if tool is None:
        return
    result = await _value(tool.finalize(data.runtime_env_id, data.termination_reason or "rollout_terminated"))
    if result and "episode_result" in result:
        result = result["episode_result"]
    if result:
        data.extra_fields["episode_result"] = result
    if "episode_result" not in data.extra_fields:
        raise ValueError("Runtime environment did not return a terminal result")
    # Session finalization may replace the terminal result; attach local policy evidence afterwards.
    data.extra_fields["episode_result"] = copy.deepcopy(data.extra_fields["episode_result"])
    data.extra_fields["episode_result"]["action_audit"] = copy.deepcopy(data.extra_fields.get("action_audit", []))
    if data.extra_fields.get("runtime_protocol") == "dive":
        terminal = data.extra_fields["episode_result"]
        if not terminal.get("metric_valid"):
            raise RuntimeError("DIVE received an invalid environment or judge result")
        if getattr(data, "gigpo_enabled", False):
            if not data.gigpo_steps:
                raise ValueError("DIVE GiGPO trajectory has no policy decision")
            if not getattr(data, "runtime_terminal_reward_recorded", False):
                score = terminal.get("official_reward") if terminal.get("official_scored") else terminal.get("attempt_reward")
                data.gigpo_steps[-1]["reward"] += float(score)
                data.runtime_terminal_reward_recorded = True


async def advance_session(loop, data, *, dyad=False):
    if data.runtime_env_context.get("protocol") == "dive":
        return await advance_dive_session(loop, data, dyad=dyad)
    raw_text = loop.tokenizer.decode(data.response_ids, skip_special_tokens=True)
    selection = getattr(data, "runtime_action_selection", None)
    action = {"raw_text": raw_text}
    if selection is not None:
        action["selected_action"] = selection["name"]
    if getattr(data, "runtime_missing_expanded_head", False):
        if selection is not None:
            raise ValueError("Missing-head feedback cannot carry a selected action")
        action["missing_expanded_head"] = True
    audit = getattr(data, "runtime_action_audit", None)
    if audit is not None:
        if audit["raw_text"] != raw_text or audit["token_ids"] != list(data.response_ids):
            raise ValueError("Runtime action differs from audited generation")
        audit.update(selection=copy.deepcopy(selection), submitted=True, submitted_raw_text=raw_text,
                     missing_expanded_head=bool(action.get("missing_expanded_head")))
    _, _, metrics = await data.runtime_env_tool.execute(data.runtime_env_id, action)
    context = await _value(data.runtime_env_tool.get_session_context(data.runtime_env_id))
    if context.get("schema_hash") != data.runtime_env_context.get("schema_hash"):
        raise ValueError("Environment changed action schema within one trajectory")
    data.runtime_env_context = context
    delta = copy.deepcopy(context.get("messages_delta", []))
    previous = copy.deepcopy(data.messages)
    assistant = {"role": "assistant", "content": raw_text}
    data.runtime_env_done = bool(context.get("done", metrics.get("done", False)))
    history = context.get("agent_messages")
    if history is not None:
        if history[:len(previous)] != previous:
            raise ValueError("Environment agent history differs from rollout-visible messages")
        extension = copy.deepcopy(history[len(previous):])
        # Official sessions can include the submitted assistant in their delta.
        if extension:
            if extension[0] != assistant:
                raise ValueError("Environment changed the submitted assistant message")
            delta = extension[1:]
        elif data.runtime_env_done:
            # Parse failures can terminate without accepting an assistant message.
            delta = []
        else:
            raise ValueError("Environment omitted the submitted assistant message")
        data.messages = copy.deepcopy(history)
    else:
        data.messages = previous + [assistant] + delta
    previous.append(assistant)
    if context.get("episode_result"):
        data.extra_fields["episode_result"] = context["episode_result"]
    format_error = bool(context.get("format_error"))
    if format_error:
        data.extra_fields["format_error_count"] = data.extra_fields.get("format_error_count", 0) + 1
    if dyad:
        if not format_error:
            data.tool_turns += 1
        data.last_env_metrics = metrics
    if not delta:
        if not data.runtime_env_done:
            raise ValueError("Environment returned no messages at a nonterminal decision point")
        data.termination_reason = "env_done"
        return True
    if format_error and not data.runtime_env_done:
        for limit_name, count_name in (("max_assistant_turns", "assistant_turns"), ("max_user_turns", "user_turns")):
            limit = getattr(loop, limit_name, None)
            if limit and getattr(data, count_name) >= limit:
                data.termination_reason = limit_name
                return True
    from agent_system.rollout.action_recovery import _closing_suffix
    closing = _closing_suffix(loop, data.prompt_ids) if format_error else []
    collect_logprobs = getattr(data, "collect_response_logprobs", False) or bool(data.response_logprobs)
    if getattr(loop, "enable_continuous_token", False):
        merged, mask, logprobs = await loop.ct_merge_non_assistant_msg(
            previous, data.messages, data.prompt_ids + closing, data.response_mask + [0] * len(closing),
            data.response_logprobs + [0.0] * len(closing) if collect_logprobs else None, tools=None)
        if dyad:
            raise ValueError("Runtime Dyad requires append-only token assembly")
        ids = merged.token_ids
        if len(mask) > loop.response_length or (format_error and len(mask) == loop.response_length):
            data.termination_reason = "response_length"
            return True
        data.prompt_ids, data.response_mask = ids, mask
        if logprobs is not None:
            data.response_logprobs = logprobs
    else:
        ids = await loop.apply_chat_template(delta, tools=None, remove_system_prompt=True)
        ids = (closing if format_error else list(loop.turn_separator)) + ids
        total_length = len(data.response_mask) + len(ids)
        if total_length > loop.response_length or (format_error and total_length == loop.response_length):
            data.termination_reason = "response_length"
            return True
        data.prompt_ids += ids
        data.response_mask += [0] * len(ids)
        if collect_logprobs:
            data.response_logprobs += [0.0] * len(ids)
        if dyad:
            data.response_dyad += ids
            data.seq_mask += [False] * len(ids)
            data.tool_mask += [False] * len(ids)
            data.dyad_allowed_action_ids += [[] for _ in ids]
    data.user_turns += 1
    if format_error and not data.runtime_env_done:
        data.extra_fields["format_retry_count"] = data.extra_fields.get("format_retry_count", 0) + 1
    if data.runtime_env_done:
        data.termination_reason = "env_done"
    return data.runtime_env_done


async def advance_dive_session(loop, data, *, dyad=False):
    from agent_system.parsers.dive import DiveFormatError, decode_response, parse_response

    context = data.runtime_env_context
    raw_text = decode_response(loop.tokenizer, data.response_ids)
    try:
        action = parse_response(raw_text, context["action_tools"], data.runtime_native_protocol,
                                call_prefix=f"{data.request_id}_{data.assistant_turns}")
        if dyad:
            selections = getattr(data, "runtime_action_selections", [])
            names = [selection["name"] for selection in selections]
            if names != [call["name"] for call in action["tool_calls"]]:
                raise DiveFormatError("Tool calls do not match actual expanded-head selections")
            action["selected_actions"] = names
    except DiveFormatError as exc:
        action = {"raw_text": raw_text, "content": "", "tool_calls": [], "format_error": str(exc)}
    audit = getattr(data, "runtime_action_audit", None)
    if audit is not None:
        audit.update(submitted=True, submitted_raw_text=raw_text,
                     selections=copy.deepcopy(getattr(data, "runtime_action_selections", None)))
    _, _, metrics = await data.runtime_env_tool.execute(data.runtime_env_id, action)
    updated = await _value(data.runtime_env_tool.get_session_context(data.runtime_env_id))
    if updated["schema_hash"] != context["schema_hash"]:
        raise ValueError("DIVE changed tool schemas within one trajectory")
    data.runtime_env_context = updated
    data.runtime_env_done = bool(updated.get("done"))
    if updated.get("episode_result"):
        data.extra_fields["episode_result"] = updated["episode_result"]
    delta = copy.deepcopy(updated.get("messages_delta", []))
    # Keep sampled assistant bytes as the policy history; parsing never rewrites them.
    data.messages.append({"role": "assistant", "content": raw_text})
    data.messages.extend(delta)
    data.last_env_metrics = metrics
    format_error = bool(action.get("format_error") or updated.get("format_error"))
    if format_error:
        data.extra_fields["format_error_count"] = data.extra_fields.get("format_error_count", 0) + 1
    elif action["tool_calls"]:
        data.tool_turns = getattr(data, "tool_turns", 0) + len(action["tool_calls"])
    if getattr(data, "gigpo_enabled", False):
        if data.gigpo_anchor != context["observation"]:
            raise ValueError("DIVE GiGPO state differs from the sampled decision state")
        data.gigpo_steps[-1]["executed"] = not format_error
        data.gigpo_anchor = updated["observation"]
        data.gigpo_initial = False
    if data.runtime_env_done:
        data.termination_reason = "env_done"
        return True
    if not delta:
        raise ValueError("Nonterminal DIVE decision returned no tool feedback")
    if getattr(loop, "enable_continuous_token", False):
        raise ValueError("DIVE requires append-only native tool transcript assembly")
    ids = await loop.apply_chat_template(delta, tools=None, remove_system_prompt=True)
    # Generation stops on EOS; append it only when the engine did not return it.
    separator = list(loop.turn_separator)
    eos = getattr(loop.tokenizer, "eos_token_id", None)
    if data.response_ids and data.response_ids[-1] == eos:
        if separator and separator[0] == eos:
            separator = separator[1:]
    elif eos is not None and (not separator or separator[0] != eos):
        separator.insert(0, eos)
    ids = separator + ids
    if len(data.response_mask) + len(ids) > loop.response_length:
        data.termination_reason = "response_length"
        return True
    data.prompt_ids += ids
    data.response_mask += [0] * len(ids)
    if getattr(data, "collect_response_logprobs", False) or data.response_logprobs:
        data.response_logprobs += [0.0] * len(ids)
    if dyad:
        data.response_dyad += ids
        data.seq_mask += [False] * len(ids)
        data.tool_mask += [False] * len(ids)
        data.dyad_allowed_action_ids += [[] for _ in ids]
    data.user_turns += 1
    return False
