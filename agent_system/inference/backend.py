"""Chat protocol adapter over the shared vLLM inference engine."""

from __future__ import annotations

import json
import math
import time
from typing import Any

from agent_system.inference.config import validate_actions
from agent_system.inference.engine import VLLMInference


def validate_request(payload: dict[str, Any], model: str) -> tuple[list[dict], list[dict], dict]:
    """Reject unsupported generation controls instead of silently changing an eval."""
    allowed = {
        "model",
        "messages",
        "tools",
        "args",
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_completion_tokens",
        "seed",
        "stop",
        "n",
        "stream",
        "chat_template_kwargs",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "tool_choice",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"Unsupported request fields: {sorted(unknown)}")
    if payload.get("model") != model:
        raise ValueError(f"Unknown model; use {model!r}")
    if payload.get("stream", False) is not False or payload.get("n", 1) != 1:
        raise ValueError("This endpoint requires stream=false and n=1; submit independent requests for repeats")
    messages = payload.get("messages")
    tools = [] if payload.get("tools") is None else payload["tools"]
    args = {} if payload.get("args") is None else payload["args"]
    if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) for m in messages):
        raise ValueError("messages must be a nonempty list of chat messages")
    if not isinstance(args, dict) or set(args) - {
        "schema",
        "schema_name",
        "protocol",
        "template_tools",
        "strip_thinking_prefill",
        "prompt_limit",
    }:
        raise ValueError(
            "args accepts schema, schema_name, protocol, template_tools, strip_thinking_prefill, prompt_limit"
        )
    for key in ("template_tools", "strip_thinking_prefill"):
        if key in args and type(args[key]) is not bool:
            raise ValueError(f"args.{key} must be boolean")
    validate_actions(tools, {key: args[key] for key in ("schema", "schema_name", "protocol") if key in args})
    if payload.get("tool_choice", "auto") not in ("auto", "none"):
        raise ValueError("tool_choice supports auto or none")
    if payload.get("tool_choice") == "none":
        if args.get("schema") or args.get("schema_name"):
            raise ValueError("tool_choice=none conflicts with an action schema")
        tools = []
    if "max_tokens" in payload and "max_completion_tokens" in payload:
        raise ValueError("Choose max_tokens or max_completion_tokens")
    maximum = payload.get("max_tokens", payload.get("max_completion_tokens"))
    if type(maximum) is not int or maximum < 1:
        raise ValueError("An explicit positive max_tokens or max_completion_tokens is required")
    for key, low, high in (("temperature", 0, 2), ("top_p", 0, 1)):
        value = payload.get(key, 1.0)
        if not isinstance(value, float | int) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be finite and in [{low}, {high}]")
    if payload.get("top_p", 1.0) == 0:
        raise ValueError("top_p must be positive")
    return messages, tools, args


class ChatCompletionBackend:
    """Render messages and serialize responses; VLLMInference owns generation."""

    def __init__(self, inference: VLLMInference) -> None:
        self.inference = inference

    async def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        from vllm import SamplingParams

        from agent_system.utils.thinking import resolve_chat_template_kwargs

        messages, tools, args = validate_request(payload, self.inference.options.model)
        template = resolve_chat_template_kwargs(
            payload.get("chat_template_kwargs", {}),
            model=self.inference.options.source_root,
            tokenizer=self.inference.tokenizer,
        )
        if tools and args.get("template_tools", True):
            template["tools"] = tools
        tokens = self.inference.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False, **template
        )
        if args.get("strip_thinking_prefill"):
            from agent_system.rollout.prompt import strip_thinking_prefill

            tokens = strip_thinking_prefill(tokens, self.inference.tokenizer)
        maximum = payload.get("max_tokens", payload.get("max_completion_tokens"))
        prompt_limit = args.get("prompt_limit", self.inference.options.context_length - maximum)
        if type(prompt_limit) is not int or prompt_limit < 1:
            raise ValueError("prompt_limit must be a positive integer")
        if len(tokens) > prompt_limit or len(tokens) + maximum > self.inference.options.context_length:
            raise ValueError(f"Prompt budget exceeded: {len(tokens)} tokens; no truncation")
        parameters = {
            key: payload[key]
            for key in (
                "temperature",
                "top_p",
                "top_k",
                "seed",
                "stop",
                "presence_penalty",
                "frequency_penalty",
                "repetition_penalty",
            )
            if key in payload
        }
        parameters["max_tokens"] = maximum
        result = await self.inference.generate(
            tokens,
            SamplingParams(**parameters),
            tools=tools,
            args={key: args[key] for key in ("schema", "schema_name", "protocol") if key in args},
        )
        request_id = result.request_id
        output = result.output.outputs[0]
        ids = list(output.token_ids)
        evidence = result.dyad
        from agent_system.parsers.native_tools import decode_response

        raw_text = decode_response(self.inference.tokenizer, ids)
        message = {"role": "assistant", "content": output.text}
        finish_reason = output.finish_reason
        if tools and args.get("protocol", "native_tools") == "native_tools":
            from agent_system.parsers.native_tools import NativeToolFormatError, native_protocol, parse_response

            try:
                action = parse_response(
                    raw_text,
                    tools,
                    native_protocol(self.inference.tokenizer),
                    call_prefix=request_id,
                    require_reasoning=bool(args.get("strip_thinking_prefill")),
                )
            except NativeToolFormatError:
                pass  # The environment/scorer must see the original malformed response.
            else:
                if action["tool_calls"]:
                    message["content"] = action["content"] or None
                    message["tool_calls"] = [
                        {
                            "id": call.get("id", f"{request_id}_{i}"),
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                            },
                        }
                        for i, call in enumerate(action["tool_calls"])
                    ]
                    finish_reason = "tool_calls"
        return {
            "id": "chatcmpl-" + request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.inference.options.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": len(ids),
                "total_tokens": len(tokens) + len(ids),
            },
            "dyad": evidence,
            "token_ids": ids,
            "prompt_token_ids": list(tokens),
            "raw_text": raw_text,
            "stop_reason": output.stop_reason,
        }
