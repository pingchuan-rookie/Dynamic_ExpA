"""OpenAI-compatible text generation and shared TRAPI wire format."""

import os
from contextlib import contextmanager

from agent_system.utils.thinking import resolve_chat_template_kwargs

from ..types import ModelResponse, TokenUsage
from .trapi import normalize_trapi_request, trapi_client


@contextmanager
def sdk_client(config, timeout_s, max_retries):
    if config.provider == "trapi":
        with trapi_client(config.base_url, timeout_s, max_retries) as client:
            yield client
    else:
        from openai import OpenAI

        key = (os.environ.get(config.api_key_env) or "EMPTY") if config.api_key_env else "EMPTY"
        with OpenAI(base_url=config.base_url, api_key=key, timeout=timeout_s, max_retries=max_retries) as client:
            yield client


def generate(config, messages, options, timeout_s, max_retries):
    params = {"max_tokens": options.max_output_tokens}
    if options.temperature is not None:
        params["temperature"] = options.temperature
    params = normalize_trapi_request(config.model, params)
    template = resolve_chat_template_kwargs(model=config.model)
    extra = {"extra_body": {"chat_template_kwargs": template}} if template else {}
    with sdk_client(config, timeout_s, max_retries) as client:
        response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **params,
            **extra,
        )
        choice = response.choices[0]
        text = choice.message.content
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Model returned no text")
        usage = response.usage
        return ModelResponse(
            text,
            choice.finish_reason,
            TokenUsage(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) if usage else None,
            params,
        )
