"""Anthropic text adapter retained for existing callers."""

import os

from ..types import ModelResponse, TokenUsage


def generate(config, messages, options, timeout_s, max_retries):
    from anthropic import Anthropic

    key = os.environ.get(config.api_key_env) if config.api_key_env else None
    if not key:
        raise ValueError("Configured API key environment variable is missing")
    params = {"max_tokens": options.max_output_tokens}
    if options.temperature is not None:
        params["temperature"] = options.temperature
    system = []
    conversation = []
    for message in messages:
        if message.role == "system":
            if conversation:
                raise ValueError("Anthropic requires system messages before the conversation")
            system.append(message.content)
        else:
            conversation.append({"role": message.role, "content": message.content})
    extra = {"system": "\n\n".join(system)} if system else {}
    with Anthropic(api_key=key, base_url=config.base_url, timeout=timeout_s, max_retries=max_retries) as client:
        response = client.messages.create(model=config.model, messages=conversation, **params, **extra)
        text = "\n".join(block.text for block in response.content if block.type == "text")
        if not text.strip():
            raise ValueError("Model returned no text")
        usage = response.usage
        return ModelResponse(
            text,
            response.stop_reason,
            TokenUsage(usage.input_tokens, usage.output_tokens, usage.input_tokens + usage.output_tokens),
            params,
        )
