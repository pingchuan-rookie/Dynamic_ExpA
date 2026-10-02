# Copyright 2025 ExpA_sys
"""Read model dimensions from a HuggingFace config, flat or nested.

Qwen2.5-3B and Qwen3-4B are `*ForCausalLM`: `hidden_size` sits at the top level.
Qwen3.5-4B is `Qwen3_5ForConditionalGeneration`, whose config is a composite and keeps every
text dimension under `text_config`; reading `config.hidden_size` on it raises AttributeError
rather than returning None, so a caller that guesses wrong dies at worker construction.

`get_text_config()` is the transformers-side accessor for exactly this and is a no-op
(returns `self`) on flat configs, which is why this module does not sniff for `text_config`.
"""

from __future__ import annotations

from typing import Any


def text_hidden_size(config: Any) -> int:
    """Hidden width of the *text* stack, which is the width Dyad's head and projector are sized to.

    A vision tower has its own, different, hidden size; sizing the head to it would produce a
    head whose shape is self-consistent everywhere and wrong everywhere.
    """
    return int(_text_config(config).hidden_size)


def text_vocab_size(config: Any) -> int:
    """Size of the base embedding table, i.e. the id at which Dyad's action ids start.

    Both sides of AGENTS.md section 1 read this: the agent loop offsets action labels by it when it
    compiles the schema, and the model runner offsets them back when it masks logits. They must
    agree, which is why both go through this one function rather than each reaching into a config.
    """
    return int(_text_config(config).vocab_size)


def _text_config(config: Any) -> Any:
    # `get_text_config()` is the transformers-side accessor for composite configs and returns
    # `self` on flat ones. Objects without it (test stubs) are used as-is; one that then also
    # lacks the attribute raises, because guessing a width here is how a run trains a head of the
    # wrong shape without anything reporting it.
    return config.get_text_config() if hasattr(config, "get_text_config") else config
