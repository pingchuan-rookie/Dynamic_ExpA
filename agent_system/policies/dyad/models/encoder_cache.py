# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Frozen encoder representations and synchronized cache views."""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Provide frozen or synchronized representations to the shared action-head builder.
# Extension point: LlmActionEncoder / DirectActionHead -> build_action_head
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CachedHidden:
    """One schema's frozen encoder output. `hidden` is [A, T, H], `mask` is [A, T]."""

    hidden: torch.Tensor
    mask: torch.Tensor
    fingerprint: str



def prompt_fingerprint(prompts: list[str], tokenizer_name: str, encoder_path: str) -> str:
    """Everything that changes the encoder's answer, in one string.

    Includes the prompts themselves rather than a schema name: two schemas can share a name across
    variants, and the prompts are what actually feed the model.
    """
    digest = hashlib.sha256()
    digest.update(encoder_path.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(tokenizer_name.encode("utf-8"))
    for prompt in prompts:
        digest.update(b"\x00")
        digest.update(prompt.encode("utf-8"))
    return digest.hexdigest()[:32]



class CachedHiddenSource:
    """An `encode_hidden`-shaped view over an projector's stored encoder output.

    The engine has the frozen encoder's output (it arrives as buffers on the projector during the
    weight sync) but no backbone to produce it. `build_action_head` asks its encoder for
    `encode_hidden(prompts)`, so this gives it something with that shape and no model behind it.

    A shim rather than a branch inside `build_action_head`: a second path through the head builder is
    how "head init has three dispatch sites and the last one always gets missed" happens again.
    """

    def __init__(self, residual_head):
        self._residual_head = residual_head

    @property
    def hidden_size(self) -> int:
        cached = self._residual_head.cached_encoder_output()
        return int(cached.hidden.shape[-1])

    def encode_hidden(self, prompts: list[str], *, use_cache: bool = True) -> CachedHidden:  # noqa: ARG002
        cached = self._residual_head.cached_encoder_output()
        if cached.hidden.shape[0] != len(prompts):
            raise RuntimeError(
                f"the synced encoder cache covers {cached.hidden.shape[0]} actions but this schema "
                f"has {len(prompts)}. The engine and the trainer compiled different action configs; "
                "building a head from this would silently mis-align every row."
            )
        return cached

    def clear_cache(self) -> None:
        """Nothing to clear: the cache is owned by the projector and replaced by each weight sync."""

