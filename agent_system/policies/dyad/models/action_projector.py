# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Poolers: a variable-length sequence of hidden states -> one vector per action.

The encoder LLM backbone reads one MCP prompt per action and produces `[T, H]` last-layer hidden states, with
T differing per action. `action_head` needs exactly one row of width H per action, so something has to
collapse T. That something is the projector, and it is the *only* trainable part of the encoder
(loss.md section 3): the encoder LLM backbone itself stays frozen.

Three of them, in increasing capacity:

  - `mean`      : plain average. Zero parameters. The degenerate baseline, and what makes
                  "did adding capacity actually help" answerable.
  - `mlp`       : score each position with a small MLP, softmax over T, weighted sum. Learns *which
                  positions matter*, one scalar per position.
  - `attention` : learnable query attending over the positions. Learns which positions matter *and*
                  what to read from them, because the values are projected rather than used raw.

All three take `[B, T, H]` plus a `[B, T]` mask and return `[B, H]`. Batching over actions rather
than looping is not just speed: with a loop, padding bugs stay invisible because every sequence is
its own batch of one.

Padding is masked, never averaged over. Left-padding a batch and then taking `.mean(1)` silently
mixes pad states into every row -- and since pad states are *valid-looking* vectors, nothing
downstream would report it. The head would just be subtly wrong for short prompts.
"""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Pool masked encoder token representations into action vectors.
# Extension point: DirectActionHead.forward -> ActionProjector.forward

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import nn


class ActionProjector(nn.Module, ABC):
    """`[B, T, H]` + `[B, T]` mask -> `[B, H]`."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = int(hidden_size)

    @abstractmethod
    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ...

    def _check(self, hidden: torch.Tensor, mask: torch.Tensor) -> None:
        if hidden.ndim != 3:
            raise ValueError(f"expected hidden [B, T, H], got {tuple(hidden.shape)}")
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {hidden.shape[-1]}")
        if mask.shape != hidden.shape[:2]:
            raise ValueError(f"expected mask {tuple(hidden.shape[:2])}, got {tuple(mask.shape)}")
        if not torch.all(mask.sum(dim=1) > 0):
            # An all-pad row would divide by zero in mean pooling and produce a uniform attention
            # distribution over nothing in the other two. Refuse rather than emit a NaN row: a NaN
            # in action_head propagates into every logit and kills the run several steps later,
            # somewhere that gives no hint about where it came from.
            raise ValueError("every row needs at least one unmasked position")


class MeanProjector(ActionProjector):
    """Plain masked average. No parameters.

    Pools encoder representations without adding trainable pooling parameters.
    The setting validator requires another trainable component whenever the schedule needs one.
    """

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._check(hidden, mask)
        w = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * w).sum(dim=1) / w.sum(dim=1)


class MlpProjector(ActionProjector):
    """Score each position with an MLP, softmax over T, weighted sum.

    One scalar per position, so this can learn "the instruction at the end matters more than the JSON
    braces" but cannot change *what* is read from a position -- the output is still a convex
    combination of the input states. That is the deliberate difference from `attention`.

    The final layer is zero-initialised, so at step 0 every logit is 0, the softmax is uniform, and
    this reduces **exactly** to `MeanProjector`: the initial pooler reads a masked mean.
    """

    def __init__(self, hidden_size: int, bottleneck: int = 128):
        super().__init__(hidden_size)
        self.score = nn.Sequential(
            nn.Linear(self.hidden_size, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._check(hidden, mask)
        logits = self.score(hidden).squeeze(-1)                      # [B, T]
        logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)         # [B, T, 1]
        return (hidden * weights).sum(dim=1)


class AttentionProjector(ActionProjector):
    """One learnable query attending over the positions (BLIP-2's Q-Former with a single query).

    Unlike `mlp`, the output is a combination of *projected* values, so this can read a transformed
    view of a position rather than the position itself.

    Also zero-initialised into an exact `MeanProjector`, but it takes three separate choices to get
    there, and all three are needed:
      - `q` is zero, so every attention logit is 0 and the weights are uniform;
      - `v_proj` is the identity, so the values are the raw hidden states;
      - `out_proj` is the identity, so nothing is mixed on the way out.
    Uniform weights over identity-projected values is the mean. Miss any one and step 0 stops being
    equivalent, which the golden test in test_action_encoder_llm.py is there to catch.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__(hidden_size)
        if self.hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {self.hidden_size} is not divisible by num_heads {num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads

        self.query = nn.Parameter(torch.zeros(self.hidden_size))
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # k_proj stays randomly initialised: with q=0 its output never reaches the logits, and
        # zeroing it too would leave the keys with no gradient signal to break symmetry from.
        nn.init.eye_(self.v_proj.weight)
        nn.init.eye_(self.out_proj.weight)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._check(hidden, mask)
        b, t, _ = hidden.shape
        q = self.query.view(1, self.num_heads, 1, self.head_dim)                       # [1, nh, 1, hd]
        k = self.k_proj(hidden).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        logits = (q * k).sum(dim=-1) / math.sqrt(self.head_dim)                        # [B, nh, T]
        logits = logits.masked_fill(mask.unsqueeze(1) == 0, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)                          # [B, nh, T, 1]
        pooled = (weights * v).sum(dim=2).reshape(b, self.hidden_size)                 # [B, H]
        return self.out_proj(pooled)


_POOLERS: dict[str, type[ActionProjector]] = {
    "mean": MeanProjector,
    "mlp": MlpProjector,
    "attention": AttentionProjector,
}


def build_projector(name: str, hidden_size: int, **kwargs) -> ActionProjector:
    cls = _POOLERS.get(name)
    if cls is None:
        raise ValueError(f"unknown projector {name!r}; known: {sorted(_POOLERS)}")
    return cls(hidden_size, **kwargs)


def projector_names() -> list[str]:
    return sorted(_POOLERS)
