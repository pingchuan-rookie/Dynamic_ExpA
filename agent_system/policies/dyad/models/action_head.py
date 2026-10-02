# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Project encoder representations into action-head weights."""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Build differentiable action weights and hold their synchronized encoder buffers.
# Extension point: Dyad model setup / forward hook / rollout head materialization
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from agent_system.policies.dyad.models.action_projector import ActionProjector, build_projector
from agent_system.policies.dyad.models.encoder_cache import CachedHidden

# Limit FP32 projection workspace by batching independent action rows. A single
# long action remains intact; this does not truncate encoder inputs or candidates.
_PROJECTOR_CHUNK_ELEMENTS = 4 * 1024 * 1024


class DirectActionHead(nn.Module):
    """Produce action rows as scale(proj(projector(hidden))).

    The reference supplies output device/dtype and, for uniform scaling, its mean row norm.
    Its row count is independent of the number of actions: callers pass the vocabulary head.
    A width-changing projection uses the standard nonzero initialization.
    Attention and MLP poolers initially compute the masked mean of encoder representations.
    """

    def __init__(
        self,
        projector: str,
        encoder_hidden: int,
        actor_hidden: int,
        *,
        scale: str = "uniform",
        projector_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.projector: ActionProjector = build_projector(projector, encoder_hidden, **(projector_kwargs or {}))
        self.projector_name = projector
        self.scale = scale
        self.proj = (
            nn.Identity() if encoder_hidden == actor_hidden
            else nn.Linear(encoder_hidden, actor_hidden, bias=False)
        )
        # A non-persistent buffer, so it follows `.to(device)` without entering the state dict.
        # MeanProjector with an identity width projection has no parameters to read a device from.
        self.register_buffer("_device_anchor", torch.zeros(1), persistent=False)
        # The frozen encoder's output for this schema, carried as buffers so it travels with the
        # module: into the checkpoint, through FSDP's broadcast, and -- the reason it is here --
        # through the policy LLM backbone->engine weight sync. The engine loader requires matching state dict keys.
        self.register_buffer("encoder_hidden", None, persistent=True)
        self.register_buffer("encoder_mask", None, persistent=True)

    def set_encoder_cache(self, hidden: torch.Tensor, mask: torch.Tensor) -> None:
        """Store the frozen encoder output. Called once per schema, before the FSDP wrap.

        Before the wrap because FSDP only broadcasts and only serialises buffers that exist at wrap
        time; a buffer filled afterwards is absent from the state dict, so the engine would receive
        the projector's parameters and none of its input.
        """
        if hidden.ndim != 3 or mask.ndim != 2:
            raise ValueError(
                f"expected hidden [A, T, H] and mask [A, T], got {tuple(hidden.shape)} and "
                f"{tuple(mask.shape)}"
            )
        if hidden.shape[:2] != mask.shape:
            raise ValueError(
                f"hidden {tuple(hidden.shape)} and mask {tuple(mask.shape)} disagree on the number "
                "of actions or positions"
            )
        device = self._device_anchor.device
        self.encoder_hidden = hidden.detach().to(device)
        self.encoder_mask = mask.detach().to(device)

    def cached_encoder_output(self) -> "CachedHidden":
        """The stored encoder output, or a clear error if nothing was ever stored."""
        if self.encoder_hidden is None or self.encoder_mask is None:
            raise RuntimeError(
                "this DirectActionHead has no encoder cache. On the trainer that means "
                "set_encoder_cache was not called before the FSDP wrap; on the engine it means the "
                "weight sync carried the projector's parameters but not its buffers. Either way the "
                "head cannot be built, and guessing a value here would produce a plausible-looking "
                "head that matches nothing."
            )
        return CachedHidden(hidden=self.encoder_hidden, mask=self.encoder_mask, fingerprint="cached")

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """`reference` sets the row norm and nothing else; its row count is unconstrained."""
        self.projector._check(hidden, mask)
        if reference.ndim != 2:
            raise ValueError(f"expected reference [rows, hidden], got {tuple(reference.shape)}")
        if reference.shape[1] != self.proj_out_features(hidden.shape[-1]):
            raise ValueError(
                f"reference is {reference.shape[1]} wide but this head emits "
                f"{self.proj_out_features(hidden.shape[-1])} -- the scale reference and the head "
                "rows have to live in the same space for a row norm to mean anything"
            )
        own_device = self._device_anchor.device

        def project(rows, row_mask):
            # Transfer only this chunk of the frozen encoder cache. Checkpoint
            # recomputation also keeps training from retaining all projected K/V.
            rows = rows.to(device=own_device, dtype=torch.float32)
            row_mask = row_mask.to(own_device)
            return self.proj(self.projector(rows, row_mask))

        row_elements = max(1, hidden.shape[1] * hidden.shape[2])
        chunk_rows = max(1, _PROJECTOR_CHUNK_ELEMENTS // row_elements)
        if hidden.shape[0] <= chunk_rows:
            head = project(hidden, mask)
        else:
            chunks = []
            for start in range(0, hidden.shape[0], chunk_rows):
                rows, row_mask = hidden[start:start + chunk_rows], mask[start:start + chunk_rows]
                if torch.is_grad_enabled():
                    chunks.append(checkpoint(project, rows, row_mask, use_reentrant=False))
                else:
                    chunks.append(project(rows, row_mask))
            head = torch.cat(chunks, dim=0)
        head = head.to(device=reference.device)
        return _apply_scale(head, reference, self.scale).to(reference.dtype)

    def proj_out_features(self, encoder_hidden: int) -> int:
        return encoder_hidden if isinstance(self.proj, nn.Identity) else self.proj.out_features

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]



def _apply_scale(head: torch.Tensor, reference: torch.Tensor, scale: str) -> torch.Tensor:
    """Set every row to a common norm. `reference` is read only by `uniform`.

    **Why a common norm at all.** All rows sharing one norm `R` makes `R` a temperature on the
    action softmax and nothing else: `logit = ||x|| * R * cos(theta)`, so `R` scales every logit
    together and only the angles separate the admissible actions. Leaving it free lets the projector move
    `R` for reasons unrelated to meaning, which reads as the policy suddenly becoming decisive or
    indecisive.

    **Why not the vocabulary head's norm.** An older docstring here said the row norm had to match
    `lm_head` because the two heads compete inside one softmax. They do not. Actions and vocabulary
    are scored as **separate distributions** -- `split_policy.compute_split_policy_outputs` on
    the trainer, the router mask in `dyad_gpu_model_runner` on the engine -- so an action's logit is
    only ever compared against other actions'. Measured on Qwen2.5-3B, `mean(||lm_head row||)` is
    1.089 anyway, so `uniform` and `unit` differ by 8%; the reason to prefer `unit` is that it does
    not make the head's scale a function of whichever backbone supplied the vocabulary.

    For reference, on that model `||x_t||` averages 181, so either setting affords a logit span of
    ~190 while separating six actions well needs about 4. Norm is not the binding constraint;
    the angular spread of the projector's outputs is.
    """
    if scale == "none":
        return head
    if scale == "unit":
        current = head.norm(dim=1, keepdim=True).clamp_min(torch.finfo(head.dtype).tiny)
        return head / current
    if scale == "uniform":
        target = reference.float().norm(dim=1).mean()
        current = head.norm(dim=1, keepdim=True).clamp_min(torch.finfo(head.dtype).tiny)
        return head * (target / current)
    raise ValueError(f"unknown scale {scale!r}; known: none, unit, uniform")



def create_action_head(actor_module: nn.Module, tokenizer, action_config: dict[str, Any], dtype: torch.dtype) -> nn.Linear:
    """The `nn.Linear` that holds the action rows. Its weight is a placeholder, not a head.

    The builder fills these rows before a real forward.
    Zero initialization makes a missed materialization visible as uniform action logits.
    """
    output_embeddings = actor_module.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise ValueError("Dyad action head initialization requires model output embeddings with a weight tensor.")

    head = nn.Linear(
        output_embeddings.weight.shape[1],
        action_config["total_size"],
        bias=False,
        device=output_embeddings.weight.device,
        dtype=dtype,
    )
    with torch.no_grad():
        head.weight.zero_()
    return head
