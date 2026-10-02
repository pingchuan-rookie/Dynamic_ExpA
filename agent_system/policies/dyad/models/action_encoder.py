# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Encode action descriptions with an owned or shared LLM backbone."""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Encode action definitions for training replay and sampling contexts.
# Extension point: build_llm_encoder -> LlmActionEncoder -> DirectActionHead
from __future__ import annotations

import hashlib
from typing import Optional

import torch
from torch import nn

from agent_system.utils.hf_config import text_hidden_size
from agent_system.policies.dyad.models.encoder_cache import CachedHidden, prompt_fingerprint


class LlmActionEncoder(nn.Module):
    """Encode action prompts with an owned or shared LLM backbone.

    Frozen representations can be cached. Trainable backbones bypass that cache
    so the action-selection loss retains its gradient path.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 1024,
        attn_implementation: Optional[str] = None,
        representation: str = "final_layer_hidden_states",
        backbone: Optional[nn.Module] = None,
        tokenizer=None,
        freeze: bool = True,
    ):
        """`pooling` selects **what the projector pools over** (settings.md section 3):

            final_layer_hidden_states  -- run the whole backbone, take the final layer
            input_embeddings    -- the embedding layer's output, no transformer blocks at all

        input_embeddings is not "final_layer_hidden_states with fewer layers": it never runs attention, so a token's representation does
        not depend on any other token. That is the point of having it as a control -- it separates
        "the projector needs contextualised tokens" from "the projector just needs the words".

        `backbone`/`tokenizer` are the seam for backbone=policy_lm, which reads the **policy LLM backbone** instead of
        loading a second one. Passing them in means this class never calls `from_pretrained`, so the
        run holds exactly one LLM. `freeze=False` leaves the caller's module alone -- that
        backbone is the policy LLM backbone, and freezing it here would silently override the training_schedule.
        """
        super().__init__()

        self.model_path = str(model_path)
        self.max_length = int(max_length)
        self.representation = str(representation)
        if self.representation not in ("final_layer_hidden_states", "input_embeddings"):
            raise ValueError(
                f"unknown representation {representation!r}; known: final_layer_hidden_states, input_embeddings"
            )

        # Recorded before the branch: "did this encoder load its own LLM" is the property that
        # decides whether the run holds one model or two, and it is worth asserting in tests.
        self._owns_backbone = backbone is None

        if backbone is not None:
            # Shared-backbone path (backbone=policy_lm). No second model, no second tokenizer.
            if tokenizer is None:
                raise ValueError(
                    "a shared backbone needs its tokenizer too: the ids fed to it must come from "
                    "the same vocabulary, and silently loading a second tokenizer from model_path "
                    "would produce ids that index the wrong rows."
                )
            self.backbone = backbone
            self.tokenizer = tokenizer
        else:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            kwargs = {"dtype": dtype}
            if attn_implementation:
                kwargs["attn_implementation"] = attn_implementation
            # AutoModel = backbone only. See the module docstring for why not AutoModelForCausalLM.
            self.backbone = AutoModel.from_pretrained(self.model_path, **kwargs).to(device).eval()

        if freeze:
            # projector_only: the whole backbone is frozen. Training the encoder LLM backbone unfreezes
            # exactly the LoRA parameters and nothing else.
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        self.device = device
        self._cache: dict[str, CachedHidden] = {}
        # Caching is only sound while the encoder is a constant function of the prompts. LoRA makes
        # its weights move, so `apply_lora` turns this off.
        self.cacheable = True
        self.lora_enabled = False
        self.full_finetune = False
        self._hidden_size = text_hidden_size(self.backbone.config)
        import os
        native = os.environ.get("DYAD_NATIVE_RESTORE_DIR")
        if native and self._owns_backbone:
            from agent_system.policies.dyad.inference.checkpoint import restore_native_encoder
            restore_native_encoder(self, native, os.environ.get("DYAD_NATIVE_MODEL_CONFIG"))


    @property
    def owns_backbone(self) -> bool:
        """False when the backbone was handed in (backbone=policy_lm)."""
        return self._owns_backbone

    @property
    def hidden_size(self) -> int:
        # Read from the config once, at construction, so it still answers after
        # `release_backbone`. Callers ask for it when sizing the projector, which outlives the
        # backbone by design.
        return self._hidden_size

    def encode_hidden(self, prompts: list[str], *, use_cache: bool = True) -> CachedHidden:
        """Prompts -> `[A, T, H]` last-layer hidden states plus the `[A, T]` attention mask.

        **Whether this runs under `no_grad` depends on whether the encoder is being trained.**
        It used to be an unconditional `@torch.no_grad()` decorator, which is right for a frozen
        encoder -- no graph, no activations retained, and this is by far the common path. But it
        also silently defeated full-parameter encoder training: the caller routed the head's
        gradient through `encode_hidden(use_cache=False)`, the parameters were unfrozen and sat in
        the optimizer, and every weight's `.grad` stayed `None` because the graph was severed here,
        one frame below. Nothing raised; the run trained the projector and reported the encoder as
        trainable.

        """
        import contextlib

        # Build an autograd graph only when the encoder is trainable; frozen encoding retains no activations.
        trains = getattr(self, "full_finetune", False) or getattr(self, "lora_enabled", False)
        ctx = contextlib.nullcontext() if trains else torch.no_grad()
        with ctx:
            return self._encode_hidden_inner(prompts, use_cache=use_cache)

    def _encode_hidden_inner(self, prompts: list[str], *, use_cache: bool = True) -> CachedHidden:
        if not prompts:
            raise ValueError("encode_hidden needs at least one prompt")
        if self.backbone is None:
            raise RuntimeError(
                "this encoder's backbone was released after its hidden states were read out; "
                "it cannot encode again. Whoever needs a fresh encode (a second schema, a LoRA'd "
                "encoder) must keep the backbone -- see release_backbone."
            )
        # The representation is part of the key: the two targets are different functions of the same
        # prompts, and a shared cache would serve one where the other was asked for -- same shape,
        # same dtype, no error anywhere.
        fingerprint = prompt_fingerprint(
            prompts, self.tokenizer.name_or_path, f"{self.model_path}#{self.representation}"
        )
        use_cache = use_cache and self.cacheable
        if use_cache and fingerprint in self._cache:
            return self._cache[fingerprint]

        batch = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        input_ids = batch["input_ids"].to(self.device)
        mask = batch["attention_mask"].to(self.device)
        if self.full_finetune and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            hidden = checkpoint(self._represent, input_ids, mask, use_reentrant=False)
        else:
            hidden = self._represent(input_ids, mask)
        cached = CachedHidden(hidden=hidden, mask=mask, fingerprint=fingerprint)
        if use_cache:
            self._cache[fingerprint] = cached
        return cached

    def encode_task_hidden(self, prompts: list[str]) -> CachedHidden:
        """Encode a complete task catalogue without dropping the target action definition.

        Dynamic CodeGym catalogues vary in length. Right-truncating all prompts to a
        fixed prefix can make every action's representation identical. Encode one row
        at a time to bound activations, preserving all tokens up to the model's limit.
        """
        from agent_system.utils.hf_config import _text_config
        limit = getattr(_text_config(self.backbone.config), "max_position_embeddings", None)
        lengths = [len(self.tokenizer.encode(p, add_special_tokens=True)) for p in prompts]
        if limit is not None and max(lengths) > limit:
            raise ValueError(f"CodeGym action definitions need {max(lengths)} tokens; encoder limit is {limit}")
        previous = self.max_length
        rows = []
        try:
            for prompt, length in zip(prompts, lengths, strict=True):
                self.max_length = length
                rows.append(self.encode_hidden([prompt], use_cache=False))
        finally:
            self.max_length = previous
        width = max(r.hidden.shape[1] for r in rows)
        hidden = torch.cat([torch.nn.functional.pad(r.hidden, (0, 0, 0, width-r.hidden.shape[1])) for r in rows])
        mask = torch.cat([torch.nn.functional.pad(r.mask, (0, width-r.mask.shape[1])) for r in rows])
        fingerprint = hashlib.sha256("".join(r.fingerprint for r in rows).encode()).hexdigest()
        return CachedHidden(hidden, mask, fingerprint)

    def unfreeze_backbone(self) -> int:
        """Full-parameter training of the encoder LLM backbone. Returns the trainable parameter count.

        The design document settles on full-parameter rather than LoRA for this dimension, so this
        is the method the `projector_and_encoder_llm` setting actually wants; `apply_lora` stays as
        the parameter-efficient alternative and is not on this path.

        Only for backbone=encoder_lm, which owns its backbone. backbone=policy_lm's is the policy LLM backbone, and
        "train that backbone" is a decision the training_schedule makes, not this switch.

        Two consequences, both invisible if missed:

          - `encode_hidden` caches, keyed on the prompts. A trainable encoder is no longer a
            constant function of its prompts -- its weights move every step -- so the cache is
            dropped here and must stay off. Left on, every step after the first reads the hidden
            states from before the first update, the encoder weights keep changing, and nothing
            downstream ever sees the change.
          - The caller still has to put these parameters in an optimizer and all-reduce their
            gradients. The engine's EncoderTrainingMixin owns that lifecycle.
        """
        if not self._owns_backbone:
            raise RuntimeError(
                "unfreeze_backbone is for an encoder that owns its backbone. This one shares the "
                "policy LM's, so training it means training the policy LM -- decided by the training_schedule."
            )
        if self.backbone is None:
            raise RuntimeError(
                "the backbone was already released, so there is nothing left to train. "
                "release_backbone is only correct for a frozen encoder."
            )
        for param in self.backbone.parameters():
            param.requires_grad_(True)
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        if trainable == 0:
            raise RuntimeError(
                "unfreeze_backbone produced no trainable parameter, which should be impossible "
                "for a loaded model -- something else froze it after construction."
            )
        self.cacheable = False
        self._cache.clear()
        self.full_finetune = True
        return trainable

    def apply_lora(self, rank: int = 16, alpha: int = 32, dropout: float = 0.0,
                   target_modules=None) -> int:
        """Attach LoRA to the encoder LLM backbone. Returns the trainable parameter count.

        Only for backbone=encoder_lm, which owns its backbone. backbone=policy_lm's is the policy LLM backbone, and
        "train that backbone" is a decision the training_schedule makes, not this switch; `build_llm_encoder`
        refuses that combination before reaching here.

        Two consequences that are easy to miss and impossible to see afterwards:

          - `encode_hidden` caches. A LoRA'd encoder is no longer a constant function of the
            prompts, so the cache is cleared here and must stay off -- otherwise every step
            after the first reads the pre-LoRA hidden states and the LoRA weights, having no effect
            on anything, still accumulate gradients and still look like they are training.
          - the base weights must end up frozen. `get_peft_model` does that itself, but it is
            asserted rather than assumed: a base that stayed trainable turns "LoRA the encoder"
            into a full fine-tune of a second LLM, which fits in memory and shows up only as a
            surprisingly good run that cannot be reproduced from the configuration.
        """
        if not self.owns_backbone:
            raise RuntimeError(
                "apply_lora is for an encoder that owns its backbone. This one shares the "
                "policy LM's, so training it means training the policy LM -- decided by the training_schedule."
            )
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=int(rank),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            bias="none",
            target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.backbone = get_peft_model(self.backbone, config)
        self.lora_enabled = True
        # The encoder is no longer a pure function of (prompts, weights): its weights move.
        self.clear_cache()
        self.cacheable = False

        trainable = [(n, p) for n, p in self.backbone.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "LoRA produced no trainable parameter. target_modules probably matched nothing in "
                f"{type(self.backbone).__name__}; nothing would be updated and no error would follow."
            )
        leaked = [n for n, _ in trainable if "lora_" not in n]
        if leaked:
            raise RuntimeError(
                f"{len(leaked)} non-LoRA parameters are still trainable, e.g. {leaked[:3]}. That is "
                "a full fine-tune of the encoder LLM wearing LoRA's name."
            )
        return sum(p.numel() for _, p in trainable)

    def _represent(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """`[A, T]` ids -> `[A, T, H]`, by whichever representation this encoder was built for."""
        if self.representation == "input_embeddings":
            # input_embeddings: the embedding table only. `get_input_embeddings()` is the one accessor every
            # transformers model exposes for it, including the VL wrappers whose text model is
            # nested a level down -- reaching for `.model.embed_tokens` works until it does not.
            embed = self.backbone.get_input_embeddings()
            if embed is None:
                raise RuntimeError(
                    f"{type(self.backbone).__name__} exposes no input embeddings, so "
                    "representation=input_embeddings cannot be built for it."
                )
            return embed(input_ids)
        module = self._hidden_state_module()
        out = module(input_ids=input_ids, attention_mask=mask, use_cache=False)
        # last_hidden_state is the upstream Transformers field, independent of our representation enum.
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError(
                f"{type(module).__name__} returned no last_hidden_state. A causal-LM head "
                "returns logits instead; the encoder needs the backbone (AutoModel), see the "
                "module docstring."
            )
        return hidden

    def _hidden_state_module(self) -> nn.Module:
        """The part of the backbone that returns hidden states, with the LM head left off.

        backbone=encoder_lm loads `AutoModel`, which already is exactly that. backbone=policy_lm shares the **actor**, and the
        actor is an `AutoModelForCausalLM` -- calling it returns logits and no hidden state at all:

            RuntimeError: Qwen2ForCausalLM returned no final_layer_hidden_states

        which is the same reason task item 3 says the encoder LLM backbone needs no lm_head. Unwrapping to
        the inner transformer is what makes final_layer_hidden_states available on a shared backbone.

        `output_hidden_states=True` would also produce the hidden states, but only after running
        `lm_head`: `[A, T, vocab]` over a whole catalogue is tens of gigabytes for a tensor that is
        discarded on the next line. The inner transformer is the same submodule either way, so
        gradients still reach the policy LLM backbone whenever the training_schedule says they should.

        Restricted to the shared-backbone path on purpose. An owned backbone may be wrapped by peft
        when the encoder LLM backbone trains, and unwrapping *that* would call `LoraModel` directly, skipping
        the wrapper that peft relies on -- a silent behaviour change on a path with no problem to fix.
        """
        if self._owns_backbone:
            return self.backbone
        inner = getattr(self.backbone, getattr(self.backbone, "base_model_prefix", ""), None)
        if isinstance(inner, nn.Module) and inner is not self.backbone:
            return inner
        return self.backbone

    def release_backbone(self) -> None:
        """Free the owned backbone once its hidden states have been read out.

        Not an optimisation. With backbone=encoder_lm the encoder LLM backbone is a second 3-4B model that
        has exactly one job -- produce `[A, T, H]` for a fixed prompt list -- and afterwards it sits
        on the GPU for the rest of the run. The engine sleeps and wakes around every weight sync, and
        its wake-up goes through an allocator with nowhere to fall back to:

            CUDA Error: out of memory at cumem_allocator.cpp:163
              ... checkpoint_manager.update_weights -> actor_rollout_update_weights

        Measured with backbone=encoder_lm + training_schedule=joint_optimization at three different `gpu_memory_utilization`
        values: the resident second LLM is what makes the two-LLM settings unrunnable next to the
        engine, not the training_schedule.

        After this the encoder cannot encode again, and `encode_hidden` says so rather than serving
        a stale cache -- "the ablation ran but read last schema's hidden states" is precisely the
        kind of result that looks fine.
        """
        if not self._owns_backbone:
            raise RuntimeError(
                "release_backbone would delete a backbone this encoder does not own. With "
                "backbone=policy_lm that backbone is the policy LM, and freeing it would delete the policy."
            )
        self.backbone = None
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_cache(self) -> None:
        self._cache.clear()

