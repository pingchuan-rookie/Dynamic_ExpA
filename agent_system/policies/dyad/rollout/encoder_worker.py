# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Host the sampling encoder replica in Ray; training owns gradients locally."""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Serve versioned encoder contexts for rollout without moving training autograd into RPC.
# Extension point: DyadStepPolicy context preparation / EncoderTrainingMixin.get_per_tensor_param
from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from agent_system.policies.dyad.models.encoder_cache import CachedHidden

DEFAULT_ACTOR_NAME = "dyad_action_encoder"


class EncoderActorImpl:
    """The body of the encoder actor, as a plain class.

    Plain rather than `@ray.remote`-decorated so it can be exercised without a cluster; the decorated
    handle is `EncoderActor` below. The `encoder` parameter is the seam that lets a test drive this
    without downloading a 4B checkpoint.
    """

    def __init__(
        self,
        model_path: str = "",
        *,
        dtype: str = "bfloat16",
        max_length: int = 1024,
        encoder: Any = None,
    ):
        self._calls = 0
        self._weight_version = "base"
        self._pending_version = None
        self._pending_weights = {}
        if encoder is not None:
            self._encoder = encoder
            return

        from agent_system.policies.dyad.models.action_encoder import LlmActionEncoder

        # Ray has already narrowed CUDA_VISIBLE_DEVICES to this policy LLM backbone's own GPUs, so `cuda` here
        # *is* the encoder's first GPU. Reading an absolute ordinal at this point would undo exactly
        # the isolation the policy LLM backbone exists to provide, so it is deliberately not consulted.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._encoder = LlmActionEncoder(
            model_path,
            device=device,
            dtype=getattr(torch, dtype, torch.bfloat16),
            max_length=max_length,
        )

        self._weight_version = getattr(self._encoder, "native_weight_version", "base")

    def begin_weight_update(self, version):
        self._pending_version = version
        self._pending_weights = {}

    def stage_weight(self, version, name, tensor):
        if version != self._pending_version or name in self._pending_weights:
            raise ValueError("Encoder weight update version/key mismatch")
        self._pending_weights[name] = tensor

    def commit_weight_update(self, version):
        if version != self._pending_version:
            raise ValueError("Encoder weight update version mismatch")
        expected = self._encoder.backbone.state_dict()
        if set(expected) != set(self._pending_weights) or any(
                expected[k].shape != v.shape for k, v in self._pending_weights.items()):
            raise ValueError("Incomplete encoder weight update")
        # Keep optimizer-precision weights; inference autocast matches trainer compute.
        self._encoder.backbone.float()
        self._encoder.backbone.load_state_dict(self._pending_weights, strict=True)
        self._encoder.clear_cache()
        self._weight_version = version
        self._pending_version = None
        self._pending_weights = {}
        return version

    def _versioned_path(self, path):
        from pathlib import Path
        if self._weight_version == "base":
            return Path(path)
        path = Path(path)
        return path.with_name(path.stem + "." + self._weight_version + path.suffix)

    def write_action_context(self, path, action_config, prompts, identity):
        from pathlib import Path
        if not identity:
            raise ValueError("Dynamic action context requires encoder identity")
        target = self._versioned_path(path)
        if target.is_file():
            payload = torch.load(target, map_location="cpu", weights_only=True)
            if payload.get("identity") != identity or payload.get("action_config") != action_config:
                raise ValueError("Cached action context identity/schema mismatch")
            return str(target)
        return self._write_context(path, action_config, prompts, identity)

    def write_codegym_context(self, path, action_config, prompts):
        return self._write_context(path, action_config, prompts)

    def _write_context(self, path, action_config, prompts, identity=None):
        import os
        import tempfile
        from pathlib import Path
        target = self._versioned_path(path)
        if target.is_file():
            return str(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Full CodeGym has thousands of interfaces: do not retain every hidden tensor on GPU.
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=torch.device(getattr(self._encoder, "device", "cpu")).type == "cuda"
                            and self._weight_version != "base"):
            cached = self._encoder.encode_task_hidden(prompts)
        payload = {"action_config": action_config, "hidden": cached.hidden.detach().cpu(),
                   "mask": cached.mask.detach().cpu(), "fingerprint": cached.fingerprint,
                   "prompts": list(prompts), "encoder_version": self._weight_version}
        if identity is not None:
            payload["identity"] = identity
        fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".pt")
        os.close(fd)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return str(target)

    def hidden_size(self) -> int:
        return int(self._encoder.hidden_size)

    def gpu_ids(self) -> list[int]:
        """The physical GPUs Ray gave this actor.

        Reported rather than inferred from `CUDA_VISIBLE_DEVICES`, which Ray has already rewritten:
        the remapped view says `0` no matter which card it is, so it cannot answer "are the encoder
        and the policy LLM backbone actually on different GPUs".
        """
        try:
            import ray

            return [int(x) for x in ray.get_gpu_ids()]
        except Exception:  # noqa: BLE001  outside a Ray worker there is nothing to report
            return []

    def actor_id(self) -> str:
        """This policy LLM backbone's own id, via the public runtime context.

        Asked of the policy LLM backbone rather than read off the handle so the check does not depend on a private
        attribute of Ray's ActorHandle, which has been renamed before.
        """
        try:
            import ray

            return str(ray.get_runtime_context().get_actor_id())
        except Exception:  # noqa: BLE001  not running as a policy LLM backbone
            return ""

    def call_count(self) -> int:
        """How many RPCs actually reached this actor. The steady-state target is that this stops
        increasing after cold start."""
        return self._calls

    def encode_hidden(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor, str]:
        """`prompts -> (hidden, mask, fingerprint)`, always on CPU.

        CPU because the caller lives in a different process that does not own this policy LLM backbone's devices:
        handing back a CUDA tensor makes Ray's deserialisation touch a device the receiver has no
        claim on. The receiver moves it to its own device.

        The fingerprint is computed **here** and carried back verbatim rather than recomputed by the
        caller. It covers the encoder path and tokenizer name, which are this process's facts; a
        caller recomputing it from its own tokenizer would produce a second, silently different key
        for the same tensors.
        """
        self._calls += 1
        cached = self._encoder.encode_hidden(prompts)
        return cached.hidden.detach().cpu(), cached.mask.detach().cpu(), cached.fingerprint



def _ray_get(ref):
    import ray

    return ray.get(ref)



class RemoteLlmActionEncoder:
    """Client-side stand-in for `LlmActionEncoder`, backed by a policy LLM backbone handle.

    Deliberately the same three-member surface -- `hidden_size`, `encode_hidden`, `clear_cache` --
    so `build_action_head` never learns whether the encoder is local or remote. A second code path
    through the head builder is how the "three head-init sites and the last one always gets missed"
    bug recorded in the action_encoder docstring happens again.
    """

    def __init__(
        self,
        handle: Any,
        *,
        hidden_size: int,
        device: str = "cuda",
        get: Optional[Callable[[Any], Any]] = None,
    ):
        self._handle = handle
        self._hidden_size = int(hidden_size)
        self.device = device
        self._get = get or _ray_get
        self._cache: dict[tuple[str, ...], CachedHidden] = {}

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def encode_hidden(self, prompts: list[str], *, use_cache: bool = True) -> CachedHidden:
        if not prompts:
            raise ValueError("encode_hidden needs at least one prompt")
        key = tuple(prompts)
        if use_cache and key in self._cache:
            return self._cache[key]

        hidden, mask, fingerprint = self._get(self._handle.encode_hidden.remote(list(prompts)))
        cached = CachedHidden(
            hidden=hidden.to(self.device),
            mask=mask.to(self.device),
            fingerprint=fingerprint,
        )
        if use_cache:
            self._cache[key] = cached
        return cached

    def clear_cache(self) -> None:
        self._cache.clear()

    def call_count(self) -> int:
        """RPCs that reached the policy LLM backbone. Used by the E2E check to prove the steady state is 0/step."""
        return int(self._get(self._handle.call_count.remote()))

    def gpu_ids(self) -> list[int]:
        return list(self._get(self._handle.gpu_ids.remote()))



def get_or_create_encoder_actor(
    cfg,
    actor_model_path: str,
    *,
    name: Optional[str] = None,
):
    """The named encoder actor, created on first call and shared afterwards.

    Refuses rather than degrades in both failure modes below. A remote encoder that quietly became a
    local one is the bug this module exists to fix, and it is invisible in the metrics.
    """
    import ray

    if not ray.is_initialized():
        raise RuntimeError(
            "DYAD_ENCODER_REMOTE=1 needs a running Ray cluster; ray.is_initialized() is False. "
            "The encoder actor is created from inside a worker, so Ray is normally already up -- "
            "getting here means this process is not a Ray worker. Falling back to an in-process "
            "encoder is not done on purpose: it would share the policy LM's GPUs, which is exactly the "
            "configuration this flag exists to avoid."
        )
    num_gpus = int(getattr(cfg, "num_gpus", 0) or 0)
    if num_gpus <= 0:
        raise ValueError(
            f"DYAD_ENCODER_NUM_GPUS must be >= 1 when DYAD_ENCODER_REMOTE=1, got {num_gpus}. "
            "A zero-GPU encoder actor runs the backbone on CPU: roughly two orders of magnitude "
            "slower, and it raises nothing."
        )

    actor_name = name or getattr(cfg, "actor_name", "") or DEFAULT_ACTOR_NAME
    encoder_actor_cls = ray.remote(EncoderActorImpl)
    return encoder_actor_cls.options(
        name=actor_name,
        # Shared across FSDP workers. With get_if_exists the constructor arguments of every call
        # after the first are ignored, which is the intent: one backbone, many clients.
        get_if_exists=True,
        num_gpus=num_gpus,
        # No lifetime="detached": the encoder should die with the training job. A detached actor
        # would survive a crashed run and hold its GPUs against the next one.
    ).remote(
        cfg.resolved_model_path(actor_model_path),
        dtype=cfg.dtype,
        max_length=cfg.max_length,
    )



def shutdown_encoder_actor(name: str = DEFAULT_ACTOR_NAME) -> bool:
    """Kill the named encoder actor if it exists. Returns whether one was found."""
    import ray

    if not ray.is_initialized():
        return False
    try:
        handle = ray.get_actor(name)
    except Exception:  # noqa: BLE001  ValueError in current Ray, but the name has moved before
        return False
    ray.kill(handle)
    return True



def build_remote_encoder(cfg, actor_model_path: str, actor_hidden: int, *, device: str = "cuda"):
    """Build a remote encoder client and a DirectActionHead on the local policy device.

    Only frozen representations cross the actor boundary; the trainable projector remains local.
    """
    from agent_system.policies.dyad.models.action_head import DirectActionHead

    handle = get_or_create_encoder_actor(cfg, actor_model_path)
    hidden_size = int(_ray_get(handle.hidden_size.remote()))
    encoder = RemoteLlmActionEncoder(handle, hidden_size=hidden_size, device=device)
    residual_head = DirectActionHead(
        cfg.projector,
        hidden_size,
        actor_hidden,
        scale=cfg.scale,
        projector_kwargs=cfg.projector_kwargs,
    ).to(device)
    return encoder, residual_head

