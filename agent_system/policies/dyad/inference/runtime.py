"""Restore Dyad weights and attach the action encoder to a vLLM engine.

Shared by standalone inference and the legacy tau HTTP adapter. This module owns
no HTTP protocol, benchmark loop or training process.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_system.inference.config import InferenceConfig
from agent_system.policies.dyad.inference.source import source_metadata

if TYPE_CHECKING:
    from collections.abc import Iterator

    from transformers import PretrainedConfig
    from vllm.v1.engine.async_llm import AsyncLLM


@contextmanager
def encoder_cpu_kernels(device: str, model_config: PretrainedConfig) -> Iterator[None]:
    """Select the installed HF reference kernels for CPU Qwen3.5 construction.

    HF selects CUDA-only optional kernels by package availability, even for CPU
    weights. GatedDeltaNet captures these functions in each module's constructor.
    Restore module globals before creating the separate native policy engine.
    """
    import torch

    if torch.device(device).type != "cpu" or model_config.model_type not in ("qwen3_5", "qwen3_5_text"):
        yield
        return
    from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation

    replacements = {
        "causal_conv1d_fn": None,
        "causal_conv1d_update": None,
        "chunk_gated_delta_rule": None,
        "fused_recurrent_gated_delta_rule": None,
        "FusedRMSNormGated": None,
    }
    previous = {name: getattr(implementation, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(implementation, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(implementation, name, value)


class DyadRuntime:
    def __init__(self, args: InferenceConfig) -> None:
        self.args = args
        if args.restore_dir:
            from agent_system.policies.dyad.inference.checkpoint import validate_restore_directory

            validate_restore_directory(args.restore_dir)
        self.source = source_metadata(args)
        from agent_system.policies.dyad.checkpoint_compat import read_saved_model_config

        self.config = read_saved_model_config(self.source["model_config"])
        self.model = self.config["model"]
        if args.action_capacity < 1:
            raise ValueError("action-capacity must be positive")
        if self.model.get("DYAD_ENCODER_BACKBONE") != "encoder_lm" or self.model.get("DYAD_ENCODER_TRAINING") not in {
            "projector_only",
            "projector_and_encoder_lm",
        }:
            raise ValueError("This backend requires an independent encoder with a supported saved training mode")
        self.engine: AsyncLLM | None = None
        self.directory: tempfile.TemporaryDirectory | None = None

    def prepare_checkpoint(self) -> dict[str, Any]:
        if not self.args.restore_dir:
            raise ValueError("Native AgenticRL requires --restore-dir for lossless tensors and audit manifest")
        from agent_system.policies.dyad.inference.checkpoint import (
            prepare_native,
            validate_restore_directory,
            verify_native_artifact,
        )

        root = validate_restore_directory(self.args.restore_dir)
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            manifest = verify_native_artifact(root, source=self.source)
        else:
            manifest = prepare_native(self.source, root, target_benchmark="request_defined")
        if source_metadata(self.args) != self.source:
            raise ValueError("Native checkpoint changed during restoration")
        self.restore_manifest = manifest
        self.restore_dir = root
        return manifest

    async def start(self) -> None:
        if self.args.checkpoint:
            self.prepare_checkpoint()
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from vllm.engine.arg_utils import AsyncEngineArgs

        from agent_system.policies.dyad.models.action_encoder import LlmActionEncoder
        from agent_system.policies.dyad.models.action_head_factory import llm_encoder_config_from_env
        from agent_system.policies.dyad.rollout.vllm.dyad_async_llm import DyadAsyncLLM
        from agent_system.utils.hf_config import text_vocab_size

        for key in list(os.environ):
            if key.startswith(("DYAD_", "VLLM_")):
                del os.environ[key]
        os.environ.update({k: str(v) for k, v in self.model.items() if k.startswith("DYAD_")})
        os.environ.update(
            VLLM_USE_V2_MODEL_RUNNER="0",
            DYAD_CODEGYM_ALL="1",
            DYAD_DYNAMIC_ACTIONS="1",
            DYAD_ACTION_CAPACITY=str(self.args.action_capacity),
            DYAD_CODEGYM_ACTION_CAPACITY=str(self.args.action_capacity),
            DYAD_ENCODER_REMOTE="0",
            DYAD_ENCODER_DEVICE=self.args.encoder_device,
        )
        os.environ.pop("DYAD_VAL_ACTION_YAML", None)
        os.environ.pop("DYAD_ENCODER_PROJECTOR_INIT", None)
        path = self.model["MODEL_PATH"]
        if self.args.checkpoint:
            os.environ["DYAD_NATIVE_MODEL_CONFIG"] = self.source["model_config"]
            from agent_system.policies.dyad.inference.source import file_digest

            encoder_identity = self.restore_manifest["encoder_identity"]
            for name, expected in encoder_identity["weight_files_sha256"].items():
                if file_digest(Path(encoder_identity["source"]) / name) != expected:
                    raise ValueError("Frozen encoder base weights changed since native restoration")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.vocab_size = text_vocab_size(AutoConfig.from_pretrained(path))
        cfg = llm_encoder_config_from_env()
        os.environ["DYAD_ENCODER_MODEL_PATH"] = cfg.resolved_model_path(path)
        # Real existing encoder implementation, preserving representation and complete
        # per-tool prompts. CPU defaults to float32; no approximate encoder is used.
        with encoder_cpu_kernels(self.args.encoder_device, AutoConfig.from_pretrained(cfg.resolved_model_path(path))):
            self.encoder = LlmActionEncoder(
                cfg.resolved_model_path(path),
                device=self.args.encoder_device,
                dtype=torch.float32 if torch.device(self.args.encoder_device).type == "cpu" else torch.bfloat16,
                max_length=cfg.max_length,
                representation=cfg.representation,
            )
        if self.args.checkpoint:
            from agent_system.policies.dyad.inference.checkpoint import restore_native_encoder

            restore_native_encoder(self.encoder, self.restore_dir, self.source["model_config"])
        self.encoder_cfg = cfg
        self.engine = DyadAsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=str(self.restore_dir / "policy") if self.args.checkpoint else path,
                tensor_parallel_size=self.args.tensor_parallel_size,
                enforce_eager=True,
                async_scheduling=False,
                generation_config="vllm",
                safetensors_load_strategy=self.args.safetensors_load_strategy,
                worker_extension_cls="agent_system.policies.dyad.inference.worker.InferenceWorkerExtension",
                max_model_len=self.args.context_length,
                max_num_seqs=self.args.max_num_seqs,
                gpu_memory_utilization=self.args.gpu_memory_utilization,
                enable_prefix_caching=False,
                dtype="bfloat16",
                language_model_only=True,
            )
        )
        if self.args.checkpoint:
            restored = await self.engine.collective_rpc("dyad_inference_restore_native", args=(str(self.restore_dir),))
        else:
            restored = await self.engine.collective_rpc(
                "dyad_inference_restore_alignment", args=(self.source["source"],)
            )
        if len(restored) != self.args.tensor_parallel_size:
            raise ValueError("Native backend worker count differs from tensor parallel size")
        if any(item["projector_sha256"] != restored[0]["projector_sha256"] for item in restored):
            raise ValueError("Native projector restoration differs across policy workers")
        if source_metadata(self.args) != self.source:
            raise ValueError("Source checkpoint/projector changed while native restoration was in progress")
        self.identity = {
            "protocol": "dyad_inference_v1",
            "backend": "DyadAsyncLLM",
            "source_identity": self.source["identity"],
            "source_identity_sha256": self.source["identity_sha256"],
            "restoration_complete": True,
            **restored[0],
            "context_length": self.args.context_length,
            "action_capacity": self.args.action_capacity,
            "runtime_policy_dtype": "bfloat16",
            "encoder_source": str(self.encoder.model_path),
            "training_benchmark": self.config["benchmark"],
            "target_benchmark": "request_defined",
        }
        self.directory = tempfile.TemporaryDirectory(prefix="dyad-inference-contexts-")

    async def close(self) -> None:
        """Release partially loaded workers and their request-local contexts."""
        try:
            if self.engine is not None:
                self.engine.shutdown()
        finally:
            self.engine = None
            if self.directory is not None:
                self.directory.cleanup()
                self.directory = None
            # Dropping the encoder also releases its device tensors.
            self.encoder = None
