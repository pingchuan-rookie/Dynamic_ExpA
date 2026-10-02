"""Serve a base, trained text, or full Dyad model through one generation API."""

from __future__ import annotations

import argparse
import json
from contextlib import asynccontextmanager
from dataclasses import fields
from typing import TYPE_CHECKING

from agent_system.inference.backend import ChatCompletionBackend
from agent_system.inference.config import InferenceConfig, resolve_source
from agent_system.inference.engine import VLLMInference

if TYPE_CHECKING:
    from fastapi import FastAPI


def parser() -> argparse.ArgumentParser:
    defaults = InferenceConfig()
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-path", help="HF model directory or native global_step_N checkpoint")
    source.add_argument("--checkpoint", help="Native global_step_N with saved model_config.json")
    source.add_argument("--projector-init", help="Exact Alignment projector, with --model-config")
    p.add_argument("--model-config")
    p.add_argument("--model", default=defaults.model, help="Served model name")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tensor-parallel-size", type=int, default=defaults.tensor_parallel_size)
    p.add_argument(
        "--max-model-len", "--context-length", dest="context_length", type=int, default=defaults.context_length
    )
    p.add_argument("--max-num-seqs", type=int, default=defaults.max_num_seqs)
    p.add_argument("--gpu-memory-utilization", type=float, default=defaults.gpu_memory_utilization)
    p.add_argument("--action-capacity", type=int, default=defaults.action_capacity)
    p.add_argument("--encoder-device", default=defaults.encoder_device)
    p.add_argument("--restore-dir", help="New artifact directory for checkpoint restoration")
    p.add_argument(
        "--safetensors-load-strategy", choices=("prefetch", "lazy", "eager"), default=defaults.safetensors_load_strategy
    )
    p.add_argument("--dry-run", "--check", dest="check", action="store_true")
    return p


def create_app(inference: VLLMInference) -> FastAPI:
    from fastapi import FastAPI, HTTPException

    backend = ChatCompletionBackend(inference)

    @asynccontextmanager
    async def lifespan(app):
        try:
            await inference.start()
            yield
        finally:
            await inference.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": inference.options.model,
                    "object": "model",
                    "owned_by": "local",
                    "root": inference.options.source_root,
                    "identity": inference.identity,
                }
            ],
        }

    @app.get("/v1/dyad/health")
    async def identity():
        return inference.identity

    @app.post("/v1/chat/completions")
    async def generate(payload: dict):
        try:
            return await backend.generate(payload)
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        config = InferenceConfig(
            **{field.name: getattr(args, field.name) for field in fields(InferenceConfig) if field.init}
        )
        if args.check:
            config = resolve_source(config)
            print(
                json.dumps(
                    {
                        "backend": "vllm",
                        "action_interface": config.action_interface,
                        "source": config.source_root,
                        "model": config.model,
                        "gpu_started": False,
                        "context_length": config.context_length,
                        "safetensors_load_strategy": config.safetensors_load_strategy,
                    }
                )
            )
            return 0
        import uvicorn

        uvicorn.run(create_app(VLLMInference(config)), host=args.host, port=args.port, workers=1)
        return 0
    except (ValueError, OSError) as exc:
        p.exit(2, f"inference: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
