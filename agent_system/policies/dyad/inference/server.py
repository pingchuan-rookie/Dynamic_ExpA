"""Serial, local HTTP boundary around the native Dyad vLLM engine.

Run in dyad-verl, not in either tau venv. Restoring never starts training.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from agent_system.inference.config import InferenceConfig
from agent_system.policies.dyad.inference.runtime import DyadRuntime
from agent_system.policies.dyad.inference.source import digest

PROTOCOL = "dyad_tau_json_v1"


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--projector-init")
    p.add_argument("--model-config")
    p.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--context-length", type=int, default=65536)
    p.add_argument(
        "--action-capacity", type=int, default=64, help="Serving capacity independent of training head width"
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--encoder-device", default="cpu", help="CPU avoids encoder/policy GPU contention")
    p.add_argument("--restore-dir", help="New audit artifact directory for lossless native AgenticRL restoration")
    p.add_argument(
        "--prepare-only", action="store_true", help="Restore native tensors and manifest on CPU, without starting GPU"
    )
    p.add_argument("--check", action="store_true", help="Validate source configuration without model loading or GPUs")
    return p


class TraceValidationError(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


class Backend(DyadRuntime):
    """Legacy tau request adapter over the shared Dyad vLLM runtime."""

    def __init__(self, args):
        super().__init__(
            InferenceConfig(
                max_num_seqs=1,
                **{
                    name: getattr(args, name)
                    for name in (
                        "checkpoint",
                        "projector_init",
                        "model_config",
                        "restore_dir",
                        "context_length",
                        "action_capacity",
                        "gpu_memory_utilization",
                        "encoder_device",
                    )
                },
            )
        )
        self.lock = asyncio.Lock()
        self.contexts = {}

    async def start(self):
        await super().start()
        self.identity["protocol"] = PROTOCOL

    async def context(self, tools):
        import torch

        from agent_system.policies.dyad.inference.schema import compile_tools
        from agent_system.policies.dyad.models.action_head_factory import encoder_prompts

        key = digest(tools)
        if key not in self.contexts:
            cfg = compile_tools(self.tokenizer, self.vocab_size, tools, self.args.action_capacity)
            prompts = encoder_prompts(cfg, description=self.encoder_cfg.description)
            hidden = await asyncio.to_thread(self.encoder.encode_task_hidden, prompts)
            path = str(Path(self.directory.name) / (key + ".pt"))
            torch.save({"action_config": cfg, "hidden": hidden.hidden.cpu(), "mask": hidden.mask.cpu()}, path)
            attestation = (await self.engine.collective_rpc("dyad_inference_attest_head", args=(path,)))[0]
            if attestation["projector_sha256"] != self.identity["projector_sha256"]:
                raise ValueError("Projector changed between restore and target head construction")
            self.contexts[key] = (cfg, path, attestation)
        return self.contexts[key]

    async def act(self, payload):
        from vllm import SamplingParams

        from agent_system.policies.dyad.inference.schema import verify_trace

        if payload.get("source_identity_sha256") != self.source["identity_sha256"]:
            raise ValueError("Request checkpoint identity mismatch")
        tools = payload["tools"]
        if payload.get("schema_sha256") != digest(tools):
            raise ValueError("Request schema fingerprint mismatch")
        from agent_system.utils.thinking import resolve_chat_template_kwargs

        thinking = payload.get("thinking", "default")
        if thinking not in ("default", "on", "off"):
            raise ValueError("thinking must be default, on, or off")
        kwargs = {} if thinking == "default" else {"enable_thinking": thinking == "on"}
        kwargs = resolve_chat_template_kwargs(
            kwargs,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        messages = payload["messages"]
        async with self.lock:
            cfg, path, head = await self.context(tools)
            tokens = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False, **kwargs
            )
            if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
                raise ValueError("Chat tokenizer must produce one nonempty list of integer token IDs")
            from agent_system.rollout.prompt import strip_thinking_prefill

            tokens = strip_thinking_prefill(tokens, self.tokenizer)
            maximum = int(payload["max_tokens"])
            if maximum < 1 or len(tokens) + maximum > self.args.context_length:
                raise ValueError("Prompt plus requested output exceeds serving context; no truncation")
            params = SamplingParams(
                temperature=float(payload["temperature"]),
                max_tokens=maximum,
                seed=int(payload["seed"]),
                extra_args={"dyad_codegym_context": path},
            )
            final = None
            async for output in self.engine.generate({"prompt_token_ids": tokens}, params, uuid.uuid4().hex):
                final = output
            if final is None:
                raise ValueError("Native engine returned no output")
            content = getattr(final, "action_content", None)
            if not isinstance(content, dict) or len(content) != 1:
                raise TraceValidationError(
                    "Native engine must return exactly one request-local action trace",
                    {
                        "identity": self.identity,
                        "head": head,
                        "action_content": content,
                        "text": final.outputs[0].text,
                        "token_ids": list(final.outputs[0].token_ids),
                        "finish_reason": final.outputs[0].finish_reason,
                        "stop_reason": final.outputs[0].stop_reason,
                        "compiled_schema_sha256": digest(cfg),
                        "schema_sha256": digest(tools),
                    },
                )
            trace = next(iter(content.values()))
            text = final.outputs[0].text
            try:
                selected = verify_trace(trace, cfg)
            except ValueError as exc:
                raise TraceValidationError(
                    str(exc),
                    {
                        "identity": self.identity,
                        "schema_sha256": digest(tools),
                        "compiled_schema_sha256": digest(cfg),
                        "action_content": trace,
                        "head": head,
                        "text": text,
                        "finish_reason": final.outputs[0].finish_reason,
                        "prompt_tokens": len(tokens),
                        "completion_tokens": len(final.outputs[0].token_ids),
                    },
                ) from exc
            # Official parsers remain responsible for complete JSON and tool argument validation.
            return {
                "identity": self.identity,
                "schema_sha256": digest(tools),
                "compiled_schema_sha256": digest(cfg),
                "action_content": trace,
                "selection": selected,
                "head": head,
                "text": text,
                "finish_reason": final.outputs[0].finish_reason,
                "prompt_tokens": len(tokens),
                "completion_tokens": len(final.outputs[0].token_ids),
            }


def main():
    args = parser().parse_args()
    backend = Backend(args)
    if args.check:
        print(
            json.dumps(
                {
                    "status": "checked_offline",
                    "source": backend.source,
                    "gpu_started": False,
                    "agentic_rl_restore_supported_topology": "two-rank DTensor",
                }
            )
        )
        return
    if args.prepare_only:
        if not args.checkpoint:
            raise ValueError("--prepare-only requires a native AgenticRL --checkpoint")
        manifest = backend.prepare_checkpoint()
        print(
            json.dumps(
                {
                    "status": "native_restored_cpu",
                    "manifest": str(backend.restore_dir / "manifest.json"),
                    "tensor_count": manifest["tensor_count"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "gpu_started": False,
                }
            )
        )
        return
    from contextlib import asynccontextmanager

    import uvicorn
    from fastapi import FastAPI, HTTPException

    @asynccontextmanager
    async def lifespan(app):
        try:
            await backend.start()
            yield
        finally:
            await backend.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/v1/dyad/health")
    async def health():
        return backend.identity

    @app.post("/v1/dyad/act")
    async def act(payload: dict):
        try:
            return await backend.act(payload)
        except TraceValidationError as exc:
            print("Dyad trace validation failed: " + str(exc), flush=True)
            raise HTTPException(status_code=422, detail={"error": str(exc), "evidence": exc.evidence}) from exc
        except ValueError as exc:
            print("Dyad request validation failed: " + str(exc), flush=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
