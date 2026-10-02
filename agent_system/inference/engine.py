"""Direct vLLM inference for text and Dyad, independent of HTTP and evaluators."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from agent_system.inference.config import ActionArguments, InferenceConfig, resolve_source, validate_actions
from agent_system.policies.dyad.inference.source import digest

if TYPE_CHECKING:
    from vllm import SamplingParams
    from vllm.outputs import RequestOutput
    from vllm.v1.engine.async_llm import AsyncLLM

    from agent_system.policies.dyad.inference.runtime import DyadRuntime


@dataclass(frozen=True)
class GenerationResult:
    """The complete vLLM output plus verified request-local Dyad sampling evidence."""

    request_id: str
    output: RequestOutput
    dyad: dict[str, Any]


class VLLMInference:
    """Own one vLLM engine and restore weights once, without a trainer or server.

    Both Python callers and the HTTP adapter submit token IDs and SamplingParams.
    Action contexts are immutable after publication; only encoding/publication is
    serialized. vLLM schedules concurrent generation requests on the same engine.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.options = config
        self.engine: AsyncLLM | None = None
        self.native: DyadRuntime | None = None
        self.contexts: dict[str, tuple[dict, str, dict]] = {}
        self.lock = asyncio.Lock()
        self._started = False
        self._closed = False

    async def __aenter__(self) -> VLLMInference:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def start(self) -> None:
        if self._started or self._closed:
            raise RuntimeError("Create a new VLLMInference instance for a new engine lifecycle")
        self._started = True
        try:
            self.options = resolve_source(self.options)
            # vLLM's JIT helpers must use the active Python environment's executables.
            os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
            await self._start()
        except BaseException:
            await self.close()
            raise

    async def _start(self) -> None:
        from transformers import AutoTokenizer
        from vllm.engine.arg_utils import AsyncEngineArgs

        if self.options.action_interface == "dyad":
            from agent_system.policies.dyad.inference.runtime import DyadRuntime

            self.native = DyadRuntime(self.options)
            await self.native.start()
            self.engine, self.tokenizer = self.native.engine, self.native.tokenizer
            self.identity = {**self.native.identity, "action_interface": "dyad", "protocol": "dyad_inference_v1"}
        else:
            if self.options.checkpoint:
                from agent_system.inference.export_policy import export_policy

                policy_path = str(Path(self.options.restore_dir) / "policy")
                export_policy(self.options.checkpoint, policy_path)
                self.options.model_path = policy_path
            from vllm.v1.engine.async_llm import AsyncLLM

            self.tokenizer = AutoTokenizer.from_pretrained(self.options.model_path)
            self.engine = AsyncLLM.from_engine_args(
                AsyncEngineArgs(
                    model=self.options.model_path,
                    tensor_parallel_size=self.options.tensor_parallel_size,
                    max_model_len=self.options.context_length,
                    max_num_seqs=self.options.max_num_seqs,
                    gpu_memory_utilization=self.options.gpu_memory_utilization,
                    dtype="bfloat16",
                    language_model_only=True,
                    generation_config="vllm",
                    safetensors_load_strategy=self.options.safetensors_load_strategy,
                )
            )
            from agent_system.inference.policy_identity import inspect_policy_weights

            self.identity = {
                "backend": "vllm",
                "action_interface": "text",
                "restoration_complete": True,
                "weights": inspect_policy_weights(Path(self.options.model_path)),
            }
        if self.native is None:
            from agent_system.policies.dyad.inference.source import file_digest

            policy_path = Path(self.options.model_path)
            self.identity["policy_path"] = str(policy_path.resolve())
            self.identity["weight_files_sha256"] = {
                path.name: file_digest(path)
                for path in sorted(policy_path.iterdir())
                if path.is_file() and path.suffix in {".safetensors", ".json", ".jinja", ".model"}
            }
        from agent_system.policies.dyad.inference.source import file_digest

        project = Path(__file__).resolve().parents[2]
        self.identity["code_files_sha256"] = {
            str(path.relative_to(project)): file_digest(path)
            for path in sorted((project / "agent_system").rglob("*.py"))
        }
        self.identity.update(
            model=self.options.model, root=self.options.source_root, context_length=self.options.context_length
        )

    async def close(self) -> None:
        """Release resources even if loading failed; repeated close is harmless."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.native is not None:
                await self.native.close()
            elif self.engine is not None:
                self.engine.shutdown()
        finally:
            self.engine = None
            self.native = None
            self.contexts.clear()

    def compile_actions(self, tools: list[dict], args: ActionArguments) -> dict:
        """Compile with the same tokenizer, schema compiler and candidates as training."""
        from agent_system.policies.dyad.actions.codegym_tasks import compile_task
        from agent_system.policies.dyad.actions.schema_config import SCHEMAS_DIR, compile_schema_file

        capacity = self.options.action_capacity
        if args.get("schema_name"):
            path = (SCHEMAS_DIR / args["schema_name"]).resolve()
            if not path.is_relative_to(SCHEMAS_DIR.resolve()):
                raise ValueError("schema_name must identify a bundled action schema")
            cfg = compile_schema_file(self.tokenizer, self.native.vocab_size, path)
            if cfg["total_size"] > capacity:
                raise ValueError("Action schema exceeds serving action capacity")
            cfg["total_size"] = capacity
            return cfg
        if args.get("schema"):
            return compile_task(
                self.tokenizer, self.native.vocab_size, args["schema"], capacity, step_protocol_version=2
            )
        protocol = args.get("protocol", "native_tools")
        if protocol == "tau_json":
            from agent_system.policies.dyad.inference.schema import compile_tools

            virtual = [t["function"] for t in tools if t["function"]["name"] == "respond"]
            if len(virtual) > 1:
                raise ValueError("Duplicate respond action")
            return compile_tools(
                self.tokenizer,
                self.native.vocab_size,
                [t for t in tools if t["function"]["name"] != "respond"],
                capacity,
                respond_definition=virtual[0] if virtual else None,
            )
        if protocol != "native_tools":
            raise ValueError(f"Unsupported action protocol: {protocol}")
        from agent_system.policies.dyad.actions.native_tools import compile_tools

        return compile_tools(self.tokenizer, self.native.vocab_size, tools, capacity)

    async def context(self, tools: list[dict], args: ActionArguments) -> tuple[dict, str, dict]:
        import torch

        from agent_system.policies.dyad.models.action_head_factory import encoder_prompts

        key = digest([tools, args.get("schema"), args.get("schema_name"), args.get("protocol")])
        if key not in self.contexts:
            cfg = self.compile_actions(tools, args)
            prompts = encoder_prompts(cfg, description=self.native.encoder_cfg.description)
            hidden = await asyncio.to_thread(self.native.encoder.encode_task_hidden, prompts)
            path = str(Path(self.native.directory.name) / (key + ".pt"))
            torch.save({"action_config": cfg, "hidden": hidden.hidden.cpu(), "mask": hidden.mask.cpu()}, path)
            heads = await self.engine.collective_rpc("dyad_inference_attest_head", args=(path,))
            if len(heads) != self.options.tensor_parallel_size:
                raise ValueError("Action head worker count differs from tensor parallel size")
            if any(h["projector_sha256"] != self.native.identity["projector_sha256"] for h in heads):
                raise ValueError("Projector changed after restoration")
            if any(h != heads[0] for h in heads):
                raise ValueError("Action head differs across tensor parallel workers")
            self.contexts[key] = cfg, path, heads[0]
        return self.contexts[key]

    async def generate(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        *,
        tools: list[dict[str, Any]] | None = None,
        args: ActionArguments | None = None,
    ) -> GenerationResult:
        """Return one complete vLLM response; no chat template or tool execution.

        Input IDs use the loaded tokenizer's base vocabulary. SamplingParams is
        cloned so concurrent callers can reuse it. Empty tools/args bypass action
        encoding and keep the restored policy's base-vocabulary distribution.
        Request context paths and text-only routing flags are owned by this engine.
        """
        from vllm.sampling_params import RequestOutputKind

        if self.engine is None or self._closed:
            raise RuntimeError("Start VLLMInference before generating")
        if (
            not isinstance(prompt_token_ids, list)
            or not prompt_token_ids
            or any(type(token) is not int or token < 0 for token in prompt_token_ids)
        ):
            raise ValueError("prompt_token_ids must be a nonempty list of nonnegative integers")
        if sampling_params.n != 1:
            raise ValueError("Inference requires n=1; submit independent requests for repeats")
        maximum = sampling_params.max_tokens
        if type(maximum) is not int or maximum < 1:
            raise ValueError("An explicit positive max_tokens is required")
        if len(prompt_token_ids) + maximum > self.options.context_length:
            raise ValueError(f"Prompt budget exceeded: {len(prompt_token_ids)} tokens; no truncation")
        tools = [] if tools is None else tools
        args = {} if args is None else args
        validate_actions(tools, args)
        params = sampling_params.clone()
        if any(key.startswith("dyad_") for key in (params.extra_args or {})):
            raise ValueError("Dyad routing is engine-owned; use tools and args to define actions")
        # This method returns a complete response even if the caller supplied DELTA.
        params.output_kind = RequestOutputKind.FINAL_ONLY
        cfg, head = None, None
        if self.native is not None:
            if tools or args.get("schema") or args.get("schema_name"):
                async with self.lock:
                    cfg, path, head = await self.context(tools, args)
                routing = {"dyad_action_context": path}
            else:
                routing = {"dyad_text_only": True}
            params.extra_args = {**(params.extra_args or {}), **routing}
        request_id = uuid4().hex
        final = None
        try:
            async for output in self.engine.generate({"prompt_token_ids": list(prompt_token_ids)}, params, request_id):
                final = output
        except BaseException:
            await self.engine.abort(request_id)
            raise
        if final is None or not final.finished or len(final.outputs) != 1:
            raise RuntimeError("Generation did not return exactly one complete output")
        output = final.outputs[0]
        if output.finish_reason == "abort" or output.stop_reason == "aborted":
            raise RuntimeError("Generation was aborted")
        ids = list(output.token_ids)
        evidence = {"action_interface": self.options.action_interface, "selections": []}
        if cfg is not None:
            content = getattr(final, "action_content", None)
            if not isinstance(content, dict) or len(content) != 1:
                raise ValueError("Dyad generation omitted its request-local action trace")
            trace = next(iter(content.values()))
            from agent_system.policies.dyad.actions.policy_replay import build_policy_trace, validate_policy_trace

            replay = build_policy_trace(trace, ids)
            validate_policy_trace(replay, ids)
            evidence["policy_trace"] = {
                "response_dyad": replay["response_dyad"],
                "seq_mask": replay["seq_mask"],
                "tool_mask": replay["tool_mask"],
                "dyad_allowed_action_ids": replay["allowed_action_ids"],
                "dyad_action_size": replay["action_size"],
            }
            if trace.get("action_config") != cfg:
                raise ValueError("Sampled action schema differs from the requested schema")
            names = {i: name for name, i in cfg["action_name_ids"].items()}
            evidence.update(
                action_content=trace,
                head=head,
                schema_sha256=digest(cfg),
                selections=[names[t] for t in trace["raw_token_ids"] if t in names],
            )
        return GenerationResult(request_id=request_id, output=final, dyad=evidence)
