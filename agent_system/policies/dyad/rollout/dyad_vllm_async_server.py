# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import asyncio
import inspect
import json
import logging
import os
from concurrent.futures import Future
from pprint import pprint
from typing import Any, Callable, Optional
from uuid import uuid4

import cloudpickle as pickle
import numpy as np
import ray
import vllm.entrypoints.cli.serve
import zmq
from packaging import version
from ray.actor import ActorHandle
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_app,
    init_app_state,
)
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.utils import CoreEngineProcManager
from vllm.v1.executor.abstract import Executor

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass

# DYAD-NOTE(verl0.9): vllm_fp8_utils -> vllm_quant_utils, apply_vllm_fp8_patches -> apply_vllm_quant_patches.
from verl.utils.vllm.vllm_quant_utils import apply_vllm_quant_patches
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput

# DYAD-NOTE(verl0.9): get_free_port / is_valid_ipv6_address moved to verl.utils.net_utils;
# get_max_position_embeddings is still in workers.rollout.utils.
#
# run_unvicorn (0.7's typo) was renamed run_uvicorn and reimplemented. 0.7 did "get_free_port, then
# bind, retry 5 times on failure", which races: another process can take the port between getting it
# and binding it. 0.9 passes port=0 and lets the kernel allocate, then reads the real port back off
# the socket (_UvicornServerAutoPort). The race is gone, which is why the new signature has no
# max_retries. The return value (port, task) is unchanged, so call sites keep their shape.
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.workers.rollout.utils import (
    get_max_position_embeddings,
    run_uvicorn,
)
# Import straight from the source module (bypassing verl.workers.rollout.vllm_rollout's __init__),
# otherwise that __init__ would form a cycle while it is importing this package.
from agent_system.policies.dyad.rollout.dyad_vllm_rollout import ExpavLLMAsyncRollout
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_vllm_max_lora_rank,
)

from agent_system.policies.dyad.rollout.vllm.dyad_async_llm import DyadAsyncLLM, validate_dyad_rollout_turn
from agent_system.utils.diagnostics import log_event

_VLLM_VERSION = version.parse(vllm.__version__)

# DYAD-NOTE(vllm0.24): there used to be three version-gated branches here; all are now flattened.
#
# The vLLM extra in pyproject.toml sets the floor at vllm>=0.18.0, so the <=0.11.0 and
# ==0.12.0 branches are unreachable in ExpA_verl. Keeping them does real harm: their bodies import
# `from vllm.utils import FlexibleArgumentParser` and
# `from vllm.entrypoints.harmony_utils import get_encoding`, both dead on 0.24, so the static scan
# (experiments/tools/check_vllm_api.py) reports them as break points and the real problems drown in
# that noise.
#
# _VLLM_VERSION itself stays: later in this file there are places that genuinely need to
# distinguish 0.12+ behaviour.
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_tcp_uri
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


# class TokenOutput(BaseModel):
#     token_ids: list[int]
#     """response token ids"""
#     log_probs: Optional[list[float]] = None
#     """logprobs of response token ids"""
#     routed_experts: Optional[Any] = None
#     """routed experts of response token ids"""
#     stop_reason: Optional[str] = None
#     """stop reason: 'completed', 'aborted', or None for unknown"""


class DyadTokenOutput(TokenOutput):
    action_content: Optional[list[Any]] = None
    server_request_id: Optional[str] = None
    finish_reason: Optional[str] = None
    engine_stop_reason: Optional[str | int] = None

class DyadExternalZeroMQDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        dp_rank_local = self.vllm_config.parallel_config.data_parallel_rank_local
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size

        addresses = os.environ["VERL_VLLM_ZMQ_ADDRESSES"].split(",")
        addresses = addresses[dp_rank_local * tp_size : (dp_rank_local + 1) * tp_size]
        self.context = zmq.Context()
        self.sockets = []
        for address in addresses:
            socket = self.context.socket(zmq.REQ)
            if address.startswith("tcp://["):
                socket.setsockopt(zmq.IPV6, 1)
            socket.connect(address)
            self.sockets.append(socket)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    if _VLLM_VERSION >= version.parse("0.12.0"):

        def execute_model(
            self, scheduler_output: "SchedulerOutput", non_block: bool = False
        ) -> "ModelRunnerOutput | None | Future[ModelRunnerOutput | None]":
            output = self.collective_rpc("execute_model", args=(scheduler_output,))
            result = output[0]
            if non_block:
                f = Future()
                f.set_result(result)
                return f
            return result

        def sample_tokens(
            self, grammar_output: "GrammarOutput | None", non_block: bool = False
        ) -> "ModelRunnerOutput | None | Future[ModelRunnerOutput | None]":
            output = self.collective_rpc("sample_tokens", args=(grammar_output,))
            result = output[0]
            if non_block:
                f = Future()
                f.set_result(result)
                return f
            return result

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
        **kwargs_extra: Any,
    ) -> list[Any]:
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        message = pickle.dumps((sent_method, args, kwargs or {}))
        for socket in self.sockets:
            socket.send(message, zmq.DONTWAIT)

        outputs = []
        for socket in self.sockets:
            outputs.append(pickle.loads(socket.recv()))

        for output in outputs:
            if isinstance(output, Exception):
                raise output
        return outputs

    def check_health(self):
        return


def _rollout_of(worker):
    """Find the rollout instance inside a trainer worker.

    Supports both a colocated worker_dict and a standalone worker. Discover the
    role by its rollout attribute because role names are configured by the trainer.
    """
    rollout = getattr(worker, "rollout", None)
    if rollout is not None:
        return rollout
    for candidate in (getattr(worker, "worker_dict", None) or {}).values():
        rollout = getattr(candidate, "rollout", None)
        if rollout is not None:
            return rollout
    keys = sorted((getattr(worker, "worker_dict", None) or {}))
    raise AttributeError(
        f"{type(worker).__name__} 里找不到 rollout 实例（worker_dict 的键：{keys}）。"
        "Dyad 的 vLLM server 要在每个 trainer worker 内部操作它那份 rollout —— "
        "rollout 与训练引擎在同一个进程里，这是 action_content 能穿过 EngineCore 的前提。"
    )


# Use __ray_call__ to access rollout methods that are not registered on the worker handle.
def _worker_zeromq_address(worker):
    return _rollout_of(worker).get_zeromq_address()


def _worker_policy_version(worker):
    return _rollout_of(worker).global_steps


def _await_in_worker(coro):
    """Run a coroutine to completion in the worker process.

    Ray invokes __ray_call__ synchronously, so returning a coroutine would leave
    resume, release, or validation-mode changes unexecuted.
    """
    import asyncio as _asyncio

    if not _asyncio.iscoroutine(coro):
        return coro
    try:
        loop = _asyncio.get_running_loop()
    except RuntimeError:
        return _asyncio.run(coro)
    # Reuse the running event loop in async Ray actors.
    return loop.run_until_complete(coro)


def _worker_resume(worker):
    """Resume the rollout inference engine in its owning worker."""
    return _await_in_worker(_rollout_of(worker).resume(tags=["kv_cache", "weights"]))


def _worker_release(worker):
    """Release rollout weights and the KV cache in their owning worker."""
    return _await_in_worker(_rollout_of(worker).release())


def _worker_set_dyad_val_mode(on: bool):
    """Build a worker callback that forwards the requested validation mode."""

    def _call(worker):
        setter = getattr(_rollout_of(worker), "set_dyad_val_mode", None)
        return _await_in_worker(setter(on)) if setter is not None else None

    return _call


class ExpavLLMHttpServer:
    """vLLM http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
    ):
        """
        Args:
            config (RolloutConfig): full config.
            model_config (HFModelConfig): model config.
            rollout_mode (RolloutMode): rollout mode.
            replica_rank (int): replica rank, a replica may contain multiple nodes.
            node_rank (int): node rank.
            gpus_per_node (int): number of gpus per node.
            nnodes (int): number of nodes.
        """
        super().__init__()
        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        # The model's self-reported context is a *ceiling*, not the value to use: it is what
        # vLLM reserves KV cache for. This line used to assign it unconditionally, which threw
        # away `rollout.max_model_len` -- invisible on Qwen2.5-3B (32768 fits) and fatal on
        # Qwen3.5-4B, which reports 262144 and asks for 262 GiB of KV cache. Same policy as
        # verl's own async server (`_validate_configs`): fill it in when unset, refuse when it
        # exceeds what the model can address, otherwise leave the configured value alone.
        max_position_embeddings = get_max_position_embeddings(self.model_config.hf_config)
        if self.config.max_model_len is None:
            self.config.max_model_len = max_position_embeddings
        elif self.config.max_model_len > max_position_embeddings:
            raise ValueError(
                f"max_model_len ({self.config.max_model_len}) should be less than or equal to "
                f"max_position_embeddings ({max_position_embeddings})"
            )
        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
        if self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port, self._master_sock = get_free_port(self._server_address)
            self._dp_master_port, self._dp_master_sock = get_free_port(self._server_address)
            logger.info(
                f"ExpavLLMHttpServer, replica_rank: {self.replica_rank}, master address: {self._master_address}, "
                f"master port: {self._master_port}, data parallel master port: {self._dp_master_port}"
            )
        else:
            self._master_address = None
            self._master_port = None

    def get_master_address(self):
        """Get master address and port for data parallel."""
        return self._master_address, self._master_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port, "non-master node should provide master address and port"
            self._master_address = master_address
            self._master_port = master_port

        # 1. setup vllm serve cli args
        engine_kwargs = self.config.get("engine_kwargs", {}).get("vllm", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if self.config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": self.config.get("limit_images")}
        if self.config.cudagraph_capture_sizes:
            engine_kwargs["cuda_graph_sizes"] = self.config.cudagraph_capture_sizes

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        override_generation_config = dict(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=1.0,
            max_new_tokens=self.config.response_length,
        )
        logger.info(f"override_generation_config: {override_generation_config}")

        logger.info(f"enable_sleep_mode: {self.config.enable_sleep_mode}")
        if not self.config.enable_sleep_mode:
            from verl.utils.device import set_expandable_segments

            set_expandable_segments(True)

        quantization = self.config.quantization

        if quantization is not None:
            _SUPPORTED_QUANTIZATION = ["fp8", "torchao"]
            if quantization not in _SUPPORTED_QUANTIZATION:
                raise ValueError(f"Currently only support {_SUPPORTED_QUANTIZATION} quantization, got: {quantization}")

            if quantization == "fp8":
                FP8_BLOCK_QUANT_KWARGS = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                }
                fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
                # Apply vllm fp8 patches
                # Will remove the patch after vllm support on-the-fly quant for rollout natively.
                apply_vllm_quant_patches()

        hf_overrides = {}
        if quantization is not None and self.config.quantization_config_file is not None:
            hf_overrides["quantization_config_file"] = self.config.quantization_config_file

        if quantization == "fp8":
            hf_overrides["quantization_config"] = fp8_block_quant_kwargs


        args = {
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "skip_tokenizer_init": False,
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enable_sleep_mode": self.config.enable_sleep_mode,
            "logprobs_mode": self.config.logprobs_mode,
            "disable_custom_all_reduce": True,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "seed": self.config.get("seed", 0),
            "override_generation_config": json.dumps(override_generation_config),
            "quantization": quantization,
            "hf_overrides": hf_overrides,
            "scheduling_policy": self.config.scheduling_policy,
            **engine_kwargs,
         }

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

        if self.config.expert_parallel_size > 1:
            assert self.gpus_per_node % self.config.tensor_model_parallel_size == 0, (
                "gpus_per_node should be divisible by tensor_model_parallel_size"
            )
            data_parallel_size_local = self.gpus_per_node // self.config.tensor_model_parallel_size
            assert len(self.workers) == data_parallel_size_local * self.config.tensor_model_parallel_size, (
                f"num workers ({len(self.workers)}) should be equal to dp_size_local "
            )
            f"({data_parallel_size_local}) * tp_size ({self.config.tensor_model_parallel_size})"

            args.update(
                {
                    "enable_expert_parallel": self.config.expert_parallel_size > 1,
                    "data_parallel_size": self.config.data_parallel_size,
                    "data_parallel_size_local": data_parallel_size_local,
                    "data_parallel_start_rank": self.node_rank * data_parallel_size_local,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._master_port,
                }
            )

        # update lora-related args
        if self.model_config.lora_rank > 0:
            args.update(
                {
                    "enable_lora": True,
                    "max_loras": 1,
                    "max_lora_rank": get_vllm_max_lora_rank(self.model_config.lora_rank),
                }
            )

        if self.config.enable_rollout_routing_replay:
            args.update({"enable_return_routed_experts": True})

        server_args = ["serve", self.model_config.local_path]
        for k, v in args.items():
            if isinstance(v, bool):
                if v:
                    server_args.append(f"--{k}")
            elif v is not None:
                server_args.append(f"--{k}")
                # Use json.dumps for dict to ensure valid JSON format
                server_args.append(json.dumps(v) if isinstance(v, dict) else str(v))

        if self.replica_rank == 0:
            pprint(server_args)

        CMD_MODULES = [vllm.entrypoints.cli.serve]
        parser = FlexibleArgumentParser(description="vLLM CLI")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        server_args = parser.parse_args(args=server_args)
        server_args.model = server_args.model_tag
        if server_args.subparser in cmds:
            cmds[server_args.subparser].validate(server_args)

        # 2. setup distributed executor backend
        distributed_executor_backend = DyadExternalZeroMQDistributedExecutor if len(self.workers) > 0 else None
        server_args.distributed_executor_backend = distributed_executor_backend

        # Resolve rollout inside each worker via __ray_call__; trainer role names are configurable.
        zmq_addresses = await asyncio.gather(
            *[worker.__ray_call__.remote(_worker_zeromq_address) for worker in self.workers]
        )
        logger.info(
            f"replica_rank={self.replica_rank}, node_rank={self.node_rank}, nnodes={self.nnodes}, "
            f"get worker zmq addresses: {zmq_addresses}"
        )
        os.environ["VERL_VLLM_ZMQ_ADDRESSES"] = ",".join(zmq_addresses)

        # 3. launch server
        if self.node_rank == 0:
            await self.run_server(server_args)
        else:
            await self.run_headless(server_args)

    async def run_server(self, args: argparse.Namespace):
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        fn_args = set(dict(inspect.signature(DyadAsyncLLM.from_vllm_config).parameters).keys())
        kwargs = {}
        if "enable_log_requests" in fn_args:
            kwargs["enable_log_requests"] = engine_args.enable_log_requests
        if "disable_log_stats" in fn_args:
            kwargs["disable_log_stats"] = engine_args.disable_log_stats

        engine_client = DyadAsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)

        # Don't keep the dummy data in memory
        await engine_client.reset_mm_cache()

        app = build_app(args)
        if _VLLM_VERSION > version.parse("0.11.0"):
            await init_app_state(engine_client, app.state, args)
        else:
            await init_app_state(engine_client, vllm_config, app.state, args)
        if self.replica_rank == 0 and self.node_rank == 0:
            logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

        self.engine = engine_client
        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        # Create the EngineConfig.
        engine_args = vllm.AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context, headless=True)

        parallel_config = vllm_config.parallel_config
        local_engine_count = parallel_config.data_parallel_size_local

        host = parallel_config.data_parallel_master_ip
        port = engine_args.data_parallel_rpc_port  # add to config too
        handshake_address = get_tcp_uri(host, port)

        # Create the engines.
        self.engine_manager = CoreEngineProcManager(
            target_fn=EngineCoreProc.run_engine_core,
            local_engine_count=local_engine_count,
            start_index=vllm_config.parallel_config.data_parallel_rank,
            local_start_index=0,
            vllm_config=vllm_config,
            local_client=False,
            handshake_address=handshake_address,
            executor_class=Executor.get_class(vllm_config),
            log_stats=not engine_args.disable_log_stats,
        )

    async def _sampled_policy_version(self):
        # Read installed weights, not the dataset's prompt submission step.
        versions = await asyncio.gather(*[
            worker.__ray_call__.remote(_worker_policy_version) for worker in self.workers
        ])
        if not versions or any(type(version) is not int or version < 0 for version in versions):
            raise ValueError("Dyad sampling requires an installed policy weight version on every worker")
        if len(set(versions)) != 1:
            raise ValueError(f"Dyad rollout workers have inconsistent policy weight versions: {versions}")
        return versions[0]

    async def generate(
        self,
        prompt_ids: list[int],                 # prompt token ids already encoded by the tokenizer/chat_template
        sampling_params: dict[str, Any],       # sampling parameters (temperature/top_p/max_tokens/logprobs, etc.)
        request_id: str,                       # used by vLLM to identify one request (for cancellation/routing/caching)
        image_data: Optional[list[Any]] = None,# multimodal: image input (in the format vLLM/VLM expects)
        video_data: Optional[list[Any]] = None,# multimodal: video input
        priority: int = 0,                     # vLLM queue priority (larger usually scheduled earlier, engine-dependent)
    ) -> DyadTokenOutput:
        """Generate sequence with token-in-token-out.
        Takes token ids in, returns the generated token ids (plus optional logprob / routed experts / stop reason).
        """
        sampling_params = dict(sampling_params)
        dyad_rollout_turn, dyad_max_turns = validate_dyad_rollout_turn(
            sampling_params.pop("_dyad_rollout_turn", None),
            sampling_params.pop("_dyad_max_turns", None),
        )
        if dyad_rollout_turn is not None:
            log_event(
                "dyad_vllm_server",
                "rollout_turn_validated",
                request_id=request_id,
                rollout_turn=dyad_rollout_turn,
                max_turns=dyad_max_turns,
            )
        # -----------------------------
        # 1) Compute the hard upper bound on "how many more tokens can be generated" (safety valve)
        # -----------------------------
        max_possible_tokens = self.config.max_model_len - len(prompt_ids)

        # If the prompt alone already exceeds max_model_len, fail fast (otherwise vLLM would error or truncate)
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        # -----------------------------
        # 2) Decide max_tokens for this call (read sampling_params first, then fall back to the default policy)
        # -----------------------------
        # This "pops" keys: note that sampling_params is mutated in place (callers reusing this dict must be careful)
        if "max_tokens" in sampling_params:
            # OpenAI-style parameter name
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params:
            # accept the sglang/HF-style parameter name too
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            # Default policy (framework-specific):
            # the goal is to keep the total "prompt + response" length close to a configured target:
            #   config.response_length + config.prompt_length
            # but the actual prompt may not be config.prompt_length (it varies with multi-turn/tools/multimodal),
            # so - len(prompt_ids) corrects for that.
            #
            # For example: you want the total length to be ~(prompt_length + response_length),
            # but the actual prompt_ids is longer/shorter, so max_tokens is adjusted dynamically.
            max_tokens = self.config.response_length + self.config.prompt_length - len(prompt_ids)

        # -----------------------------
        # 3) Clamp max_tokens into the legal range [0, max_possible_tokens]
        # -----------------------------
        # - lower bound 0: a negative value means "generate nothing", and is rewritten to 0 here
        # - upper bound max_possible_tokens: prevents overrunning the model context
        max_tokens = max(0, min(max_tokens, max_possible_tokens))

        # Belt-and-braces assertion (after the clamp above this always holds in theory)
        assert max_tokens <= max_possible_tokens, (
            f"max_tokens {max_tokens} exceeds available context space {max_possible_tokens}"
        )

        # -----------------------------
        # 4) Handle logprobs / repetition_penalty / SamplingParams
        # -----------------------------
        # The style here is a bit "switch-like":
        # - if sampling_params has logprobs=True (or any truthy value), logprobs is set to 0
        #   (in vLLM logprobs usually means "return the top-k logprobs"; 0 is commonly used to mean
        #   "return the logprob of the chosen token" or the minimal configuration)
        # - otherwise it is set to None (do not return logprobs, saving overhead)
        #
        # Note: this line also pops sampling_params["logprobs"] (if present)
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None


        # If the caller did not pass repetition_penalty, fall back to a default (1.0 when unset)
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))

        # Convert the dict into vLLM's SamplingParams object (the type the vLLM engine expects)
        # max_tokens is passed explicitly here; the other sampling parameters (temperature/top_p/stop, etc.)
        # are injected via **sampling_params
        from vllm.sampling_params import RequestOutputKind


        # Change this knob here directly if a different output kind is needed
        OUTPUT_KIND = RequestOutputKind.CUMULATIVE
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            output_kind= OUTPUT_KIND,
            **sampling_params
            )

        # -----------------------------
        # 5) Qwen2.5-VL "image token dedup/fixup"
        # -----------------------------
        # The chat_template/processor of some VLMs inserts special image tokens,
        # which in some cases can be duplicated, corrupting the context or wasting length, hence this dedup.
        # (This is model/processor specific patch-up logic.)
        prompt_ids = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)

        # -----------------------------
        # 6) Assemble the multimodal data (vLLM's multi_modal_data format)
        # -----------------------------
        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data

        # vLLM's prompt wrapper: token ids + multimodal data
        prompt = TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data)

        # -----------------------------
        # 7) (optional) LoRA attachment: if LoRA is enabled and the engine already loaded it, enable it for this request
        # -----------------------------
        lora_request = None
        if self.model_config.lora_rank > 0:
            # list_loras() returns the loras currently loaded in the vLLM engine (usually lora_int_id or names)
            # The await here shows it is an async RPC / async method
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME,
                    lora_int_id=VLLM_LORA_INT_ID,
                    lora_path=VLLM_LORA_PATH,
                )

        # -----------------------------
        # 8) Kick off generation: engine.generate returns an async generator (streaming)
        # -----------------------------

        policy_version = await self._sampled_policy_version()
        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,  # None means no LoRA
            priority=priority,
        )

        # -----------------------------
        # 9) Consume the async stream: here we choose to "keep only the last output" (final result)
        # -----------------------------
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            # vLLM streaming keeps yielding partial results
            # Each iteration overwrites, so the last one (containing the complete token_ids) survives
            final_res = output

        # The generator should yield at least once in theory, otherwise this would be None
        assert final_res is not None
        if await self._sampled_policy_version() != policy_version:
            raise ValueError("Dyad policy weights changed during a single generation")
        version_fields = {
            "global_steps": policy_version,
            "min_global_steps": policy_version,
            "max_global_steps": policy_version,
        }
        # -----------------------------
        # 10) Extract the generated token ids (only the first candidate: outputs[0])
        # -----------------------------
        # vLLM may support n>1 (multiple candidates); here we always take the 1st candidate sequence
        token_ids = final_res.outputs[0].token_ids
        action_content_by_request = getattr(final_res, "action_content", None)
        action_content = [action_content_by_request] if action_content_by_request else None

        # -----------------------------
        # 11) (optional) extract log_probs
        # -----------------------------
        log_probs = None
        if sampling_params.logprobs is not None:
            # final_res.outputs[0].logprobs is usually "one dict/struct per position", holding several token->logprob entries
            # The code here assumes that for position i, logprobs can be indexed by token_ids[i] to get that token's logprob
            # >>> DYAD-BEGIN(dyad): take the logprob by position, not by token-id lookup
            # Dyad's router rewrites the sampled **action id** into the **written token** that
            # actually lands in the sequence (see the _parse_ids write-back in
            # DyadGPUModelRunner.sample_tokens). vLLM's logprobs dict is keyed by the **originally
            # sampled** token, so looking up the rewritten token_ids[i] is a guaranteed KeyError --
            # measured as `KeyError: 40`.
            #
            # The right reading is "the logprob of whatever was actually sampled at this
            # position", which is exactly the dict's only entry (when logprobs=0) or its largest:
            # verl sets sampling_params["logprobs"] to 0, and vLLM then returns just the chosen
            # token per position. Taking that single value avoids the id lookup and is also the
            # correct semantics -- the log-probability of what the policy actually chose at this
            # step, which for Dyad is the action, not the characters of its action surface form.
            #
            # The id lookup stays as the preferred path: at non-Dyad positions the token was not
            # rewritten, so when the lookup succeeds it is used and behaviour matches upstream
            # exactly.
            def _pick_logprob(lp_at_pos, tok_id):
                entry = lp_at_pos.get(tok_id)
                if entry is not None:
                    return entry.logprob
                if len(lp_at_pos) == 1:
                    return next(iter(lp_at_pos.values())).logprob
                # Several candidates and the id is absent: take the largest, i.e. the sampled one
                return max(e.logprob for e in lp_at_pos.values())

            log_probs = [
                _pick_logprob(logprobs, token_ids[i])
                for i, logprobs in enumerate(final_res.outputs[0].logprobs)
            ]
            # <<< DYAD-END

        # -----------------------------
        # 12) (optional) MoE routing replay: obtain the routed experts information
        # -----------------------------
        routed_experts = None
        if self.config.enable_rollout_routing_replay:
            # This is usually the record of which experts an MoE model picked per layer/per token
            routed_experts = final_res.outputs[0].routed_experts

        # -----------------------------
        # 13) stop_reason: map vLLM's finish_reason onto this framework's stop_reason
        # -----------------------------
        finish_reason = final_res.outputs[0].finish_reason
        if finish_reason == "abort":
            stop_reason = "aborted"         # cancelled/interrupted
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"       # normal stop or length cap (both normalized to completed)
        else:
            stop_reason = finish_reason     # reserved for more reasons in the future

        # vLLM's detokenizer may omit a stop token from text, but retains its ID
        # in CompletionOutput.token_ids. Preserve that exact sequence, including
        # alternative EOS IDs and router-forced turn endings. Appending a model
        # config EOS here invents an unrecorded decision and breaks strict replay.
        token_ids = list(token_ids)

        # -----------------------------
        # 14) Pack and return
        # -----------------------------
        log_event(
            "dyad_vllm_server",
            "generate_result",
            request_id=request_id,
            rollout_turn=dyad_rollout_turn,
            max_turns=dyad_max_turns,
            prompt_length=len(prompt_ids),
            token_ids=token_ids,
            log_probs=log_probs,
            action_content=action_content,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
            **version_fields,
        )
        return DyadTokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=routed_experts,
            stop_reason=stop_reason,
            action_content=action_content,
            server_request_id=request_id,
            finish_reason=finish_reason,
            engine_stop_reason=getattr(final_res.outputs[0], "stop_reason", None),
            extra_fields=version_fields,
        )


    async def wake_up(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            # DYAD-NOTE(verl0.9): Resume each trainer worker's inference engine.
            # Dyad hybrid execution uses these engines through ZeroMQ; waking only the server leaves workers asleep.
            await asyncio.gather(*[worker.__ray_call__.remote(_worker_resume)
                                   for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            if self.node_rank == 0:
                await self.engine.wake_up(tags=["kv_cache", "weights"])
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            # Pair each worker resume with a worker release.
            if self.node_rank == 0:
                await self.engine.reset_prefix_cache()
            await asyncio.gather(*[worker.__ray_call__.remote(_worker_release)
                                   for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            if self.node_rank == 0:
                await self.engine.reset_prefix_cache()
                await self.engine.sleep(level=1)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def clear_kv_cache(self):
        if self.node_rank == 0:
            await self.engine.reset_prefix_cache()

    async def set_dyad_val_mode(self, on: bool):
        # Cross-env validation: forward the val-mode switch to each worker (and on to the Dyad model runner).
        # Under HYBRID/COLOCATED the worker actor owns the rollout/model_runner; STANDALONE is skipped.
        if self.rollout_mode in (RolloutMode.HYBRID, RolloutMode.COLOCATED):
            await asyncio.gather(*[worker.__ray_call__.remote(_worker_set_dyad_val_mode(on))
                                   for worker in self.workers])
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip set_dyad_val_mode in standalone mode")

    async def wait_for_requests_to_drain(self):
        await self.engine.wait_for_requests_to_drain()

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort all ongoing generation requests.

        Returns:
            dict[str, Any]: Dictionary containing:
                - aborted_count: Number of requests aborted
                - request_ids: List of aborted request IDs
        """
        try:
            # Take an atomic snapshot to avoid race conditions with the vLLM engine thread
            request_states_snapshot = list(self.engine.output_processor.request_states.items())
            request_ids = [req_id for req_id, _ in request_states_snapshot]

            if not request_ids:
                return {"aborted_count": 0, "request_ids": []}

            # For each request, create an abort output and put it to its queue
            # This allows the generator to receive the aborted result
            from vllm.v1.engine import FinishReason

            for _, req_state in request_states_snapshot:
                request_output = req_state.make_request_output(
                    [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
                )
                req_state.queue.put(request_output)

            # Abort requests in the output processor and engine core
            self.engine.output_processor.abort_requests(request_ids)
            await self.engine.engine_core.abort_requests_async(request_ids)

            # Try to reset prefix cache to ensure clean state
            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info("Prefix cache reset after abort")

            logger.info(f"Aborted {len(request_ids)} requests: {request_ids}")
            return {"aborted_count": len(request_ids), "request_ids": request_ids}

        except Exception as e:
            logger.error(f"Error aborting requests: {e}")
            return {"aborted_count": 0, "request_ids": [], "error": str(e)}

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort a specific generation request.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Dictionary containing abort result.
        """
        try:
            request_states = self.engine.output_processor.request_states
            req_state = request_states.get(request_id)

            if req_state is None:
                return {"aborted": False, "error": f"Request {request_id} not found"}

            # Create abort output and put it to the queue
            from vllm.v1.engine import FinishReason

            request_output = req_state.make_request_output(
                [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
            )
            req_state.queue.put(request_output)

            # Abort in output processor and engine core
            self.engine.output_processor.abort_requests([request_id])
            await self.engine.engine_core.abort_requests_async([request_id])

            # Try to reset prefix cache to ensure clean state
            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info(f"Prefix cache reset after abort request {request_id}")

            logger.info(f"Aborted request: {request_id}")
            return {"aborted": True, "request_id": request_id}

        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")
            return {"aborted": False, "request_id": request_id, "error": str(e)}


_rollout_worker_actor_cls = ray.remote(ExpavLLMAsyncRollout)


class ExpavLLMReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(ExpavLLMHttpServer)

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        worker_dict_cls = RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
        )
        return worker_dict_cls

    async def set_dyad_val_mode(self, on: bool):
        """Cross-env validation: forward the val-mode switch to each http server."""
        await asyncio.gather(*[server.set_dyad_val_mode.remote(on) for server in self.servers])

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # get node_id of all workers
        worker_node_ids = await asyncio.gather(
            *[
                worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                for worker in self.workers
            ]
        )

        # For non-data parallel case, there's only one server whether it's single or multi nodes.
        nnodes, gpus_per_node = self.nnodes, self.gpus_per_node
        if self.config.data_parallel_size == 1:
            nnodes = 1
            gpus_per_node = self.world_size

        # create server actor in each node with node affinity
        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_node : (node_rank + 1) * gpus_per_node]
            node_id = worker_node_ids[node_rank * gpus_per_node]
            name = (
                f"dyadvllm_server_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"epxavllm_server_reward_{self.replica_rank}_{node_rank}"
            )
            name = name + f"_{uuid4().hex[:8]}"
            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_node,
                nnodes=nnodes,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

    async def sleep(self):
        """Sleep each rollout server."""
        # Drain DP engines for safe sleep.
        await self.servers[0].wait_for_requests_to_drain.remote()
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def abort_all_requests(self) -> dict[str, Any]:
        """Abort all ongoing generation requests across all servers.

        Returns:
            dict[str, Any]: Combined abort results from all servers.
        """
        results = await asyncio.gather(*[server.abort_all_requests.remote() for server in self.servers])

        total_aborted = sum(r.get("aborted_count", 0) for r in results)
        all_request_ids = []
        for r in results:
            all_request_ids.extend(r.get("request_ids", []))

        return {
            "aborted_count": total_aborted,
            "request_ids": all_request_ids,
            "server_results": results,
        }

    async def abort_request(self, request_id: str) -> dict[str, Any]:
        """Abort a specific request. Tries all servers since we don't know which one has it.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Abort result.
        """
        # TODO(petersh6): we should only abort on the server that has the request.
        results = await asyncio.gather(*[server.abort_request.remote(request_id) for server in self.servers])

        for r in results:
            if r.get("aborted", False):
                return r

        return {"aborted": False, "request_id": request_id, "error": "Request not found on any server"}


def _resolve_prompt_debug_tokenizer(server):
    processor = getattr(server.model_config, "processor", None)
    engine = getattr(server, "engine", None)
    candidates = (
        getattr(server, "tokenizer", None),
        getattr(engine, "tokenizer", None),
        getattr(server.model_config, "tokenizer", None),
        getattr(processor, "tokenizer", None),
    )
    return next((tokenizer for tokenizer in candidates if tokenizer is not None), None)


def _decode_prompt_ids(tokenizer, prompt_ids: list[int]) -> str:
    try:
        return tokenizer.decode(
            prompt_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(prompt_ids, skip_special_tokens=False)


async def _print_decoded_prompt_before_generate(
    server,
    prompt_ids: list[int],
    request_id: str,
) -> None:
    tokenizer = _resolve_prompt_debug_tokenizer(server)
    if tokenizer is None:
        print(
            f"[vllm_async_server] Cannot decode prompt before generate: "
            f"tokenizer not found, request_id={request_id}.",
            flush=True,
        )
        return

    try:
        loop = asyncio.get_running_loop()
        prompt_text = await loop.run_in_executor(
            None,
            _decode_prompt_ids,
            tokenizer,
            prompt_ids,
        )
    except Exception as exc:
        print(
            f"[vllm_async_server] Failed to decode prompt before generate: "
            f"{exc!r}, request_id={request_id}.",
            flush=True,
        )
        return

    print(
        "\n"
        + "=" * 80
        + "\n"
        + "[vllm_async_server] ACTUAL PROMPT TEXT BEFORE engine.generate\n"
        + f"[vllm_async_server] request_id: {request_id}\n"
        + f"[vllm_async_server] prompt token length: {len(prompt_ids)}\n"
        + "=" * 80
        + "\n"
        + prompt_text
        + "\n"
        + "=" * 80
        + "\n",
        flush=True,
    )


def _qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    """Deduplicate consecutive image tokens in prompt_ids for Qwen2.5-VL, since vLLM will replicate the
    <|image_pad|> and <|video_pad|> token by image_data.

    For example,
    ```
    <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
    =>
    <|vision_start|><|image_pad|><|vision_end|>
    ```
    """
    if processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        prompt_ids = np.array(prompt_ids)

        # Create a mask where True indicates elements to keep
        mask = np.ones(len(prompt_ids), dtype=bool)

        # Find where the array equals the value
        is_value = (prompt_ids == processor.image_token_id) | (prompt_ids == processor.video_token_id)

        # Find consecutive duplicates by checking if previous element is also the value
        mask[1:] &= ~(is_value[1:] & is_value[:-1])

        return prompt_ids[mask].tolist()
    else:
        return prompt_ids
