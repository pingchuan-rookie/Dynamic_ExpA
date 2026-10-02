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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import getpass
import logging
import os
from dataclasses import asdict
from types import MethodType
from typing import Any, Generator

import cloudpickle as pickle
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from torch.distributed.device_mesh import DeviceMesh
from vllm.config import LoRAConfig

from verl.utils.ray_utils import get_event_loop

# DYAD-NOTE(vllm0.24): the whole vllm.worker package is gone; only vllm.v1.worker remains, so the
# old try/except fallback now always takes the except branch. Importing the v1 path directly stops
# the static scan reporting vllm.worker.worker_base as a break point forever.
# Upstream removal commit: 6a113d9aed8221a9c234535958e70e34ab6cac5b
from vllm.v1.worker.worker_base import WorkerWrapperBase

from packaging import version as vs

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL, get_version
from verl.utils.device import is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge

# DYAD-NOTE(verl0.9): the quantization helpers moved from vllm_fp8_utils to vllm_quant_utils, and
# apply_vllm_fp8_patches was renamed apply_vllm_quant_patches (its scope widened from fp8 to
# quantization in general).
from verl.utils.vllm.vllm_quant_utils import apply_vllm_quant_patches, is_fp8_model, load_quanted_weights
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout

# DYAD-NOTE(verl0.9): get_free_port / is_valid_ipv6_address moved from workers.rollout.utils to the
# general utils.net_utils. workers.rollout.utils now keeps only rollout-specific things.
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_vllm_max_lora_rank,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

VLLM_ASCEND_REQUIRED_ENV_VARS = {"VLLM_ALL2ALL_BACKEND": "flashinfer_all2allv", "VLLM_ASCEND_ENABLE_NZ": "0"}

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


if is_version_ge(pkg="vllm", minver="0.7.3"):
    VLLMHijack.hijack()


def _check_vllm_version_for_sleep_level():
    # https://github.com/vllm-project/vllm/issues/25171
    minver = "0.11.0"
    current_version = get_version("vllm")
    if not current_version:
        logger.warning("Could not determine vLLM version, assuming an older version for sleep_level configuration.")
        return False
    return vs.parse(current_version) >= vs.parse(minver)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)



class ExpavLLMAsyncRollout(BaseRollout):
    """ExpavLLMAsyncRollout is a thin wrapper of WorkerWrapperBase, which is engine in single worker process."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = self.model_config.tokenizer
        self.inference_engine: WorkerWrapperBase = None
        self.global_steps = None
        self.address = self._init_zeromq()
        self.lora_config = (
            {"max_loras": 1, "max_lora_rank": get_vllm_max_lora_rank(self.model_config.lora_rank)}
            if self.model_config.lora_rank > 0
            else {}
        )

        if config.layered_summon or (config.expert_parallel_size > 1 and not _check_vllm_version_for_sleep_level()):
            logger.warning("Setting the sleep level to 1 may cause a memory overflow.")
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip = ray.util.get_node_ip_address().strip("[]")
                port, sock = get_free_port(ip)
                if is_valid_ipv6_address(ip):
                    address = f"tcp://[{ip}]:{port}"
                    self.socket.setsockopt(zmq.IPV6, 1)
                else:
                    address = f"tcp://{ip}:{port}"
            self.socket.bind(address)

        loop = get_event_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    async def _loop_forever(self):
        while True:
            try:
                message = await self.socket.recv()
                method, args, kwargs = pickle.loads(message)
                result = await self._execute_method(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"ExpavLLMAsyncRollout _loop_forever error: {e}")
                await self.socket.send(pickle.dumps(e))
                break

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        # TODO: For ascend NPU, when the corresponding vllm-ascend version is upgraded to v0.13.0,
        # please remove the VLLM_ASCEND_REQUIRED_ENV_VARS variable replacement action.
        # This is only a fix for vllm version < v0.13.0.
        if is_npu_available:
            for k in VLLM_ASCEND_REQUIRED_ENV_VARS:
                if k not in os.environ:
                    os.environ[k] = VLLM_ASCEND_REQUIRED_ENV_VARS[k]

        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        device_name = "NPU" if is_npu_available else "GPU"
        all_kwargs[0]["local_rank"] = (
            0
            if not ray_noset_visible_devices()
            else int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        )
        self.vllm_config = all_kwargs[0]["vllm_config"]
        if self.lora_config:
            lora_dtype = getattr(torch, self.config.dtype)
            self.vllm_config.lora_config = LoRAConfig(lora_dtype=lora_dtype, **self.lora_config)
        if self.config.quantization is not None:
            _SUPPORTED_QUANTIZATION = ["fp8", "torchao"]
            if self.config.quantization not in _SUPPORTED_QUANTIZATION:
                raise ValueError(
                    f"Currently only support {_SUPPORTED_QUANTIZATION} quantization, got: {self.config.quantization}"
                )

            if self.config.quantization == "fp8":
                # Apply vllm fp8 patches
                # Will remove the patch after vllm support on-the-fly quant for rollout natively.
                apply_vllm_quant_patches()

        # >>> DYAD-BEGIN(vllm0.24): WorkerWrapperBase's constructor signature changed
        # 0.12: WorkerWrapperBase(vllm_config=...) -- the config is given at construction.
        # 0.24: WorkerWrapperBase(rpc_rank=0, global_rank=None) -- no vllm_config; the config is
        #       taken by init_worker(all_kwargs) from all_kwargs[rpc_rank]["vllm_config"].
        #       Still passing vllm_config gives
        #       TypeError: WorkerWrapperBase.__init__() got an unexpected keyword argument 'vllm_config'.
        #
        # Each Ray actor here hosts exactly one worker (one TP rank), so rpc_rank is always 0,
        # matching vllm's own uniproc_executor (only the multiproc path uses local_rank).
        # global_rank takes the global rank computed just above, for logging and distributed init.
        # global_rank must be the **index within the engine's TP group**, not the training-side
        # global RANK. 0.24's WorkerWrapperBase.initialize_from_config uses it to index
        # kv_cache_configs[self.global_rank], and that list's length is **this engine's** world size
        # (tensor_model_parallel_size), not the job's. Passing the training-side RANK means that
        # with TP=1 and dp=2, rank=1 indexes a list of length 1 -- an outright
        # IndexError: list index out of range, raised inside the worker process, with the driver
        # only showing "Dyad Engine core initialization failed".
        _tp_size = max(1, int(self.config.tensor_model_parallel_size))
        self.inference_engine = WorkerWrapperBase(
            rpc_rank=0,
            global_rank=all_kwargs[0]["rank"] % _tp_size,
        )
        self.inference_engine.init_worker(all_kwargs)
        # <<< DYAD-END

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        else:
            # >>> DYAD-BEGIN(vllm0.24): execute_method was removed; use run_method
            # 0.12's WorkerWrapperBase.execute_method was `run_method(self, method, args, kwargs)`
            # plus a layer of error logging. 0.24 deleted that wrapper, but run_method itself is
            # still in vllm.v1.serial_utils. Method resolution is unchanged: the wrapper's own
            # methods first, then __getattr__ falls through to self.worker.
            #
            # Keep the error-logging layer: these calls run over distributed RPC, and one rank
            # raising while the others wait is a deadlock (vllm#3455). Without the log all anyone
            # sees is a hang, with no indication why.
            from vllm.v1.serial_utils import run_method

            try:
                return run_method(self.inference_engine, method, args, kwargs)
            except Exception:
                logger.exception(
                    "Error executing method %r on ExpavLLMAsyncRollout. "
                    "This might cause deadlock in distributed execution.",
                    method,
                )
                raise
            # <<< DYAD-END

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine:
            self.inference_engine.wake_up(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        global_steps = kwargs.get("global_steps")
        if global_steps is not None and (type(global_steps) is not int or global_steps < 0):
            raise ValueError("Dyad weight version must be a nonnegative integer")
        # A failed or partial load must not leave the old version advertised.
        self.global_steps = None
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            # In async mode, make sure the old lora is removed before adding the new one
            self.inference_engine.worker.remove_lora(VLLM_LORA_INT_ID)
            weights = dict(weights)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=asdict(peft_config),
                lora_tensors=weights,
            )
            self.inference_engine.worker.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model_runner = self.inference_engine.worker.model_runner
            model = model_runner.model
            patch_vllm_moe_model_weight_loader(model)

            weights = list(weights)
            # Three-way split. Every tensor lands in exactly one bucket, and a tensor that silently
            # matched none of them would just never be synced -- so the partition is asserted below
            # rather than assumed.
            #
            # action_head.* is a materialized sampling head, not the trainable projector.
            # Rebuild it from the synchronized projector and encoder-cache buffers instead
            # of loading a potentially stale copy. dyad_residual_head.* is the retained
            # checkpoint/transport namespace for the current direct head.
            base_weights = []
            adapter_weights = []
            for name, tensor in weights:
                if "dyad_residual_head." in name:
                    adapter_weights.append((name, tensor))
                elif "action_head." in name:
                    continue          # re-derived, never loaded
                else:
                    base_weights.append((name, tensor))
            head_weights = len(weights) - len(base_weights) - len(adapter_weights)
            assert len(base_weights) + len(adapter_weights) + head_weights == len(weights)

            # Add the FP8 related logic here as sharding manager has been deprecated.
            # Check if FP8 quantization is enabled and apply appropriate weight loading
            if is_fp8_model(model_runner.vllm_config):
                logger.info(f"FP8 model detected (async): {model_runner.vllm_config.quant_config}")
                # Convert bf16 weights to fp8 format before loading
                loaded_params = load_quanted_weights(base_weights, model_runner)
                logger.info(f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}")
            else:
                logger.info("Loading standard weights (non-FP8, async)")
                model.load_weights(base_weights)

            # Projector parameters and encoder-cache buffers must arrive before the rebuild.
            # Reversing this order samples with a stale head and breaks rollout/log-prob pairing.
            if adapter_weights:
                loaded = model_runner.load_dyad_residual_head(adapter_weights)
                logger.info(f"vLLM loaded Dyad encoder projector, tensors: {loaded}")
            elif getattr(model_runner, "_dyad_residual_head", None) is not None:
                raise RuntimeError(
                    "this engine has an Dyad encoder projector but the policy LM's weight stream carried "
                    "no dyad_residual_head.* tensors. The engine would then sample with its "
                    "initial projector forever while the trainer's moves -- silently, since the "
                    "shapes still match. Check that the projector is registered inside actor_module "
                    "before the FSDP wrap (agent_system/policies/dyad/training/dyad_workers.py)."
                )

            # Materialize only after all direct-head inputs have reached the current version.
            model_runner.reinit_action_head_from_lm_head()
            logger.info("vLLM rebuilt Dyad action head from synced encoder/projector state")

        # Publish only after LM, projector and materialized action head agree.
        self.global_steps = global_steps

    async def set_dyad_val_mode(self, on: bool):
        """Cross-env validation hook: switch the rollout Dyad action head/config.

        When training and validation use different environments (e.g. train on
        CodeGym, validate on ALFWorld), the trained action head does not transfer.
        ``on=True`` makes the model runner switch to the configured val schema and
        rebuild the val action head from the synchronized encoder/projector state;
        ``on=False`` restores the training schema/head. No-op when no val
        schema was configured on the runner.
        """
        model_runner = self.inference_engine.worker.model_runner
        if on:
            enter = getattr(model_runner, "enter_val_mode", None)
            if callable(enter):
                enter()
        else:
            exit_fn = getattr(model_runner, "exit_val_mode", None)
            if callable(exit_fn):
                exit_fn()

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode.

        Note: ExpavLLMAsyncRollout uses async server mode and does not support synchronous
        generation. Since SPMD mode was retired (PR #4411), the generation workflow
        should use the async server interface instead.

        Raises:
            NotImplementedError: Always raised as sync generation is not supported.
        """
        raise NotImplementedError(
            "ExpavLLMAsyncRollout does not support synchronous generate_sequences(). "
            "The vLLM SPMD mode was retired in PR #4411. For batch generation, "
            "please use the async server interface via vLLMReplica and AsyncLLMServerManager, "
            "or use HFRollout for synchronous generation. "
            "See https://github.com/volcengine/verl/issues/4682 for more details."
        )

    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address