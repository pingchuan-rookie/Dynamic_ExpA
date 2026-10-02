# >>> DYAD-BEGIN(vllm0.24): restore the dependency the decorators need
# `instrument` is 0.24's tracing-span decorator. Losing it does not crash anything; these methods
# simply stop producing spans -- a silent loss of functionality, exactly the class of failure the
# rebase exists to avoid.
from agent_system.policies.dyad.actions.task_context import dynamic_actions_enabled, action_capacity
from vllm.tracing import instrument
# <<< DYAD-END
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""

import gc
import os
from contextlib import AbstractContextManager, nullcontext
from types import NoneType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_pp_group,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
# DYAD-NOTE(vllm0.24): set_random_seed moved from vllm.model_executor to vllm.utils.torch_utils
from vllm.utils.torch_utils import set_random_seed
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm.platforms import current_platform
# DYAD-NOTE(vllm0.24): vllm.profiler.gpu_profiler was deleted; both wrappers moved to
# vllm.profiler.wrapper
from vllm.profiler.wrapper import CudaProfilerWrapper, TorchProfilerWrapper
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import MemorySnapshot, memory_profiling
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.worker_base import WorkerBase
from vllm.v1.worker.gpu_worker import Worker

# >>> DYAD-BEGIN(vllm0.24): dependencies introduced by syncing 0.24's verbatim methods
from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors
from vllm.utils.mem_utils import format_gib
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.v1.worker.utils import request_memory
# <<< DYAD-END

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig


class DyadWorker(Worker):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        # >>> DYAD-BEGIN(vllm0.24) DYAD-REBASED: taken from 0.24 verbatim; Dyad's only change is the
    # final else branch, which swaps GPUModelRunnerV1 for an DyadGPUModelRunner built from the
    # schema compiled per DYAD_ACTION_YAML.
    # The 0.12 baseline was missing all of the following upstream logic. None of it crashes; each
    # one just silently does less:
    #   assigned_physical_gpu_ids -- publishes the logical->physical GPU map (NIC affinity, P2P)
    #   logical_device_id_to_visible_device_id -- only correct once CUDA_VISIBLE_DEVICES is reordered
    #   init_workspace_manager -- the workspace for DBO's two microbatches
    #   request_memory()/MemorySnapshot(device=) -- replaces computing total*util by hand
    #   data_parallel_index -- replaces data_parallel_rank
    # DYAD-NOTE(vllm0.24): decorators restored. ast.get_source_segment's text for a FunctionDef does
    # not include the decorator lines, which is how the earlier sync tool dropped them wholesale.
    @instrument(span_name='Init device')
    def init_device(self):
        if self.device_config.device_type == "cuda":
            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            parallel_config = self.parallel_config
            if (
                parallel_config.distributed_executor_backend
                not in ("ray", "external_launcher")
                and parallel_config.data_parallel_backend != "ray"
                and parallel_config.nnodes_within_dp == 1
            ):
                # Use local DP rank if available, otherwise use global DP rank.
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = self.parallel_config.data_parallel_index

                tp_pp_world_size = (
                    self.parallel_config.pipeline_parallel_size
                    * self.parallel_config.tensor_parallel_size
                )

                # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
                self.local_rank += dp_local_rank * tp_pp_world_size

            # Publish the logical-to-physical mapping for topology queries
            # such as NIC affinity and P2P checks.
            assigned_physical_gpu_ids = parallel_config.assigned_physical_gpu_ids
            if assigned_physical_gpu_ids is not None:
                from vllm.platforms.interface import set_assigned_physical_gpu_ids

                set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)
                assert self.local_rank < len(assigned_physical_gpu_ids), (
                    f"local_rank {self.local_rank} is out of bounds for "
                    f"assigned_physical_gpu_ids {assigned_physical_gpu_ids}"
                )
                # NOTE(patch pr45026): local_world_size is derived from
                # parallel_config.nnodes, which is only set for the "mp"
                # multi-node backend. With the "ray"/"external_launcher"
                # backends nnodes stays 1, so local_world_size collapses to
                # the full world_size and this check wrongly fires on
                # cross-node deployments. assigned_physical_gpu_ids is already
                # per-node and the local_rank bound above fully validates the
                # mapping for these backends, so skip the check for them.
                if parallel_config.distributed_executor_backend not in (
                    "ray",
                    "external_launcher",
                ):
                    assert self.parallel_config.local_world_size <= len(
                        assigned_physical_gpu_ids
                    ), (
                        f"local_world_size ({self.parallel_config.local_world_size})"
                        " exceeds assigned_physical_gpu_ids count "
                        f"({len(assigned_physical_gpu_ids)})"
                    )
            else:
                assert self.local_rank < torch.accelerator.device_count(), (
                    f"DP adjusted local rank {self.local_rank} is out of "
                    f"bounds for {torch.accelerator.device_count()} devices."
                )

            visible_device_index = (
                current_platform.logical_device_id_to_visible_device_id(self.local_rank)
            )
            self.device = torch.device(f"cuda:{visible_device_index}")
            torch.accelerator.set_device_index(self.device)

            current_platform.check_if_supports_dtype(self.model_config.dtype)

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

            if self.use_v2_model_runner:
                logger.info_once("Using V2 Model Runner")

            # Set random seed.
            set_random_seed(self.model_config.seed)

            # Now take memory snapshot after NCCL is initialized
            gc.collect()
            torch.accelerator.empty_cache()

            # take current memory snapshot
            self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
            self.requested_memory = request_memory(init_snapshot, self.cache_config)
            logger.debug("worker init memory snapshot: %r", self.init_snapshot)
            logger.debug(
                "worker requested memory: %sGiB", format_gib(self.requested_memory)
            )
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        # Construct the model runner
        if self.use_v2_model_runner:
            # Dyad is not implemented on the v2 runner, and taking this branch does not fail -- it
            # produces a plain GPUModelRunner with no action_head, no admissible-set masking and no
            # action_content. Dyad silently degrades into ordinary sampling, and the first symptom is
            # an AttributeError from a completely unrelated place several minutes later:
            #   'GPUModelRunner' object has no attribute 'reinit_action_head_from_lm_head'
            #   (agent_system/policies/dyad/rollout/dyad_vllm_rollout.py, during the first weight sync)
            #
            # vLLM 0.24 chooses v2 per model (VllmConfig.use_v2_model_runner). Qwen2.5-3B gets v1 and
            # Qwen3-4B gets v2, which is why this only appeared when a second model was tried.
            #
            # Refuse instead of degrading. `VLLM_USE_V2_MODEL_RUNNER=0` forces v1 and is set by
            # experiments/shared/train_eval/<env>/dyad.sh; it also has to survive the trip into the Ray
            # actors, which is why main_dyad injects the VLLM_ prefix into runtime_env.env_vars.
            raise RuntimeError(
                "Dyad requires the v1 GPU model runner, but vLLM selected the v2 runner for this "
                "model. Dyad's engine fork (DyadGPUModelRunner) only extends v1, so continuing "
                "would run ordinary sampling with no action head and no admissible-set masking. "
                "Set VLLM_USE_V2_MODEL_RUNNER=0 (and make sure it reaches the Ray actors -- "
                "runtime_env.env_vars must carry the VLLM_ prefix)."
            )
        else:
            from .dyad_gpu_model_runner import DyadGPUModelRunner
            from agent_system.policies.dyad.actions.schema_config import (
                DEFAULT_ACTION_CONFIG_PATH, load_action_config_from_yaml, resolve_schema_path,
            )
            # Sampling and replay share the default; explicit saved schemas retain their protocol.
            yaml_name = os.environ.get("DYAD_ACTION_YAML")
            yaml_path = resolve_schema_path(yaml_name) if yaml_name else DEFAULT_ACTION_CONFIG_PATH
            if dynamic_actions_enabled():
                from agent_system.policies.dyad.actions.task_context import bootstrap_schema
                self.raw_action_config = bootstrap_schema(action_capacity())
            else:
                self.raw_action_config = load_action_config_from_yaml(yaml_path)
            # Pass the source yaml path through to the tool_parser (parse by schema instead of
            # relying on the global environment variable alone).
            if isinstance(self.raw_action_config, dict) and not dynamic_actions_enabled():
                self.raw_action_config["_parse_yaml_path"] = str(yaml_path)
            # Cross-env val: when DYAD_VAL_ACTION_YAML is given, training uses the schema above
            # (e.g. CodeGym) and validation uses this val schema (e.g. ALFWorld). The runner
            # rebuilds the val action head on the fly from the current base weights during the
            # validation phase. When unset, training is single-env and behaviour is unchanged.
            self.raw_val_action_config = None
            val_yaml_name = os.environ.get("DYAD_VAL_ACTION_YAML")
            if val_yaml_name:
                val_yaml_path = resolve_schema_path(val_yaml_name)
                self.raw_val_action_config = load_action_config_from_yaml(val_yaml_path)
                if isinstance(self.raw_val_action_config, dict):
                    self.raw_val_action_config["_parse_yaml_path"] = str(val_yaml_path)
            self.model_runner = DyadGPUModelRunner(
                self.vllm_config,
                self.device,
                self.raw_action_config,
                self.raw_val_action_config,
            )

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        # Sampling phase: pick a token from the logits / grammar constraints / etc.
        return self.model_runner.sample_tokens(grammar_output)

    @torch.inference_mode()
        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def execute_model(
            self, scheduler_output: "SchedulerOutput"
        ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
            # ensure any previous non-blocking PP sends are complete
            if self._pp_send_work:
                for handle in self._pp_send_work:
                    handle.wait()
                self._pp_send_work = []

            intermediate_tensors = None
            forward_pass = scheduler_output.total_num_scheduled_tokens > 0
            num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            all_gather_tensors = {}
            compilation_config = self.vllm_config.compilation_config
            parallel_config = self.vllm_config.parallel_config

            if (
                parallel_config.pipeline_parallel_size > 1
                and compilation_config.pass_config.enable_sp
                and forward_pass
            ):
                # currently only supported by V1 GPUModelRunner
                assert not self.use_v2_model_runner
                num_scheduled_tokens_np = np.array(
                    list(scheduler_output.num_scheduled_tokens.values()),
                    dtype=np.int32,
                )
                # TODO(lucas): This is pretty gross; ideally we should only ever call
                # `_determine_batch_execution_and_padding` once (will get called again
                # in `execute_model`) but this requires a larger refactor of PP.
                _, batch_desc, _, _, _ = (
                    self.model_runner._determine_batch_execution_and_padding(
                        num_tokens=num_scheduled_tokens,
                        num_reqs=len(num_scheduled_tokens_np),
                        num_scheduled_tokens_np=num_scheduled_tokens_np,
                        max_num_scheduled_tokens=num_scheduled_tokens_np.max(),
                        use_cascade_attn=False,  # TODO(lucas): Handle cascade attention
                    )
                )
                all_gather_tensors = {
                    "residual": not is_residual_scattered_for_sp(
                        self.vllm_config, batch_desc.num_tokens
                    )
                }

            if forward_pass and not get_pp_group().is_first_rank:
                tensor_dict, comm_handles, comm_postprocess = (
                    get_pp_group().irecv_tensor_dict(
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                )
                assert tensor_dict is not None
                intermediate_tensors = AsyncIntermediateTensors(
                    tensor_dict,
                    comm_handles=comm_handles,
                    comm_postprocess=comm_postprocess,
                )

            with self.annotate_profile(scheduler_output):
                output = self.model_runner.execute_model(
                    scheduler_output, intermediate_tensors
                )
                if (
                    self.use_v2_model_runner
                    and self.model_runner.is_pooling_model
                    and output is None
                ):
                    output = self.model_runner.pool()  # type: ignore
                if isinstance(
                    output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType
                ):
                    return output

            assert isinstance(output, IntermediateTensors)
            parallel_config = self.vllm_config.parallel_config
            assert (
                parallel_config.distributed_executor_backend != "external_launcher"
                and not get_pp_group().is_last_rank
            )

            # launch non-blocking send of intermediate tensors
            self._pp_send_work = get_pp_group().isend_tensor_dict(
                output.tensors,
                all_gather_group=get_tp_group(),
                all_gather_tensors=all_gather_tensors,
            )

            return None


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment of the worker."""

    parallel_config = vllm_config.parallel_config

    # Initialize batch invariance, which relates to execution paths/optimizations that are
    # independent of the batch size.
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    init_batch_invariance()

    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    # distributed init method:
    # - use the externally provided one if given
    # - otherwise fall back to env://
    init_method = distributed_init_method or "env://"

    init_distributed_environment(
        parallel_config.world_size, rank, init_method, local_rank, backend
    )

    # Initialize the model parallel groups:
    # - tensor parallel
    # - pipeline parallel
    # - prefill context parallel
    # - decode context parallel
    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
    )

    # Initialize the encoder connector. Note this comes before KV cache initialization:
    # under EPD disagg an encoder-only instance does not necessarily initialize KV caches,
    # so the ec connector must not depend on the KV cache already existing.
    ensure_ec_transfer_initialized(vllm_config)



import functools
import inspect
def _wrap_worker_methods(cls):
    exclude = {
        "_my_hook",   # do not wrap your own hook, that would recurse
    }

    for name, attr in list(cls.__dict__.items()):
        if name in exclude:
            continue
        if name.startswith("__") and name.endswith("__"):
            continue

        # Only plain instance methods are handled.
        if inspect.isfunction(attr):
            def make_wrapper(func, method_name):
                @functools.wraps(func)
                def wrapper(self, *args, **kwargs):
                    return func(self, *args, **kwargs)
                return wrapper

            setattr(cls, name, make_wrapper(attr, name))

    return cls


_wrap_worker_methods(DyadWorker)
