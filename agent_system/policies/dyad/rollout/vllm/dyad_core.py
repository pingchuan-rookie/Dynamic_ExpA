# >>> DYAD-BEGIN(vllm0.24): restore the dependency the decorators need
# `instrument` is 0.24's tracing-span decorator. Losing it does not crash anything; these methods
# simply stop producing spans -- a silent loss of functionality, exactly the class of failure the
# rebase exists to avoid.
from vllm.tracing import instrument
# <<< DYAD-END
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import queue
import signal
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, TypeVar, cast

import msgspec
import zmq

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import (
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
    UtilityResult,
    # DYAD-NOTE(vllm0.24): a type introduced by syncing 0.24's verbatim methods
    PauseMode,
)
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    # DYAD-NOTE(vllm0.24): get_device_indices was deleted in favour of
    # get_physical_gpu_ids_for_local_dp_rank -- both the return value and the side effect changed;
    # see the call site below.
    get_physical_gpu_ids_for_local_dp_rank,
)
from vllm.v1.executor import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.version import __version__ as VLLM_VERSION
from .dyad_outputs import DyadEngineCoreOutputs


from multiprocessing.queues import Queue  # DYAD-NOTE(vllm0.24): introduced by syncing 0.24 verbatim

# >>> DYAD-BEGIN(vllm0.24): dependencies introduced by syncing 0.24's verbatim methods
from vllm.v1.engine import EngineCoreReadyResponse
from vllm.distributed import cleanup_dist_env_and_memory
import vllm.envs as envs
import gc
from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
# <<< DYAD-END

# >>> DYAD-BEGIN(vllm0.24): dependencies introduced by syncing 0.24's verbatim methods
from vllm.v1.engine import EEPNotificationType
from vllm.v1.engine.core import EngineShutdownState
from vllm.v1.engine.tensor_ipc import TensorIpcReceiver
# Added when run_engine_core / _handle_client_request were rebased onto 0.24:
#   SignalCallback           0.24 factored signal handling into a callback type (v1/engine/utils)
#   maybe_init_worker_tracer 0.24 initialises the tracer inside the engine subprocess
#   numa_utils               NUMA core pinning, new in 0.24
from vllm.v1.engine.utils import SignalCallback
from vllm.tracing import maybe_init_worker_tracer
from vllm.utils import numa_utils
# <<< DYAD-END

# >>> DYAD-BEGIN(vllm0.24): dependencies introduced by syncing 0.24's verbatim methods
from vllm.v1.engine import EEP_NOTIFICATION_CALL_ID
from vllm.v1.engine import EngineCoreOutput
from vllm.v1.engine import FinishReason
from vllm.v1.utils import IterationDetails
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.utils import compute_iteration_details
from collections import defaultdict
from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind
from functools import partial
# <<< DYAD-END

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar("_R")  # Return type for collective_rpc


def _is_single_action_payload(action_content: dict) -> bool:
    """
    Decide whether action_content is a single payload rather than {request_id: payload}.

    You can add further discriminating conditions based on your own payload fields.
    """
    return (
        isinstance(action_content, dict)
        and (
            "raw_token_ids" in action_content
            or "tool_call" in action_content
            or "action_name" in action_content
        )
    )


def _normalize_action_content_by_request_id(
    outputs: list,
    action_content: dict | None,
) -> dict | None:
    """
    Normalize action_content into {request_id: payload}.

    Two input shapes are supported:
    1. New format:
       {
           "0_xxx": payload0,
           "1_xxx": payload1,
       }

    2. Legacy format:
       payload

       The legacy format is only allowed when outputs has length 1, otherwise there is
       no way to tell which request_id it belongs to.
    """
    if not action_content:
        return None

    request_ids = [output.request_id for output in outputs]
    request_id_set = set(request_ids)

    if _is_single_action_payload(action_content):
        if len(outputs) != 1:
            raise ValueError(
                "Single action_content payload cannot be mapped to multiple "
                f"EngineCoreOutput items. outputs request_ids={request_ids}. "
                "Use {request_id: payload} format instead."
            )

        return {
            outputs[0].request_id: action_content,
        }

    if not isinstance(action_content, dict):
        raise TypeError(
            f"action_content must be dict or None, got {type(action_content)}"
        )

    # New format: {request_id: payload}
    matched = {
        request_id: payload
        for request_id, payload in action_content.items()
        if request_id in request_id_set
    }

    # action_content is non-empty but nothing matched the current EngineCoreOutputs,
    # which usually means the keys are wrong.
    if action_content and not matched:
        raise ValueError(
            "action_content does not match any EngineCoreOutput.request_id. "
            f"outputs request_ids={request_ids}, "
            f"action_content keys={list(action_content.keys())}"
        )

    return matched or None


def _action_content_slices(
    engine_core_outputs: dict,
    action_content: dict,
) -> dict:
    """Split one step's action_content into the slice each output group actually owns.

    `update_from_output` returns one `EngineCoreOutputs` **per client**, while a step's
    action_content covers whichever requests made an Dyad decision -- and those need not be
    spread across every client. Handing the whole payload to each group therefore fails as soon
    as two clients are served in one step: the group that does not own the deciding request sees
    a key it cannot place and raises.

    Orphan detection stays, it just moves up a level: a payload matching no output *anywhere* is
    still an error, because replay could not rebuild that request's mask and Dyad would degrade
    into ordinary sampling without raising anything (AGENTS.md section 1).

    Returns `{engine_idx: slice or None}`; None means "this group owns none of it", which is a
    normal step, not a failure.
    """
    all_outputs = [out for group in engine_core_outputs.values() for out in group.outputs]
    by_request = _normalize_action_content_by_request_id(all_outputs, action_content) or {}
    slices = {}
    for engine_idx, core_outputs in engine_core_outputs.items():
        owned = {out.request_id for out in core_outputs.outputs}
        this_slice = {rid: payload for rid, payload in by_request.items() if rid in owned}
        slices[engine_idx] = this_slice or None
    return slices


def to_dyad_engine_core_outputs(
    obj: "EngineCoreOutputs | DyadEngineCoreOutputs",
    action_content: dict | None = None,
) -> "DyadEngineCoreOutputs":
    """
    Convert EngineCoreOutputs into DyadEngineCoreOutputs.

    Note:
    action_content must be aligned by request_id, so its final shape should be:

        {
            request_id: payload
        }

    Broadcasting a single payload to all outputs is not allowed when rollout.n > 1.
    """

    if not isinstance(obj, (EngineCoreOutputs, DyadEngineCoreOutputs)):
        raise TypeError(
            f"expected EngineCoreOutputs or DyadEngineCoreOutputs, got {type(obj)}"
        )

    if action_content is None:
        action_content = getattr(obj, "action_content", None)

    action_content_by_req = _normalize_action_content_by_request_id(
        obj.outputs,
        action_content,
    )

    if isinstance(obj, DyadEngineCoreOutputs):
        obj.action_content = action_content_by_req
        return obj

    return DyadEngineCoreOutputs(
        engine_index=obj.engine_index,
        outputs=obj.outputs,
        scheduler_stats=obj.scheduler_stats,
        timestamp=obj.timestamp,
        utility_output=obj.utility_output,
        finished_requests=obj.finished_requests,
        wave_complete=obj.wave_complete,
        start_wave=obj.start_wave,
        action_content=action_content_by_req,
    )

class DyadEngineCore:
    """Inner loop of vLLM's Engine."""
        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def __init__(
            self,
            vllm_config: VllmConfig,
            executor_class: type[Executor],
            log_stats: bool,
            executor_fail_callback: Callable | None = None,
            include_finished_set: bool = False,
        ):
            # plugins need to be loaded at the engine/scheduler level too
            from vllm.plugins import load_general_plugins

            load_general_plugins()

            self.vllm_config = vllm_config
            if not vllm_config.parallel_config.data_parallel_rank_local:
                logger.info(
                    "Initializing a V1 LLM engine (v%s) with config: %s",
                    VLLM_VERSION,
                    vllm_config,
                )

            self.log_stats = log_stats

            # Setup Model.
            self.model_executor = executor_class(vllm_config)
            if executor_fail_callback is not None:
                self.model_executor.register_failure_callback(executor_fail_callback)

            self.available_gpu_memory_for_kv_cache = -1

            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                self._eep_scale_up_before_kv_init()

            # Setup KV Caches and update CacheConfig after profiling.
            kv_cache_config = self._initialize_kv_caches(vllm_config)
            self.structured_output_manager = StructuredOutputManager(vllm_config)

            # Setup scheduler.
            Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

            if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
                # Encoder models without KV cache don't support
                # chunked prefill. But do SSM models?
                if vllm_config.scheduler_config.enable_chunked_prefill:
                    logger.warning("Disabling chunked prefill for model without KVCache")
                    vllm_config.scheduler_config.enable_chunked_prefill = False

            scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
                kv_cache_config, vllm_config
            )

            self.scheduler: SchedulerInterface = Scheduler(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                structured_output_manager=self.structured_output_manager,
                include_finished_set=include_finished_set,
                log_stats=self.log_stats,
                block_size=scheduler_block_size,
                hash_block_size=hash_block_size,
            )
            self.use_spec_decode = vllm_config.speculative_config is not None
            self.check_for_draft_tokens = (
                self.use_spec_decode or vllm_config.model_config.is_diffusion
            )
            if self.scheduler.connector is not None:  # type: ignore
                self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

            mm_registry = MULTIMODAL_REGISTRY
            self.mm_receiver_cache = mm_registry.engine_receiver_cache_from_config(
                vllm_config
            )

            # If a KV connector is initialized for scheduler, we want to collect
            # handshake metadata from all workers so the connector in the scheduler
            # will have the full context
            kv_connector = self.scheduler.get_kv_connector()
            if kv_connector is not None:
                # Collect and store KV connector xfer metadata from workers
                # (after KV cache registration)
                xfer_handshake_metadata = (
                    self.model_executor.get_kv_connector_handshake_metadata()
                )

                if xfer_handshake_metadata:
                    # xfer_handshake_metadata is list of dicts from workers
                    # Each dict already has structure {(pp_rank, tp_rank): metadata}
                    # Merge all worker dicts into a single dict
                    content: dict[tuple[int, int], Any] = {}
                    for worker_dict in xfer_handshake_metadata:
                        if worker_dict is not None:
                            content.update(worker_dict)
                    kv_connector.set_xfer_handshake_metadata_pp_aware(content)

            # Setup batch queue for pipeline parallelism.
            # Batch queue for scheduled batches. This enables us to asynchronously
            # schedule and execute batches, and is required by pipeline parallelism
            # to eliminate pipeline bubbles.
            self.batch_queue_size = vllm_config.max_concurrent_batches
            self.batch_queue: (
                deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
            ) = None
            if self.batch_queue_size > 1:
                logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
                self.batch_queue = deque(maxlen=self.batch_queue_size)

            self.is_ec_consumer = (
                vllm_config.ec_transfer_config is None
                or vllm_config.ec_transfer_config.is_ec_consumer
            )
            self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

            self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
            if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
                caching_hash_fn = get_hash_fn_by_name(
                    vllm_config.cache_config.prefix_caching_hash_algo
                )
                init_none_hash(caching_hash_fn)

                self.request_block_hasher = get_request_block_hasher(
                    hash_block_size, caching_hash_fn
                )

            self.step_fn = (
                self.step if self.batch_queue is None else self.step_with_batch_queue
            )
            self.async_scheduling = vllm_config.scheduler_config.async_scheduling

            self.aborts_queue = queue.Queue[list[str]]()

            self._idle_state_callbacks: list[Callable] = []

            # Mark the startup heap as static so that it's ignored by GC.
            # Reduces pause times of oldest generation collections.
            freeze_gc_heap()
            # If enable, attach GC debugger after static variable freeze.
            maybe_attach_gc_debug_callback()
            # Enable environment variable cache (e.g. assume no more
            # environment variable overrides after this point)
            enable_envs_cache()


        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    # DYAD-NOTE(vllm0.24): decorators restored exactly as in 0.24. ast.get_source_segment's text for
    # a FunctionDef starts at `def` and **excludes the decorator lines**, which is how the earlier sync
    # tool dropped them wholesale. Their order matters too: @staticmethod must sit above @instrument,
    # or instrument receives a staticmethod object and reading __code__ raises AttributeError.
    @instrument(span_name='Prepare model')
    def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig:
            start = time.time()

            # register all kvcache specs in enginecore process.
            register_all_kvcache_specs(vllm_config)

            # Get all kv cache needed by the model
            kv_cache_specs = self.model_executor.get_kv_cache_specs()

            # Some layers (e.g. Prefix LM attention) run non-causally and tag their
            # KV cache spec with ``non_causal=True``. The specs are collected here in
            # the engine-core process (the same process that builds the scheduler),
            # so this is the multiproc-safe place to translate that layer-level
            # signal into a scheduling policy: chunked prefill and prefix caching
            # both assume causal attention and would corrupt non-causal prefill.
            if any(
                getattr(spec, "non_causal", False)
                for worker_specs in kv_cache_specs
                for spec in worker_specs.values()
            ):
                if vllm_config.scheduler_config.enable_chunked_prefill:
                    logger.info(
                        "Disabling chunked prefill: model has non-causal attention layers."
                    )
                    vllm_config.scheduler_config.enable_chunked_prefill = False
                if vllm_config.cache_config.enable_prefix_caching:
                    logger.info(
                        "Disabling prefix caching: model has non-causal attention layers."
                    )
                    vllm_config.cache_config.enable_prefix_caching = False

            has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
            if has_kv_cache:
                if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                    # NOTE(yongji): should already be set
                    # during _eep_scale_up_before_kv_init
                    assert self.available_gpu_memory_for_kv_cache > 0
                    available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                        kv_cache_specs
                    )
                else:
                    # Profiles the peak memory usage of the model to determine how
                    # much memory can be allocated for kv cache.
                    available_gpu_memory = self.model_executor.determine_available_memory()
                    self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
            else:
                # Attention free models don't need memory for kv cache
                available_gpu_memory = [0] * len(kv_cache_specs)

            assert len(kv_cache_specs) == len(available_gpu_memory)

            # Track max_model_len before KV cache config to detect auto-fit changes
            max_model_len_before = vllm_config.model_config.max_model_len

            kv_cache_configs = get_kv_cache_configs(
                vllm_config, kv_cache_specs, available_gpu_memory
            )

            # If auto-fit reduced max_model_len, sync the new value to workers.
            # This is needed because workers were spawned before memory profiling
            # and have the original (larger) max_model_len cached.
            max_model_len_after = vllm_config.model_config.max_model_len
            if max_model_len_after != max_model_len_before:
                self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

            scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
            vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
            kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
            if kv_cache_groups:
                vllm_config.cache_config.block_size = min(
                    g.kv_cache_spec.block_size for g in kv_cache_groups
                )
                num_tokens, max_concurrency = get_kv_cache_capacity(
                    vllm_config, scheduler_kv_cache_config
                )
                vllm_config.cache_config.kv_cache_size_tokens = num_tokens
                vllm_config.cache_config.kv_cache_max_concurrency = max_concurrency

            vllm_config.validate_block_size()

            # Initialize kv cache and warmup the execution
            self.model_executor.initialize_from_config(kv_cache_configs)

            elapsed = time.time() - start
            compile_time = vllm_config.compilation_config.compilation_time
            encoder_compile_time = vllm_config.compilation_config.encoder_compilation_time
            if encoder_compile_time > 0:
                logger.info_once(
                    "init engine (profile, create kv cache, warmup model) took "
                    "%.2f s (compilation: %.2f s — language_model: %.2f s, "
                    "encoder: %.2f s)",
                    elapsed,
                    compile_time + encoder_compile_time,
                    compile_time,
                    encoder_compile_time,
                )
            elif compile_time > 0:
                logger.info_once(
                    "init engine (profile, create kv cache, warmup model) took "
                    "%.2f s (compilation: %.2f s)",
                    elapsed,
                    compile_time,
                )
            else:
                logger.info_once(
                    "init engine (profile, create kv cache, warmup model) took %.2f s",
                    elapsed,
                )
            return scheduler_kv_cache_config

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_executor.supported_tasks

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def add_request(self, request: Request, request_wave: int = 0):
            """Add request to the scheduler.

            `request_wave`: indicate which wave of requests this is expected to
            belong to in DP case
            """
            # Validate the request_id type.
            if not isinstance(request.request_id, str):
                raise TypeError(
                    f"request_id must be a string, got {type(request.request_id)}"
                )

            if pooling_params := request.pooling_params:
                supported_pooling_tasks = [
                    task for task in self.get_supported_tasks() if task in POOLING_TASKS
                ]

                if pooling_params.task not in supported_pooling_tasks:
                    raise ValueError(
                        f"Unsupported task: {pooling_params.task!r} "
                        f"Supported tasks: {supported_pooling_tasks}"
                    )

            if request.kv_transfer_params is not None and (
                not self.scheduler.get_kv_connector()
            ):
                logger.warning(
                    "Got kv_transfer_params, but no KVConnector found. "
                    "Disabling KVTransfer for this request."
                )

            self.scheduler.add_request(request)
            if request.abort_immediately:
                # Immediately abort so the connector's request_finished hook runs
                # to free any pre-admission KV-transfer resources.
                self.abort_requests([request.request_id])

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """Execute the model and log detailed info on failure."""
        try:
            yield
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(
                self.vllm_config, scheduler_output, self.scheduler.make_stats()
            )
            raise err

    def _log_err_callback(self, scheduler_output: SchedulerOutput):
        """Log error details of a future that's not expected to return a result."""

        def callback(f, sched_output=scheduler_output):
            with self.log_error_detail(sched_output):
                result = f.result()
                assert result is None

        return callback

    # >>> DYAD-BEGIN(vllm0.24) DYAD-REBASED: taken from 0.24 verbatim; Dyad only adds the block at the
    # end, which attaches action_content to EngineCoreOutputs.
    # Upstream 0.24 logic the 0.12 copy was missing: schedule()'s _should_throttle_prefills() argument
    # (prefill throttling; without it a long prompt starves decode), the log_iteration_details context,
    # and aborts that arrive during model execution being consumed before outputs are processed
    # (_process_aborts_queue).
    def step(self) -> tuple[dict[int, DyadEngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # >>> DYAD-BEGIN(dyad): attach the action selected this step to the output
        # model_output.action_content is filled in by DyadGPUModelRunner.sample_tokens.
        # Here upstream's EngineCoreOutputs is swapped for the Dyad version that carries
        # action_content, which then travels get_output_async -> _run_output_handler ->
        # process_outputs to the training side.
        # This step is the **only exit** of Dyad's data path: dropping it raises nothing, it just
        # leaves action_content None throughout, so replay cannot rebuild the mask and Dyad silently
        # degrades into ordinary sampling.
        action_content = getattr(model_output, "action_content", None)
        if action_content:
            slices = _action_content_slices(engine_core_outputs, action_content)
            for engine_idx, core_outputs in list(engine_core_outputs.items()):
                engine_core_outputs[engine_idx] = to_dyad_engine_core_outputs(
                    core_outputs,
                    slices[engine_idx],
                )
        # <<< DYAD-END

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def post_step(self, model_executed: bool) -> None:
            # When using async scheduling we can't get draft token ids in advance,
            # so we update draft token ids in the worker process and don't
            # need to update draft token ids here.
            if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
                draft_token_ids = self.model_executor.take_draft_token_ids()
                if draft_token_ids is not None:
                    self.scheduler.update_draft_token_ids(draft_token_ids)

    # DYAD-REBASED(vllm0.24): this method is vLLM 0.24 verbatim; Dyad's only change is swapping
    # upstream class names / log strings for the Dyad* ones (referencing an upstream class would route
    # around Dyad's changes without raising). No logic of its own.
    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, DyadEngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """

        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
            if self.is_ec_consumer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output, exec_future))
                if len(batch_queue) < self.batch_queue_size and (
                    model_executed or self.scheduler.has_requests()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, model_executed

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                # None from sample_tokens() implies that the original execute_model()
                # call failed - raise that exception.
                exec_model_fut.result()
                raise RuntimeError("unexpected error")

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # When draft tokens are used with structured output, validate them
            # before computing the grammar bitmask for the deferred request.
            if self.check_for_draft_tokens:
                draft_token_ids = self.model_executor.take_draft_token_ids()
                if draft_token_ids is not None:
                    # Update the draft token ids in the scheduler output to
                    # filter out the invalid spec tokens, which will be padded
                    # with -1 and skipped by the grammar bitmask computation.
                    self.scheduler.update_draft_token_ids_in_output(
                        draft_token_ids, deferred_scheduler_output
                    )
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

        return engine_core_outputs, model_executed

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def shutdown(self):
            logger.debug_once("[shutdown] EngineCore: tearing down local resources")
            self.structured_output_manager.clear_backend()
            if self.model_executor:
                self.model_executor.shutdown()
            if self.scheduler:
                self.scheduler.shutdown()

            # Undo the gc.freeze() from __init__ so that the objects allocated
            # during engine startup (model weights, KV caches, etc.) become
            # visible to the garbage collector again. Without this, deleting
            # the engine in-process (e.g. unit tests) leaks GPU memory.
            gc.unfreeze()
            # Tear down distributed state initialized in this EngineCore process
            # before it exits and release cached memory.
            cleanup_dist_env_and_memory()
            logger.debug_once("[shutdown] EngineCore: local resource teardown complete")

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
            self.model_executor.profile(is_start, profile_prefix)

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the multi-modal cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # The cache either exists in EngineCore or WorkerWrapperBase
        if self.mm_receiver_cache is not None:
            self.mm_receiver_cache.clear_cache()

        self.model_executor.reset_mm_cache()

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def reset_prefix_cache(
            self, reset_running_requests: bool = False, reset_connector: bool = False
        ) -> bool:
            return self.scheduler.reset_prefix_cache(
                reset_running_requests, reset_connector
            )

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None | Future:
            """Put the engine to sleep at the specified level.

            Args:
                level: Sleep level.
                    - Level 0: Pause scheduling only. Requests are still accepted
                               but not processed. No GPU memory changes.
                    - Level 1: Offload model weights to CPU, discard KV cache.
                    - Level 2: Discard all GPU memory.
                mode: Pause mode - how to deal with any existing requests, see
                    documentation of pause_scheduler method.
            """

            # Pause scheduler before sleeping.
            clear_prefix_cache = level >= 1
            pause_future = self.pause_scheduler(mode=mode, clear_cache=clear_prefix_cache)
            if level < 1:
                return pause_future

            # Level 1+: Delegate to executor for GPU memory management
            model_executor = self.model_executor
            if pause_future is None:
                model_executor.sleep(level)
                return None

            future = Future[Any]()

            def pause_complete(f: Future):
                try:
                    f.result()  # propagate any exception
                    future.set_result(model_executor.sleep(level))
                except Exception as e:
                    future.set_exception(e)

            logger.info("Waiting for in-flight requests to complete before sleeping...")
            pause_future.add_done_callback(pause_complete)
            return future

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def wake_up(self, tags: list[str] | None = None):
            """Wake up the engine from sleep.

            Args:
                tags: Tags to wake up. Use ["scheduling"] for level 0 wake up.
            """
            if tags is not None and "scheduling" in tags:
                # Remove "scheduling" from tags if there are other tags to process.
                tags = [t for t in tags if t != "scheduling"]

            if tags is None or tags:
                self.model_executor.wake_up(tags)

            # Partial wakes intentionally keep the remaining allocations asleep.
            # Resume scheduling only once all executor memory is resident again.
            if not self.model_executor.is_sleeping:
                self.resume_scheduler()

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def is_sleeping(self) -> bool:
            """Check if engine is sleeping at any level."""
            return self.is_scheduler_paused() or self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_sharded_state(
            path=path, pattern=pattern, max_size=max_size
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def get_kv_cache_group_metadata(self) -> list[dict[str, int | str | None]]:
            """Return msgspec-serializable metadata for scheduler KV cache groups."""
            kv_cache_config = getattr(self.scheduler, "kv_cache_config", None)
            if kv_cache_config is None:
                return []

            metadata: list[dict[str, int | str | None]] = []
            for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
                spec = group.kv_cache_spec
                metadata.append(
                    {
                        "group_idx": group_idx,
                        "kind": get_kv_cache_spec_kind(spec).value,
                        "block_size": spec.block_size,
                        "sliding_window": getattr(spec, "sliding_window", None),
                    }
                )
            return metadata

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    # DYAD-NOTE(vllm0.24): decorators restored exactly as in 0.24. ast.get_source_segment's text for
    # a FunctionDef starts at `def` and **excludes the decorator lines**, which is how the earlier sync
    # tool dropped them wholesale. Their order matters too: @staticmethod must sit above @instrument,
    # or instrument receives a staticmethod object and reading __code__ raises AttributeError.
    @contextmanager
    def log_iteration_details(self, scheduler_output: SchedulerOutput | None):
            if not self.vllm_config.observability_config.enable_logging_iteration_details:
                yield
                return
            # 0-token step: let the dummy_batch wrapper log it (avoids double-log).
            if scheduler_output and scheduler_output.total_num_scheduled_tokens == 0:
                yield
                return
            self._iteration_index = getattr(self, "_iteration_index", 0)
            # scheduler_output=None marks a DP dummy iteration.
            if scheduler_output is None:
                iteration_details = IterationDetails(0, 0, 0, 0)
                is_dummy = True
            else:
                iteration_details = compute_iteration_details(scheduler_output)
                is_dummy = False
            before = time.monotonic()
            yield
            logger.info(
                "".join(
                    [
                        "Iteration(",
                        str(self._iteration_index),
                        "): ",
                        str(iteration_details.num_ctx_requests),
                        " context requests, ",
                        str(iteration_details.num_ctx_tokens),
                        " context tokens, ",
                        str(iteration_details.num_generation_requests),
                        " generation requests, ",
                        str(iteration_details.num_generation_tokens),
                        " generation tokens, iteration elapsed time: ",
                        format((time.monotonic() - before) * 1000, ".2f"),
                        " ms",
                        " (dummy)" if is_dummy else "",
                    ]
                )
            )
            self._iteration_index += 1

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _should_throttle_prefills(self) -> bool:
            """Whether to defer new prefills this step (DP prefill balancing).
            Overridden by the DP engine core; never throttles otherwise."""
            return False

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _process_aborts_queue(self):
            if not self.aborts_queue.empty():
                request_ids = []
                while not self.aborts_queue.empty():
                    ids = self.aborts_queue.get_nowait()
                    # Should be a list here, but also handle string just in case.
                    request_ids.extend((ids,) if isinstance(ids, str) else ids)
                # More efficient to abort all as a single batch.
                self.abort_requests(request_ids)

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def reset_encoder_cache(self) -> None:
            """Reset the encoder cache to invalidate all cached encoder outputs.

            This should be called when model weights are updated to ensure
            stale vision embeddings computed with old weights are not reused.
            Clears both the scheduler's cache manager and the GPU model runner's cache.
            """
            # NOTE: Since this is mainly for debugging, we don't attempt to
            # re-sync the internal caches (P0 sender, P1 receiver)
            if self.scheduler.has_unfinished_requests():
                logger.warning(
                    "Resetting the encoder cache when requests are "
                    "in progress may lead to desynced internal caches."
                )

            # Reset the scheduler's encoder cache manager (logical state)
            self.scheduler.reset_encoder_cache()
            # Reset the GPU model runner's encoder cache (physical storage)
            self.model_executor.reset_encoder_cache()

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _reset_caches(
            self,
            reset_running_requests: bool = True,
            reset_connector: bool = True,
        ) -> None:
            # reset_connector=True so external connectors clear alongside
            # local caches, matching the pause_generation(clear_cache=True)
            # contract. No-op when no connector is configured.
            self.reset_prefix_cache(
                reset_running_requests=reset_running_requests,
                reset_connector=reset_connector,
            )
            self.reset_mm_cache()
            self.reset_encoder_cache()

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def pause_scheduler(
            self, mode: PauseMode = "abort", clear_cache: bool = True
        ) -> Future | None:
            """Pause generation; behavior depends on mode.

            All pause modes queue new adds -- "abort" and "keep" skip step();
            "wait" allows step() so in-flight requests can drain.

            - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
              outputs to be sent (when running with output_queue), optionally
              clear caches, then complete the returned Future.
            - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
              optionally clear caches, then complete the returned Future.
            - ``keep``: Set PAUSED_ALL; return a Future that completes when the
              output queue is empty.
            """
            if mode not in ("keep", "abort", "wait"):
                raise ValueError(f"Invalid pause mode: {mode}")
            if mode == "wait":
                raise ValueError("'wait' mode can't be used in inproc-engine mode")

            if mode == "abort":
                self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)

            pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
            self.scheduler.set_pause_state(pause_state)
            if clear_cache:
                self._reset_caches()

            return None

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def resume_scheduler(self) -> None:
            """Resume the scheduler and flush any requests queued while paused."""
            self.scheduler.set_pause_state(PauseState.UNPAUSED)

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def is_scheduler_paused(self) -> bool:
            """Return whether the scheduler is in any pause state."""
            return self.scheduler.pause_state != PauseState.UNPAUSED

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _eep_scale_up_before_kv_init(self):
            raise NotImplementedError

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _eep_send_engine_core_notification(
            self,
            notification_type: EEPNotificationType,
            vllm_config: VllmConfig | None = None,
        ):
            raise NotImplementedError


class DyadEngineCoreProc(DyadEngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"

        # >>> DYAD-BEGIN(vllm0.24) DYAD-REBASED: taken from 0.24 verbatim, keeping Dyad's single change
    # Dyad's entire change here relative to 0.12 is annotating the output_queue element type as
    # DyadEngineCoreOutputs instead of EngineCoreOutputs. The 0.24 version adds several instance
    # attributes (tensor_ipc_receiver among them) that the synced 0.24 verbatim methods depend on --
    # porting only Dyad's two lines without following the rest of 0.24 gives
    # AttributeError: 'DyadEngineCoreProc' object has no attribute 'tensor_ipc_receiver'.
    # DYAD-NOTE(vllm0.24): decorators restored exactly as in 0.24. ast.get_source_segment's text for
    # a FunctionDef starts at `def` and **excludes the decorator lines**, which is how the earlier sync
    # tool dropped them wholesale. Their order matters too: @staticmethod must sit above @instrument,
    # or instrument receives a staticmethod object and reading __code__ raises AttributeError.
    @instrument(span_name='EngineCoreProc init')
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
        *,
        engine_index: int = 0,
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, DyadEngineCoreOutputs] | bytes]()  # DYAD-MODIFIED: Dyad output type
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False
        self.shutdown_state = EngineShutdownState.RUNNING

        # Receiver for tensor IPC
        self.tensor_ipc_receiver: TensorIpcReceiver | None = None
        if tensor_queue is not None:
            self.tensor_ipc_receiver = TensorIpcReceiver(tensor_queue)
            logger.info("Using tensor IPC queue for multimodal tensor sharing")

        with self._perform_handshakes(
            handshake_address,
            identity,
            local_client,
            vllm_config,
            client_handshake_address,
        ) as addresses:
            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            logger.debug(
                "Has DP Coordinator: %s, stats publish address: %s",
                self.has_coordinator,
                self.frontend_stats_publish_address,
            )
            internal_dp_balancing = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb
            )
            # Only publish request queue stats to coordinator for "internal"
            # and "hybrid" LB modes.
            self.publish_dp_lb_stats = internal_dp_balancing

            self.addresses = addresses
            self.process_input_queue_block = True
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                self._eep_send_engine_core_notification(
                    EEPNotificationType.NEW_CORE_ENGINES_INIT_READY,
                    vllm_config=vllm_config,
                )
            self._init_data_parallel(vllm_config)

            super().__init__(
                vllm_config,
                executor_class,
                log_stats,
                executor_fail_callback,
                internal_dp_balancing,
            )

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            ready_event = threading.Event()
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            # Don't complete handshake until DP coordinator ready message is
            # received.
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            vllm_config,
            vllm_config.parallel_config,
        )
        if client_handshake_address is None:
            with handshake as addresses:
                yield addresses
        else:
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, vllm_config
            )
            with handshake as addresses, local_handshake as client_addresses:
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    @contextmanager
        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def _perform_handshake(
            self,
            ctx: zmq.Context,
            handshake_address: str,
            identity: bytes,
            local_client: bool,
            headless: bool,
            vllm_config: VllmConfig,
            parallel_config_to_update: ParallelConfig | None = None,
        ) -> Generator[EngineZmqAddresses, None, None]:
            with make_zmq_socket(
                ctx,
                handshake_address,
                zmq.DEALER,
                identity=identity,
                linger=5000,
                bind=False,
            ) as handshake_socket:
                # Register engine with front-end.
                addresses = self.startup_handshake(
                    handshake_socket, local_client, headless, parallel_config_to_update
                )
                yield addresses

                # Send ready message.
                ready_msg = {
                    "status": "READY",
                    "local": local_client,
                    "headless": headless,
                }
                # Include config hash for DP configuration validation
                if vllm_config.parallel_config.data_parallel_size > 1:
                    ready_msg["parallel_config_hash"] = (
                        vllm_config.parallel_config.compute_hash()
                    )

                handshake_socket.send(msgspec.msgpack.encode(ready_msg))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses:
        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # Receive initialization message.
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    # DYAD-REBASED(vllm0.24): this method is vLLM 0.24 verbatim; Dyad's only change is swapping
    # upstream class names / log strings for the Dyad* ones (referencing an upstream class would route
    # around Dyad's changes without raising). No logic of its own.
    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch DyadEngineCore busy loop in background process."""

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        engine_core: DyadEngineCoreProc | None = None
        signal_callback: SignalCallback | None = None
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config: ParallelConfig = vllm_config.parallel_config
            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
            if data_parallel:
                parallel_config.data_parallel_rank_local = local_dp_rank
                process_title = f"EngineCore_DP{dp_rank}"
            else:
                process_title = "DyadEngineCore"
            set_process_title(process_title)
            maybe_init_worker_tracer("vllm.engine_core", "engine_core", process_title)
            decorate_logs()
            if parallel_config.numa_bind:
                numa_utils.log_current_affinity_state(process_title)

            if data_parallel and vllm_config.kv_transfer_config is not None:
                # modify the engine_id and append the dp_rank to it to ensure
                # that the kv_transfer_config is unique for each DP rank.
                vllm_config.kv_transfer_config.engine_id = (
                    f"{vllm_config.kv_transfer_config.engine_id}_dp{dp_rank}"
                )
                logger.debug(
                    "Setting kv_transfer_config.engine_id to %s",
                    vllm_config.kv_transfer_config.engine_id,
                )

            parallel_config.data_parallel_index = dp_rank
            if data_parallel and vllm_config.model_config.is_moe:
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                engine_core = DyadDPEngineCoreProc(*args, **kwargs)
            else:
                # Non-MoE DP ranks are completely independent, so treat like DP=1.
                # Note that parallel_config.data_parallel_index will still reflect
                # the original DP rank.
                parallel_config.data_parallel_size = 1
                parallel_config.data_parallel_size_local = 1
                parallel_config.data_parallel_rank = 0
                engine_core = DyadEngineCoreProc(*args, engine_index=dp_rank, **kwargs)

            assert engine_core is not None

            def wakeup_engine():
                # Wakes up idle engine via input_queue when shutdown is requested
                # Not safe in a signal handler - we may interrupt the main thread
                # while it is holding the non-reentrant input_queue.mutex
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum, frame):
                signal_name = signal.Signals(signum).name
                logger.info(
                    "[shutdown] DyadEngineCore: trigger received signal=%s",
                    signal_name,
                )
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.info_once("[shutdown] DyadEngineCore: exiting busy loop")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("DyadEngineCore failed to start.")
            else:
                logger.exception("DyadEngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    # >>> DYAD-BEGIN(vllm0.24) DYAD-REBASED: taken from 0.24 verbatim.
    # Dyad's 0.12 copy only added an incrementing counter (debug residue) that did nothing, so it is
    # not kept.
    # The upstream 0.24 logic the 0.12 copy was missing matters: the loop condition changed from
    # `while True` to `while self._handle_shutdown()`, raising SystemExit on exit.
    # Staying on 0.12 means the engine process never exits on a shutdown signal and has to be killed
    # from outside.
    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

        raise SystemExit

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def _process_input_queue(self):
            """Exits when an engine step needs to be performed."""

            waited = False
            while not self.has_work() and self.is_running():
                # Notify callbacks waiting for engine to become idle.
                self._notify_idle_state_callbacks()
                if self.input_queue.empty():
                    # Drain aborts queue; all aborts are also processed via input_queue.
                    with self.aborts_queue.mutex:
                        self.aborts_queue.queue.clear()
                    if logger.isEnabledFor(DEBUG):
                        logger.debug("EngineCore waiting for work.")
                        waited = True
                block = self.process_input_queue_block
                try:
                    req = self.input_queue.get(block=block)
                    self._handle_client_request(*req)
                except queue.Empty:
                    break
                if not block:
                    break

            if waited:
                logger.debug("EngineCore loop active.")

            # Handle any more client requests.
            while not self.input_queue.empty():
                req = self.input_queue.get_nowait()
                self._handle_client_request(*req)


        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def _process_engine_step(self) -> bool:
            """Called only when there are unfinished local requests."""

            # Step the engine core.
            outputs, model_executed = self.step_fn()
            # Put EngineCoreOutputs into the output queue.
            for output in outputs.items() if outputs else ():
                self.output_queue.put_nowait(output)
            # Post-step hook.
            self.post_step(model_executed)

            # If no model execution happened but there is still scheduler work
            # (e.g. WAITING_FOR_REMOTE_KVS or delayed KV connector frees), yield
            # the GIL briefly to allow background transfer threads to make progress.
            if not model_executed and self.scheduler.has_requests():
                time.sleep(0.001)

            return model_executed


    # DYAD-REBASED(vllm0.24): this method is vLLM 0.24 verbatim; Dyad's only change is swapping
    # upstream class names / log strings for the Dyad* ones (referencing an upstream class would route
    # around Dyad's changes without raising). No logic of its own.
    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.WAKEUP:
            return
        elif request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            if self._reject_add_in_shutdown(req):
                return
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            if self._reject_utility_in_shutdown(client_idx, call_id, method_name):
                return
            output = UtilityOutput(call_id)
            # Lazily look-up utility method so that failure will be handled/returned.
            get_result = lambda: (
                (method := getattr(self, method_name))
                and method(*self._convert_msgspec_args(method, args))
            )
            enqueue_output = lambda out: self.output_queue.put_nowait(
                (client_idx, DyadEngineCoreOutputs(utility_output=out))
            )
            self._invoke_utility_method(method_name, get_result, output, enqueue_output)
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(DyadEngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def process_input_sockets(
            self,
            input_addresses: list[str],
            coord_input_address: str | None,
            identity: bytes,
            ready_event: threading.Event,
        ):
            """Input socket IO thread."""

            # Msgpack serialization decoding with optional tensor IPC receiver.
            add_request_decoder = MsgpackDecoder(
                EngineCoreRequest, oob_tensor_provider=self.tensor_ipc_receiver
            )
            generic_decoder = MsgpackDecoder(oob_tensor_provider=self.tensor_ipc_receiver)

            with ExitStack() as stack, zmq.Context() as ctx:
                input_sockets = [
                    stack.enter_context(
                        make_zmq_socket(
                            ctx, input_address, zmq.DEALER, identity=identity, bind=False
                        )
                    )
                    for input_address in input_addresses
                ]
                if coord_input_address is None:
                    coord_socket = None
                else:
                    coord_socket = stack.enter_context(
                        make_zmq_socket(
                            ctx,
                            coord_input_address,
                            zmq.XSUB,
                            identity=identity,
                            bind=False,
                        )
                    )
                    # Send subscription message to coordinator.
                    coord_socket.send(b"\x01")

                # Register sockets with poller.
                poller = zmq.Poller()
                ready_response = EngineCoreReadyResponse(
                    max_model_len=self.vllm_config.model_config.max_model_len,
                    num_gpu_blocks=self.vllm_config.cache_config.num_gpu_blocks or 0,
                    block_size=self.vllm_config.cache_config.block_size,
                    dp_stats_address=self.frontend_stats_publish_address,
                    dtype=str(self.vllm_config.model_config.dtype).removeprefix("torch."),
                    vllm_version=VLLM_VERSION,
                    world_size=self.vllm_config.parallel_config.world_size,
                    data_parallel_size=self.vllm_config.parallel_config.data_parallel_size,
                    kv_cache_size_tokens=(
                        self.vllm_config.cache_config.kv_cache_size_tokens
                    ),
                    kv_cache_max_concurrency=(
                        self.vllm_config.cache_config.kv_cache_max_concurrency
                    ),
                )
                ready_payload = msgspec.msgpack.encode(ready_response)
                for input_socket in input_sockets:
                    # Send initial message to each input socket - this is required
                    # before the front-end ROUTER socket can send input messages
                    # back to us.
                    input_socket.send(ready_payload)
                    poller.register(input_socket, zmq.POLLIN)

                if coord_socket is not None:
                    # Wait for ready message from coordinator.
                    assert coord_socket.recv() == b"READY"
                    poller.register(coord_socket, zmq.POLLIN)

                ready_event.set()
                del ready_event
                while True:
                    for input_socket, _ in poller.poll():
                        # (RequestType, RequestData)
                        type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                        # NOTE(yongji): ignore READY message sent by DP coordinator
                        # that is used to notify newly started engines
                        if type_frame.buffer == b"READY":
                            assert input_socket == coord_socket
                            continue
                        request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                        # Deserialize the request data.
                        request: Any
                        if request_type == EngineCoreRequestType.ADD:
                            req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                            try:
                                request = self.preprocess_add_request(req)
                            except Exception:
                                self._handle_request_preproc_error(req)
                                continue
                        else:
                            request = generic_decoder.decode(data_frames)

                            if request_type == EngineCoreRequestType.ABORT:
                                # Aborts are added to *both* queues, allows us to eagerly
                                # process aborts while also ensuring ordering in the input
                                # queue to avoid leaking requests. This is ok because
                                # aborting in the scheduler is idempotent.
                                self.aborts_queue.put_nowait(request)

                        # Push to input queue for core busy loop.
                        self.input_queue.put_nowait((request_type, request))


    def process_output_sockets(
            self,
            output_paths: list[str],
            coord_output_path: str | None,
            engine_index: int,
    ):
        """Output socket IO thread."""

        encoder = MsgpackEncoder()

        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []

        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()


        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]

            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None
                else None
            )

            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()

                if output == DyadEngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break

                assert not isinstance(output, bytes)

                client_index, outputs = output

                outputs.engine_index = engine_index

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue


                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()

                buffers = encoder.encode_into(outputs, buffer)

                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )

                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))

                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def has_work(self) -> bool:
            """Returns true if the engine should be stepped."""
            return (
                self.engines_running
                or self.scheduler.has_requests()
                or bool(self.batch_queue)
            )

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def is_running(self) -> bool:
            """Returns true if shutdown has not been requested."""
            return self.shutdown_state == EngineShutdownState.RUNNING

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _notify_idle_state_callbacks(self) -> None:
            while self._idle_state_callbacks:
                callback = self._idle_state_callbacks.pop()
                callback(self)

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _handle_shutdown(self) -> bool:
            # Check if shutdown was requested and handle it
            if self.shutdown_state == EngineShutdownState.RUNNING:
                return True

            if self.shutdown_state == EngineShutdownState.REQUESTED:
                shutdown_timeout = self.vllm_config.shutdown_timeout
                mode = "abort" if shutdown_timeout == 0 else "drain"

                logger.info(
                    "[shutdown] EngineCore: start mode=%s timeout=%ds",
                    mode,
                    shutdown_timeout,
                )

                if shutdown_timeout == 0:
                    num_requests = self.scheduler.get_num_unfinished_requests()
                    if num_requests > 0:
                        logger.info(
                            "[shutdown] EngineCore: aborting in-flight requests count=%d",
                            num_requests,
                        )
                    aborted_reqs = self.scheduler.finish_requests(
                        None, RequestStatus.FINISHED_ABORTED
                    )
                    self._send_abort_outputs(aborted_reqs)
                else:
                    num_requests = self.scheduler.get_num_unfinished_requests()
                    if num_requests > 0:
                        logger.info(
                            "[shutdown] EngineCore: draining in-flight requests "
                            "count=%d timeout=%ds",
                            num_requests,
                            shutdown_timeout,
                        )

                self.shutdown_state = EngineShutdownState.SHUTTING_DOWN

            # Exit when no work remaining
            if not self.has_work():
                logger.info(
                    "[shutdown] EngineCore: request processing complete; "
                    "starting resource teardown"
                )
                return False

            return True

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _reject_add_in_shutdown(self, request: Request) -> bool:
            if self.shutdown_state == EngineShutdownState.RUNNING:
                return False

            logger.debug(
                "[shutdown] EngineCore: rejecting new request request_id=%s",
                request.request_id,
            )
            self._send_abort_outputs_to_client([request.request_id], request.client_index)
            return True

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _reject_utility_in_shutdown(
            self, client_idx: int, call_id: int, method_name: str
        ) -> bool:
            if self.shutdown_state == EngineShutdownState.RUNNING:
                return False

            logger.warning(
                "[shutdown] EngineCore: rejecting utility call method=%s",
                method_name,
            )
            output = UtilityOutput(call_id, failure_message="Server shutting down")
            self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=output))
            )
            return True

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    # DYAD-NOTE(vllm0.24): decorators restored exactly as in 0.24. ast.get_source_segment's text for
    # a FunctionDef starts at `def` and **excludes the decorator lines**, which is how the earlier sync
    # tool dropped them wholesale. Their order matters too: @staticmethod must sit above @instrument,
    # or instrument receives a staticmethod object and reading __code__ raises AttributeError.
    @staticmethod
    def _invoke_utility_method(
            name: str, get_result: Callable, output: UtilityOutput, enqueue_output: Callable
        ):
            try:
                result = get_result()
                if isinstance(result, Future):
                    # Defer utility output handling until future completion.
                    callback = lambda future: DyadEngineCoreProc._invoke_utility_method(
                        name, future.result, output, enqueue_output
                    )
                    result.add_done_callback(callback)
                    return
                output.result = UtilityResult(result)
            except Exception as e:
                logger.exception("Invocation of %s method failed", name)
                output.failure_message = f"Call to {name} method failed: {str(e)}"
            enqueue_output(output)

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _handle_request_preproc_error(self, request: EngineCoreRequest) -> None:
            """Log and return a request-scoped error response for exceptions raised
            from the add request preprocessing in the input socket processing thread.
            """
            logger.exception(
                "Unexpected error pre-processing request %s", request.request_id
            )
            self._send_error_outputs_to_client([request.request_id], request.client_index)

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def pause_scheduler(
            self, mode: PauseMode = "abort", clear_cache: bool = True
        ) -> Future | None:
            """Pause generation; behavior depends on mode.

            All pause modes queue new adds -- "abort" and "keep" skip step();
            "wait" allows step() so in-flight requests can drain.

            - ``abort``: Set PAUSED_NEW, abort all requests, wait for abort
              outputs to be sent (when running with output_queue), optionally
              clear caches, then complete the returned Future.
            - ``wait``: Set PAUSED_NEW (queue adds, keep stepping); when drained,
              optionally clear caches, then complete the returned Future.
            - ``keep``: Set PAUSED_ALL; return a Future that completes when the
              output queue is empty.
            """
            if mode not in ("keep", "abort", "wait"):
                raise ValueError(f"Invalid pause mode: {mode}")

            def engine_idle_callback(engine: "DyadEngineCoreProc", future: Future[Any]) -> None:
                if clear_cache:
                    engine._reset_caches()
                future.set_result(None)

            if mode == "abort":
                aborted_reqs = self.scheduler.finish_requests(
                    None, RequestStatus.FINISHED_ABORTED
                )
                self._send_abort_outputs(aborted_reqs)

            pause_state = PauseState.PAUSED_ALL if mode == "keep" else PauseState.PAUSED_NEW
            self.scheduler.set_pause_state(pause_state)

            if self._pause_complete():
                if clear_cache:
                    self._reset_caches()
                return None

            future = Future[Any]()
            self._idle_state_callbacks.append(partial(engine_idle_callback, future=future))
            return future

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _pause_complete(self) -> bool:
            """Returns True if the pause has fully completed and the caller can
            return ``None`` synchronously; False if the pause is still pending
            and the caller should register an idle-state callback to finish it.
            """
            return not self.has_work()

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _send_finish_outputs_to_client(
            self, req_ids: list[str], client_index: int, finish_reason: FinishReason
        ) -> None:
            outputs = [
                EngineCoreOutput(req_id, [], finish_reason=finish_reason)
                for req_id in req_ids
            ]
            eco = EngineCoreOutputs(finished_requests=req_ids, outputs=outputs)
            self.output_queue.put_nowait((client_index, eco))

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _send_abort_outputs_to_client(
            self, req_ids: list[str], client_index: int
        ) -> None:
            self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ABORT)

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _send_error_outputs_to_client(
            self, req_ids: list[str], client_index: int
        ) -> None:
            self._send_finish_outputs_to_client(req_ids, client_index, FinishReason.ERROR)

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _send_abort_outputs(self, aborted_reqs: list[tuple[str, int]]) -> None:
            # TODO(nick) this will be moved inside the scheduler
            if aborted_reqs:
                # Map client_index to list of request_ids that belong to that client.
                by_client = defaultdict[int, set[str]](set)
                for req_id, client_index in aborted_reqs:
                    by_client[client_index].add(req_id)
                for client_index, req_ids in by_client.items():
                    self._send_abort_outputs_to_client(list(req_ids), client_index)



class DyadDPEngineCoreProc(DyadEngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def __init__(
            self,
            vllm_config: VllmConfig,
            local_client: bool,
            handshake_address: str,
            executor_class: type[Executor],
            log_stats: bool,
            client_handshake_address: str | None = None,
            tensor_queue: Queue | None = None,
        ):
            assert vllm_config.model_config.is_moe, (
                "DPEngineCoreProc should only be used for MoE models"
            )

            scheduler_config = vllm_config.scheduler_config
            self.prefill_schedule_interval = scheduler_config.prefill_schedule_interval

            # Counts forward-passes of the model so that we can synchronize
            # finished with DP peers every N steps.
            self.step_counter = 0
            self.current_wave = 0
            self.last_counts = (0, 0)

            # Two-phase pause protocol state. When pending_pause is True, the
            # engine keeps stepping (dummy batches) while waiting for all DP
            # ranks to also set pending_pause. Once all ranks agree via
            # all-reduce, ignore_start_dp_wave is set so that stale
            # START_DP_WAVE messages cannot re-wake the engines.
            self.pending_pause = False
            self.ignore_start_dp_wave = False

            from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

            self.eep_scaling_state: ElasticEPScalingState | None = None

            # Initialize the engine.
            dp_rank = vllm_config.parallel_config.data_parallel_rank
            super().__init__(
                vllm_config,
                local_client,
                handshake_address,
                executor_class,
                log_stats,
                client_handshake_address,
                engine_index=dp_rank,
                tensor_queue=tensor_queue,
            )

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def _init_data_parallel(self, vllm_config: VllmConfig):
            # Configure GPUs and stateless process group for data parallel.
            parallel_config = vllm_config.parallel_config
            dp_rank = parallel_config.data_parallel_rank
            dp_size = parallel_config.data_parallel_size
            local_dp_rank = parallel_config.data_parallel_rank_local

            assert dp_size > 1
            assert local_dp_rank is not None
            assert 0 <= local_dp_rank <= dp_rank < dp_size

            self.dp_rank = dp_rank
            self.dp_size = dp_size
            dp_group, dp_store = parallel_config.stateless_init_dp_group(return_store=True)
            self.dp_group, self.dp_store = dp_group, dp_store

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    # DYAD-REBASED(vllm0.24): this method is vLLM 0.24 verbatim; Dyad's only change is swapping
    # upstream class names / log strings for the Dyad* ones (referencing an upstream class would route
    # around Dyad's changes without raising). No logic of its own.
    def add_request(self, request: Request, request_wave: int = 0):
        super().add_request(request, request_wave)
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif (
                not self.engines_running
                and self.scheduler.pause_state == PauseState.UNPAUSED
            ):
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.engines_running = True
                self.output_queue.put_nowait(
                    (-1, DyadEngineCoreOutputs(start_wave=self.current_wave))
                )

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def _handle_client_request(
            self, request_type: EngineCoreRequestType, request: Any
        ) -> None:
            if request_type == EngineCoreRequestType.START_DP_WAVE:
                if self.ignore_start_dp_wave:
                    return
                new_wave, exclude_eng_index = request
                if exclude_eng_index != self.engine_index and (
                    new_wave >= self.current_wave
                ):
                    self.current_wave = new_wave
                    if not self.engines_running:
                        logger.debug(
                            "EngineCore starting idle loop for wave %d.",
                            new_wave,
                        )
                        self.engines_running = True
            else:
                super()._handle_client_request(request_type, request)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts, step_counter=self.step_counter, current_wave=self.current_wave
            )
            self.output_queue.put_nowait((-1, DyadEngineCoreOutputs(scheduler_stats=stats)))

    # DYAD-REBASED(vllm0.24): this method is vLLM 0.24 verbatim; Dyad's only change is swapping
    # upstream class names / log strings for the Dyad* ones (referencing an upstream class would route
    # around Dyad's changes without raising). No logic of its own.
    def run_busy_loop(self):
        """Core busy loop of the DyadEngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # Publish request counts before and after GPU step to ensure freshness.
            self._maybe_publish_request_counts()

            if self.eep_scaling_state is not None:
                _ = self.eep_scaling_state.progress()
                if self.eep_scaling_state.is_complete():
                    if self.eep_scaling_state.worker_type == "removing":
                        raise SystemExit
                    self.process_input_queue_block = True
                    self.eep_scaling_state = None

            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # Execute a dummy pass when no ready requests ran, unless the
                # engine is sleeping.
                elif not self.model_executor.is_sleeping:
                    with self.log_iteration_details(None):
                        self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            DyadEngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0

        raise SystemExit

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
            # Optimization - only perform finish-sync all-reduce every 32 steps.
            self.step_counter += 1
            if self.step_counter % 32 != 0:
                return True

            has_unfinished, pause_consensus = ParallelConfig.sync_dp_state(
                self.dp_group,
                has_unfinished=local_unfinished,
                pending_pause=self.pending_pause,
            )

            if pause_consensus:
                self.ignore_start_dp_wave = True
                self.pending_pause = False
                logger.debug("DP pause consensus reached, ignoring START_DP_WAVE.")

            return has_unfinished

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def reinitialize_distributed(
            self, reconfig_request: ReconfigureDistributedRequest
        ) -> None:
            from copy import deepcopy

            from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

            new_parallel_config = deepcopy(self.vllm_config.parallel_config)
            old_dp_size = new_parallel_config.data_parallel_size
            new_parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
            if (
                reconfig_request.new_data_parallel_rank
                != ReconfigureRankType.KEEP_CURRENT_RANK
            ):
                new_parallel_config.data_parallel_rank = (
                    reconfig_request.new_data_parallel_rank
                )
            new_parallel_config.data_parallel_master_ip = (
                reconfig_request.new_data_parallel_master_ip
            )
            new_parallel_config.data_parallel_master_port = (
                reconfig_request.new_data_parallel_master_port
            )
            new_parallel_config._data_parallel_master_port_list = (
                reconfig_request.new_data_parallel_master_port_list
            )
            new_parallel_config._coord_store_port = reconfig_request.coord_store_port

            is_scale_down = reconfig_request.new_data_parallel_size < old_dp_size
            is_shutdown = (
                reconfig_request.new_data_parallel_rank
                == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
            )

            self.eep_scaling_state = ElasticEPScalingState(
                model_executor=self.model_executor,
                engine_core=self,
                vllm_config=self.vllm_config,
                new_parallel_config=new_parallel_config,
                worker_type="removing" if is_shutdown else "existing",
                scale_type="scale_down" if is_scale_down else "scale_up",
                reconfig_request=reconfig_request,
            )
            self.process_input_queue_block = False
            logger.info(
                "[Elastic EP] Received reconfiguration request and starting scaling up/down"
            )

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _pause_complete(self) -> bool:
            """Two-phase DP-aware pause.

            Phase 1: Set local pause state and ``pending_pause`` flag. If the
            engines are idle, kick-start them by setting ``engines_running`` to
            True so ranks enter the stepping loop and reach the all-reduce
            consensus checkpoint in ``_has_global_unfinished_reqs``.

            Phase 2 (in ``_has_global_unfinished_reqs``): Once the all-reduce
            confirms that **all** ranks have ``pending_pause`` set, collectively
            stop stepping and set ``ignore_start_dp_wave`` so that stale
            ``START_DP_WAVE`` messages cannot re-wake any engine.
            """
            self.pending_pause = True
            self.engines_running = True

            return False

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def resume_scheduler(self):
            if self.pending_pause or (self.engines_running and self.ignore_start_dp_wave):
                raise RuntimeError(
                    "resume_scheduler called while pause is still in "
                    "flight. Wait for the pause future to resolve before "
                    "resuming."
                )
            if self.engines_running:
                logger.debug("Resume called while engines are not paused, ignoring.")
                return

            super().resume_scheduler()
            self.ignore_start_dp_wave = False

            # Barrier: wait for all DP ranks to have resumed (and cleared
            # ignore_start_dp_wave) before any rank starts stepping. Uses
            # the existing all-reduce which is safe because engines are
            # stopped.
            has_global_unfinished = ParallelConfig.has_unfinished_dp(
                self.dp_group, self.scheduler.has_unfinished_requests()
            )

            if has_global_unfinished:
                self.engines_running = True

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def barrier(self):
            """Blocking barrier on the DP process group (test-only utility)."""
            import torch.distributed as dist

            dist.barrier(group=self.dp_group)

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _should_throttle_prefills(self) -> bool:
            # Throttle new prefills to cadence-aligned steps for DP balancing.
            # step_counter is identical across DP ranks. On a fresh wave the
            # counter is 0, so prefills are admitted immediately after idle.
            return (
                self.prefill_schedule_interval > 1
                and self.step_counter % self.prefill_schedule_interval != 0
            )

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _eep_send_engine_core_notification(
            self,
            notification_type: EEPNotificationType,
            vllm_config: VllmConfig | None = None,
        ):
            """
            Send notifications to EngineCoreClient, which can then forward
            the notifications to other engine core processes. It is used for:
            1) In scale up: new core engines to notify existing core engines
               that they are ready;
            2) In scale down: removing core engines to notify EngineCoreClient
               so EngineCoreClient can release their ray placement groups;
            3) Both scale up/down: to notify EngineCoreClient that existing
               core engines have already switched to the new parallel setup.
            """
            if vllm_config is None:
                dp_rank = self.vllm_config.parallel_config.data_parallel_rank
            else:
                dp_rank = vllm_config.parallel_config.data_parallel_rank
            notification_data = (notification_type.value, dp_rank)
            outputs = EngineCoreOutputs(
                utility_output=UtilityOutput(
                    call_id=EEP_NOTIFICATION_CALL_ID,
                    result=UtilityResult(notification_data),
                )
            )
            outputs.engine_index = self.engine_index

            if hasattr(self, "output_thread") and self.output_thread.is_alive():
                self.output_queue.put_nowait((0, outputs))
            else:
                encoder = MsgpackEncoder()
                with (
                    zmq.Context() as ctx,
                    make_zmq_socket(
                        ctx, self.addresses.outputs[0], zmq.PUSH, linger=4000
                    ) as socket,
                ):
                    socket.send_multipart(encoder.encode(outputs))

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def eep_handle_engine_core_notification(
            self, notification_type: str | EEPNotificationType
        ):
            """
            Handle notification received from EngineCoreClient
            (forwarded from new core engines).
            """
            assert self.eep_scaling_state is not None
            if isinstance(notification_type, str):
                notification_type = EEPNotificationType(notification_type)
            self.eep_scaling_state.handle_notification(notification_type)

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried over verbatim.
    def _eep_scale_up_before_kv_init(self):
            from vllm.distributed.elastic_ep.elastic_state import ElasticEPScalingState

            self.eep_scaling_state = ElasticEPScalingState(
                model_executor=self.model_executor,
                engine_core=self,
                vllm_config=self.vllm_config,
                new_parallel_config=self.vllm_config.parallel_config,
                worker_type="new",
                scale_type="scale_up",
                reconfig_request=None,
            )
            self.eep_scaling_state.run_pre_kv_init_states()
            self.process_input_queue_block = False


class DPEngineCoreActor(DyadDPEngineCoreProc):
    """
    Ray actor for running EngineCore in a data parallel context
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_rank = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but vLLM workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # This is problematic because when the vLLM worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error. See:
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # and get_accelerator_ids_for_accelerator_resource() in worker.py
        # of ray.
        self._set_visible_devices(vllm_config, local_dp_rank)

        super().__init__(vllm_config, local_client, "", executor_class, log_stats)

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        from vllm.platforms import current_platform

        if current_platform.is_xpu():
            pass
        else:
            device_control_env_var = current_platform.device_control_env_var
            self._set_assigned_physical_gpu_ids(
                vllm_config, local_dp_rank, device_control_env_var
            )

    # >>> DYAD-BEGIN(vllm0.24): from "mutate an environment variable" to "write into parallel_config"
    #
    # 0.12: get_device_indices(...) returned a string written into os.environ[CUDA_VISIBLE_DEVICES],
    #       using the process environment to hand this DP rank its visible devices.
    # 0.24: that function was deleted in favour of get_physical_gpu_ids_for_local_dp_rank(...),
    #       which returns list[int] and **no longer touches the environment** -- the assignment is
    #       stored in vllm_config.parallel_config.assigned_physical_gpu_ids for downstream to read.
    #
    # This is not a rename: the carrier of the side effect moved from the process environment to a
    # config object. Keeping 0.12's os.environ assignment leaves 0.24's downstream never reading it,
    # which shows up as every DP rank landing on the same GPUs -- no error, just OOM or wrong results.
    # The method name follows upstream too, to keep diffing against upstream core.py easy.
    def _set_assigned_physical_gpu_ids(
        self, vllm_config: VllmConfig, local_dp_rank: int, device_control_env_var: str
    ):
        world_size = vllm_config.parallel_config.world_size
        try:
            physical_gpu_ids = get_physical_gpu_ids_for_local_dp_rank(
                device_control_env_var,
                local_dp_rank,
                world_size,
                user_assigned_gpu_ids=(
                    vllm_config.parallel_config.assigned_physical_gpu_ids
                ),
            )
            vllm_config.parallel_config.assigned_physical_gpu_ids = physical_gpu_ids
        except IndexError as e:
            raise Exception(
                f"Error computing assigned_physical_gpu_ids: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e
    # <<< DYAD-END

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ):
        """
        For Ray, we don't need to actually perform handshake.
        All addresses information is known before the policy LLM backbone creation.
        Therefore, we simply yield these addresses.
        """
        yield self.addresses

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the policy LLM backbone) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            self.run_busy_loop()
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()
