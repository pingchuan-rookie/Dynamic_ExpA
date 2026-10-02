# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import socket
import time
from collections.abc import AsyncGenerator, Iterable, Mapping
from copy import copy
from typing import Any, cast

import numpy as np
import torch
from typing_extensions import deprecated

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
# DYAD-NOTE(vllm0.24): the whole vllm.entrypoints.utils module was deleted and
# _validate_truncation_size has no replacement. 0.24 moved prompt-truncation validation into the
# renderer and the call site was removed accordingly; see generate() below.
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
# from vllm.v1.dyad.dyad_outputs import  DyadRequestOutput
# from vllm.outputs import PoolingRequestOutput

from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
# DYAD-NOTE(vllm0.24): init_tokenizer_from_config was deleted. 0.24 introduced the renderer, which
# owns the tokenizer, and InputProcessor now takes a renderer instead of a tokenizer.
from vllm.tokenizers import TokenizerLike
from vllm.renderers import renderer_from_config
from vllm.tracing import init_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.usage.usage_lib import UsageContext
from vllm.utils.async_utils import cancel_task_threadsafe
from vllm.utils.collection_utils import as_list
from vllm.utils.math_utils import cdiv
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.input_processor import InputProcessor
from .dyad_output_processor import DyadOutputProcessor, DyadRequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import (
    StatLoggerFactory,
    StatLoggerManager,
    load_stat_logger_plugin_factories,
)
from vllm.v1.metrics.prometheus import shutdown_prometheus
from vllm.v1.metrics.stats import IterationStats

from .dyad_core_client import DyadEngineCoreClient


from vllm.v1.engine.async_llm import AsyncLLM
from vllm.inputs import EngineInput  # DYAD-NOTE(vllm0.24): introduced by syncing 0.24's verbatim methods

from vllm.engine.protocol import StreamingInput  # DYAD-NOTE(vllm0.24): introduced by syncing 0.24's verbatim methods

# >>> DYAD-BEGIN(vllm0.24): dependencies introduced by syncing 0.24's verbatim methods
from vllm.v1.engine.async_llm import InputStreamError
from vllm.outputs import STREAM_FINISHED
# <<< DYAD-END

logger = init_logger(__name__)


def validate_dyad_rollout_turn(rollout_turn: int | None, max_turns: int | None) -> tuple[int | None, int | None]:
    """Validate Dyad trajectory turn metadata before submitting a vLLM request."""
    if rollout_turn is None and max_turns is None:
        return None, None
    if rollout_turn is None or max_turns is None:
        raise ValueError("Dyad rollout turn validation requires both rollout_turn and max_turns.")

    rollout_turn = int(rollout_turn)
    max_turns = int(max_turns)
    if rollout_turn <= 0:
        raise ValueError(f"Dyad rollout_turn must be positive, got {rollout_turn}.")
    if max_turns <= 0:
        raise ValueError(f"Dyad max_turns must be positive, got {max_turns}.")
    if rollout_turn > max_turns:
        raise RuntimeError(f"Dyad rollout turn {rollout_turn} exceeds max_turns={max_turns}.")
    return rollout_turn, max_turns


class DyadAsyncLLM(AsyncLLM):
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list[StatLoggerFactory] | None = None,
        aggregate_engine_logging: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None:
        maybe_register_config_serialize_by_value()


        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.observability_config = vllm_config.observability_config
        self.log_requests = log_requests
        custom_stat_loggers = list(stat_loggers or [])
        custom_stat_loggers.extend(load_stat_logger_plugin_factories())
        has_custom_loggers = bool(custom_stat_loggers)
        self.log_stats = log_stats or has_custom_loggers
        if not log_stats and has_custom_loggers:
            logger.info(
                "AsyncLLM created with log_stats=False, "
                "but custom stat loggers were found; "
                "enabling logging without default stat loggers."
            )
        # >>> DYAD-BEGIN(vllm0.24): the renderer owns the tokenizer
        # 0.12: call init_tokenizer_from_config here, then hand the tokenizer to InputProcessor.
        # 0.24: renderer_from_config builds the renderer, the tokenizer is one of its attributes, and
        #       InputProcessor takes the renderer. The skip_tokenizer_init branch is handled inside
        #       the renderer, so there is nothing to decide here any more.
        self.renderer = renderer = renderer_from_config(self.vllm_config)
        self.input_processor = InputProcessor(self.vllm_config, renderer)
        # <<< DYAD-END
        self.io_processor = get_io_processor(
            self.vllm_config,
            self.model_config.io_processor_plugin,
        )
        self.output_processor = DyadOutputProcessor(
            self.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
        )
        endpoint = self.observability_config.otlp_traces_endpoint
        if endpoint is not None:
            tracer = init_tracer("vllm.llm_engine", endpoint)
            self.output_processor.tracer = tracer
        self.engine_core = DyadEngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )
        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                engine_idxs=self.engine_core.engine_ranks_managed,
                custom_stat_loggers=custom_stat_loggers,
                enable_default_loggers=log_stats,
                client_count=client_count,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()
        self._pause_cond = asyncio.Condition()
        self._paused = False
        self.output_handler: asyncio.Task | None = None
        try:
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass
        # DYAD-NOTE(vllm0.24): 0.12's torch profiler block was deleted. 0.24's vllm.envs no longer
        # has VLLM_TORCH_PROFILER_DIR / VLLM_TORCH_PROFILER_DISABLE_ASYNC_LLM /
        # VLLM_PROFILER_MAX_ITERS, and 0.24's async_llm.py contains no profiler code at all (the
        # capability moved elsewhere). Keeping it would raise
        # AttributeError: module 'vllm.envs' has no attribute 'VLLM_TORCH_PROFILER_DIR'.
        else:
            self.profiler = None

    @property
    @deprecated(
        "`AsyncLLM.processor` has been renamed to `AsyncLLM.input_processor`. "
        "The old name will be removed in v0.13."
    )
    def processor(self):
        return self.input_processor

    # >>> DYAD-BEGIN(vllm0.24) DYAD-REBASED: taken from 0.24 verbatim; Dyad adds only the worker_cls
    # and dyad_config lines. The 0.12 copy was missing 0.24's new arguments: enable_log_requests,
    # aggregate_engine_logging, client_count and client_index -- without them, in a multi-client
    # setup every client believes it is client 0.
    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "DyadAsyncLLM":
        # >>> DYAD-BEGIN(dyad): make the engine start an DyadWorker, and mark dyad_config
        # worker_cls decides which Worker the engine process instantiates. Without this line it
        # starts the upstream Worker -> upstream GPUModelRunner, and Dyad's sampling hooks are not
        # in the call chain at all -- nothing raises, Dyad simply does not take effect.
        vllm_config.parallel_config.worker_cls = f"{__package__}.dyad_gpu_worker.DyadWorker"
        vllm_config.dyad_config = "ALFworld"
        assert hasattr(vllm_config, "dyad_config") and vllm_config.dyad_config is not None, \
            "vllm_config.dyad_config is missing or None"
        # <<< DYAD-END
        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            aggregate_engine_logging=aggregate_engine_logging,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
    ) -> "DyadAsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""
        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        vllm_config.parallel_config.worker_cls = f"{__package__}.dyad_gpu_worker.DyadWorker"
        vllm_config.dyad_config = "ALFworld"
        assert hasattr(vllm_config, "dyad_config") and vllm_config.dyad_config is not None, \
            "vllm_config.dyad_config is missing or None"
        # Create the AsyncLLM.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    async def add_request(
            self,
            request_id: str,
            prompt: EngineCoreRequest | PromptType,
            params: SamplingParams | PoolingParams,
            arrival_time: float | None = None,
            lora_request: LoRARequest | None = None,
            tokenization_kwargs: dict[str, Any] | None = None,
            trace_headers: Mapping[str, str] | None = None,
            priority: int = 0,
            data_parallel_rank: int | None = None,
            prompt_text: str | None = None,
            # DYAD-NOTE(vllm0.24): 0.24 added these two reasoning-related arguments to add_request.
            # Upstream's generate() passes them by keyword, so not accepting them is a TypeError.
            # Dyad does not use reasoning chains; they are accepted and ignored, purely to stay
            # signature-compatible with upstream.
            reasoning_ended: bool | None = None,
            reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> DyadRequestOutputCollector:
        """Add a new request to the AsyncLLM and return the collector used to gather its outputs."""

        if self.errored:
            raise EngineDeadError()

        # Besides ordinary generation (sampling), vLLM may also serve pooling / embedding requests.
        is_pooling = isinstance(params, PoolingParams)

        # Outputs produced later during decoding are pushed into this queue / collector.
        # output_kind comes from params, and may e.g. decide whether deltas or final results are returned.
        queue = DyadRequestOutputCollector(output_kind=params.output_kind, request_id=request_id)

        # ----------------------------
        # Step 1: normalize the input into an EngineCoreRequest
        # ----------------------------
        # prompt may already be an EngineCoreRequest (the caller upstream already processed it),
        # or it may just be a raw prompt (e.g. str / token ids / mapping).
        if isinstance(prompt, EngineCoreRequest):
            request = prompt
        else:
            # When prompt is not yet an EngineCoreRequest, prompt_text must not be passed in early,
            # because it is inferred below from the raw prompt.
            assert prompt_text is None

            # Call input_processor to wrap the user input into an EngineCoreRequest.
            # This usually performs:
            # 1. prompt normalization
            # 2. tokenizer-related processing
            # 3. params validation / completion
            # 4. attaching LoRA, trace, priority, DP rank and similar information
            # DYAD-NOTE(vllm0.24): process_inputs gained a required supported_tasks argument,
            # positioned after params. The old positional call put arrival_time into the
            # supported_tasks slot, and iterating it later raised
            # TypeError: 'NoneType' object is not iterable -- reported inside add_request, with no
            # sign that an argument had shifted. Everything else is passed by keyword now, so the
            # next upstream insertion cannot repeat this.
            request = self.input_processor.process_inputs(
                request_id,
                prompt,
                params,
                supported_tasks=await self.get_supported_tasks(),
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
            )

            # prompt_text is recorded mainly for display or debugging on the output side.
            if isinstance(prompt, str):
                prompt_text = prompt

            elif isinstance(prompt, Mapping):
                prompt_text = cast(str | None, prompt.get("prompt"))

        # >>> DYAD-BEGIN(vllm0.24): set external_req_id in one place
        # 0.24's RequestState.from_new_request contains
        # `assert request.external_req_id is not None`.
        # The field is set by InputProcessor.assign_request_id, and that is the **caller's**
        # responsibility -- process_inputs does not call it, and upstream sets it on the Renderer
        # path. Neither of Dyad's branches (hand-built EngineCoreRequest, or via process_inputs) goes
        # through the Renderer, so it is filled in here for both. Placed after the branches merge, so
        # both paths are covered. Semantics match assign_request_id: external_req_id is the
        # externally visible request id.
        if getattr(request, "external_req_id", None) is None:
            request.external_req_id = request.request_id
        # <<< DYAD-END

        # This is crucial:
        # process_inputs() may "clone and modify" params, so the original params must not be used
        # afterwards; request.params is the authoritative one.
        params = request.params

        # ----------------------------
        # Step 2: handle a single request vs a fan-out request
        # ----------------------------
        if is_pooling or params.n == 1:
            # Actually register the request with:
            # - the output_processor of the current process
            # - the background engine_core process
            await self._add_request(request, prompt_text, None, 0, queue)
            return queue

        # Reaching here means:
        # - not pooling
        # - and it is a sampling request
        # - and n > 1, so it must be fanned out into multiple child requests
        parent_params = params
        assert isinstance(parent_params, SamplingParams)

        # Create a parent request object for the n>1 case.
        # ParentRequest is responsible for managing:
        # - the original request_id
        # - the request_id of each child
        # - the SamplingParams of each child
        # - how the multiple child outputs are finally aggregated back together
        parent_request = ParentRequest(request_id, parent_params)

        for idx in range(parent_params.n):
            # child_params usually rewrites n to 1, since each child only produces one result.
            request_id, child_params = parent_request.get_child_info(idx)

            # To avoid one unnecessary copy, the last child reuses the original request;
            # the earlier children use a shallow copy(request).
            child_request = request if idx == parent_params.n - 1 else copy(request)

            child_request.request_id = request_id

            # Note the field name here is sampling_params, while above we uniformly read
            # request.params; which one applies depends on the attribute definitions inside
            # EngineCoreRequest.
            child_request.sampling_params = child_params

            # Register the child request with the system, and tell output_processor:
            # - which parent_request it belongs to
            # - its index idx within that parent
            await self._add_request(
                child_request, prompt_text, parent_request, idx, queue
            )

        # The caller sees a single collector, but multiple child requests are attached
        # underneath; output_processor is responsible for aggregating them into this queue.
        return queue

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    async def generate(
            self,
            prompt: EngineCoreRequest
            | PromptType
            | EngineInput
            | AsyncGenerator[StreamingInput, None],
            sampling_params: SamplingParams,
            request_id: str,
            *,
            prompt_text: str | None = None,
            lora_request: LoRARequest | None = None,
            tokenization_kwargs: dict[str, Any] | None = None,
            trace_headers: Mapping[str, str] | None = None,
            priority: int = 0,
            data_parallel_rank: int | None = None,
            reasoning_ended: bool | None = None,
            reasoning_parser_kwargs: dict[str, Any] | None = None,
        ) -> AsyncGenerator[RequestOutput, None]:
            """
            Main function called by the API server to kick off a request
                * 1) Making an AsyncStream corresponding to the Request.
                * 2) Processing the Input.
                * 3) Adding the Request to the Detokenizer.
                * 4) Adding the Request to the EngineCore (separate process).

            A separate output_handler loop runs in a background AsyncIO task,
            pulling outputs from EngineCore and putting them into the
            per-request AsyncStream.

            The caller of generate() iterates the returned AsyncGenerator,
            returning the RequestOutput back to the caller.
            """

            # DYAD-NOTE(vllm0.24): 0.24 annotates this as vllm's RequestOutputCollector, but this
            # class's add_request returns an DyadRequestOutputCollector (a standalone class, not a
            # subclass of it). The annotation follows suit so nobody later assumes this is the
            # upstream object.
            q: DyadRequestOutputCollector | None = None
            try:
                q = await self.add_request(
                    request_id,
                    prompt,
                    sampling_params,
                    lora_request=lora_request,
                    tokenization_kwargs=tokenization_kwargs,
                    trace_headers=trace_headers,
                    priority=priority,
                    data_parallel_rank=data_parallel_rank,
                    prompt_text=prompt_text,
                    reasoning_ended=reasoning_ended,
                    reasoning_parser_kwargs=reasoning_parser_kwargs,
                )

                # The output_handler task pushes items into the queue.
                # This task pulls from the queue and yields to caller.
                finished = False
                while not finished:
                    # Note: drain queue without await if possible (avoids
                    # task switching under load which helps performance).
                    out = q.get_nowait() or await q.get()

                    # Note: both OutputProcessor and EngineCore handle their
                    # own request cleanup based on finished.
                    assert isinstance(out, RequestOutput)
                    finished = out.finished
                    if out is not STREAM_FINISHED:
                        yield out

            # If the request is disconnected by the client, generate()
            # is cancelled or the generator is garbage collected. So,
            # we abort the request if we end up here.
            except (asyncio.CancelledError, GeneratorExit):
                if q is not None:
                    await self.abort(q.request_id, internal=True)
                if self.log_requests:
                    logger.info("Request %s aborted.", request_id)
                raise

            # Engine is dead. Do not abort since we shut down.
            except EngineDeadError:
                if self.log_requests:
                    logger.info("Request %s failed (engine dead).", request_id)
                raise

            # Request validation error.
            except ValueError as e:
                if self.log_requests:
                    logger.info("Request %s failed (bad request): %s.", request_id, e)
                raise

            # Error from input stream generator - propagate directly.
            except InputStreamError as e:
                if q is not None:
                    await self.abort(q.request_id, internal=True)
                if self.log_requests:
                    logger.info("Request %s failed (input error): %s.", request_id, e)
                raise e.cause from e

            # Unexpected error in the generate() task (possibly recoverable).
            except Exception as e:
                if q is not None:
                    await self.abort(q.request_id, internal=True)
                if self.log_requests:
                    try:
                        s = f"{e.__class__.__name__}: {e}"
                    except Exception as e2:
                        s = (
                            f"{e.__class__.__name__}: "
                            "error during printing an exception of class"
                            + e2.__class__.__name__
                        )
                    logger.info("Request %s failed due to %s.", request_id, s)
                raise EngineGenerateError() from e
            finally:
                if q is not None:
                    q.close()


    # >>> DYAD-BEGIN(vllm0.24) DYAD-REBASED: taken from 0.24 verbatim; Dyad adds exactly one thing,
    # an extra outputs.action_content argument to process_outputs.
    # Upstream logic the 0.12 copy was missing: return immediately when output_handler already exists
    # (so no duplicate task is created); logger_manager switched to the mutable list _logger_ref
    # (elastic EP scaling needs to update it in place, and referencing self directly creates a
    # reference cycle that is never collected); renderer.stat_mm_cache() added to the log.
    def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        if self.output_handler is not None:
            return

        # Ensure that the task doesn't have a circular ref back to the AsyncLLM
        # object, or else it won't be garbage collected and cleaned up properly.
        engine_core = self.engine_core
        output_processor = self.output_processor
        log_stats = self.log_stats
        # We use a mutable list for logger_manager so that it can be updated
        # during elastic EP scaling (see scale_elastic_ep) without creating
        # a circular reference via self.
        self._logger_ref = [self.logger_manager]
        logger_ref = self._logger_ref
        renderer = self.renderer
        chunk_size = envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE

        async def output_handler():
            try:
                while True:
                    # 1) Pull EngineCoreOutputs from the EngineCore.
                    outputs = await engine_core.get_output_async()
                    num_outputs = len(outputs.outputs)

                    iteration_stats = (
                        IterationStats() if (log_stats and num_outputs) else None
                    )

                    # Split outputs into chunks of at most
                    # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                    # event loop for too long.
                    engine_core_outputs = outputs.outputs
                    for start in range(0, num_outputs, chunk_size):
                        end = start + chunk_size
                        outputs_slice = engine_core_outputs[start:end]
                        # 2) Process EngineCoreOutputs.
                        # >>> DYAD-BEGIN(dyad): carry this batch's action_content into output
                        # processing. Dyad's core data path: the action chosen by sample_tokens on
                        # the rollout side comes out via DyadEngineCoreOutputs.action_content and
                        # must be handed to process_outputs here, or the training side never learns
                        # what the policy actually chose.
                        # Omitting this argument raises nothing -- the parameter on
                        # DyadOutputProcessor.process_outputs has a default -- action_content is
                        # simply always None, the training side cannot rebuild the mask, and Dyad
                        # silently degrades into ordinary sampling.
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice,
                            outputs.timestamp,
                            iteration_stats,
                            outputs.action_content,
                        )
                        # <<< DYAD-END
                        # NOTE: RequestOutputs are pushed to their queues.
                        assert not processed_outputs.request_outputs

                        # Allow other asyncio tasks to run between chunks
                        if end < num_outputs:
                            await asyncio.sleep(0)

                        # 3) Abort any reqs that finished due to stop strings.
                        if processed_outputs.reqs_to_abort:
                            await engine_core.abort_requests_async(
                                processed_outputs.reqs_to_abort
                            )

                    output_processor.update_scheduler_stats(outputs.scheduler_stats)

                    # 4) Logging.
                    # TODO(rob): make into a coroutine and launch it in
                    # background thread once Prometheus overhead is non-trivial.
                    if logger_ref[0]:
                        logger_ref[0].record(
                            engine_idx=outputs.engine_index,
                            scheduler_stats=outputs.scheduler_stats,
                            iteration_stats=iteration_stats,
                            mm_cache_stats=renderer.stat_mm_cache(),
                        )
            except Exception as e:
                logger.exception("AsyncLLM output_handler failed.")
                output_processor.propagate_error(e)

        self.output_handler = asyncio.create_task(output_handler())

    def _get_dyad_tokenizer(self):
        from transformers import AutoTokenizer
        tokenizer_name = getattr(self.model_config, "tokenizer", None)
        if tokenizer_name is None:
            tokenizer_name = self.model_config.model

        revision = getattr(self.model_config, "tokenizer_revision", None)
        if revision is None:
            revision = getattr(self.model_config, "revision", None)

        self._dyad_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.model_config.trust_remote_code,
            revision=revision,
        )


    def _encode_text(self, text: str) -> list[int]:
        return list(self._dyad_tokenizer.encode(text, add_special_tokens=False))
