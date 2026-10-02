# >>> DYAD-BEGIN(vllm0.24): restore the dependency the decorators need
# `instrument` is 0.24's tracing-span decorator. Losing it does not crash anything; these methods
# simply stop producing spans -- a silent loss of functionality, exactly the class of failure the
# rebase exists to avoid.
from vllm.tracing import instrument
# <<< DYAD-END
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
import multiprocessing
import queue
import sys
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Thread
from typing import Any, TypeAlias, TypeVar

import msgspec.msgpack
import zmq
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.utils.async_utils import in_loop
from vllm.utils.network_utils import (
    close_sockets,
    get_open_port,
    get_open_zmq_inproc_path,
    make_zmq_socket,
)
from vllm.v1.engine import (
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
    # DYAD-NOTE(vllm0.24): types introduced by syncing 0.24's verbatim methods
    PauseMode,
)
from vllm.v1.engine.coordinator import DPCoordinator
from .dyad_core import DyadEngineCore, DyadEngineCoreProc
from vllm.v1.engine.exceptions import EngineDeadError
from .dyad_core_engine import (
    CoreEngineActorManager,
    CoreEngineProcManager,
    launch_core_engines,
)
from vllm.v1.executor import Executor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr
from .dyad_outputs import DyadEngineCoreOutputs

logger = init_logger(__name__)

AnyFuture: TypeAlias = asyncio.Future[Any] | Future[Any]

_R = TypeVar("_R")  # Return type for collective_rpc

EngineIdentity = bytes


class DyadEngineCoreClient(ABC):
    """
    EngineCoreClient: subclasses handle different methods for pushing
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
    """

    @staticmethod
    def make_client(
        multiprocess_mode: bool,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ) -> "DyadEngineCoreClient":
        # TODO: support this for debugging purposes.
        if asyncio_mode and not multiprocess_mode:
            raise NotImplementedError(
                "Running EngineCore in asyncio without multiprocessing "
                "is not currently supported."
            )

        if multiprocess_mode and asyncio_mode:
            return DyadEngineCoreClient.make_async_mp_client(
                vllm_config, executor_class, log_stats
            )

        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(vllm_config, executor_class, log_stats)

        return DyadInprocClient(vllm_config, executor_class, log_stats)

    # DYAD-NOTE(vllm0.24): decorators restored exactly as 0.24 has them. ast.get_source_segment's
    # text for a FunctionDef starts at `def` and **excludes the decorator lines**, which is how the
    # earlier sync tool dropped them wholesale. Their order matters too: @staticmethod must sit
    # above @instrument, or instrument receives a staticmethod object and reading __code__ raises
    # AttributeError.
    # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim. Dyad's only change is swapping
    # upstream class names and log strings for the Dyad* versions (referencing an upstream class
    # would bypass Dyad's changes without raising); there is no logic of its own.
    @staticmethod
    @instrument(span_name="Overall Loading")
    def make_async_mp_client(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncMPClient":
        parallel_config = vllm_config.parallel_config
        client_args = (
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )
        if parallel_config.data_parallel_size > 1:
            if parallel_config.data_parallel_external_lb:
                # External load balancer - client per DP rank.
                return DPAsyncMPClient(*client_args)
            # Internal load balancer - client balances to all DP ranks.
            return DPLBAsyncMPClient(*client_args)
        return AsyncMPClient(*client_args)

    @abstractmethod
        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def shutdown(self, timeout: float | None = None) -> None: ...

    def get_output(self) -> EngineCoreOutputs:
        raise NotImplementedError

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        raise NotImplementedError

    def add_request(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
            raise NotImplementedError

    def reset_mm_cache(self) -> None:
        raise NotImplementedError

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def reset_prefix_cache(
            self, reset_running_requests: bool = False, reset_connector: bool = False
        ) -> bool:
            raise NotImplementedError

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
            raise NotImplementedError

    def wake_up(self, tags: list[str] | None = None) -> None:
        raise NotImplementedError

    def is_sleeping(self) -> bool:
        raise NotImplementedError

    def execute_dummy_batch(self) -> None:
        raise NotImplementedError

    async def execute_dummy_batch_async(self) -> None:
        raise NotImplementedError

    def abort_requests(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        raise NotImplementedError

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        raise NotImplementedError

    def dp_engines_running(self) -> bool:
        """Returns True id data parallel engines are collectively in a
        running state."""
        raise NotImplementedError

    async def scale_elastic_ep(self, new_data_parallel_size: int) -> None:
        raise NotImplementedError

    async def get_output_async(self) -> DyadEngineCoreOutputs:
        raise NotImplementedError

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
        raise NotImplementedError

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    async def profile_async(
            self, is_start: bool = True, profile_prefix: str | None = None
        ) -> None:
            raise NotImplementedError

    async def reset_mm_cache_async(self) -> None:
        raise NotImplementedError

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    async def reset_prefix_cache_async(
            self, reset_running_requests: bool = False, reset_connector: bool = False
        ) -> bool:
            raise NotImplementedError

        # DYAD-REBASED(vllm0.24): this method is vllm 0.24 verbatim, not an Dyad change.
    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:
            raise NotImplementedError

    async def wake_up_async(self, tags: list[str] | None = None) -> None:
        raise NotImplementedError

    async def is_sleeping_async(self) -> bool:
        raise NotImplementedError

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    async def remove_lora_async(self, lora_id: int) -> bool:
        raise NotImplementedError

    async def list_loras_async(self) -> set[int]:
        raise NotImplementedError

    async def pin_lora_async(self, lora_id: int) -> bool:
        raise NotImplementedError

    async def save_sharded_state_async(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        raise NotImplementedError

    async def collective_rpc_async(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        raise NotImplementedError

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried in.
    def reset_encoder_cache(self) -> None:
            raise NotImplementedError

    # DYAD-NOTE(vllm0.24): new in 0.24. The fork is a standalone copy and inherits nothing, so the
    # upstream text is carried in.
    async def reset_encoder_cache_async(self) -> None:
            raise NotImplementedError


class DyadInprocClient(DyadEngineCoreClient):
    """
    InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """

    def __init__(self, *args, **kwargs):
        self.engine_core = DyadEngineCore(*args, **kwargs)

    def get_output(self) -> EngineCoreOutputs:
        outputs, _ = self.engine_core.step_fn()
        return outputs and outputs.get(0) or EngineCoreOutputs()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.engine_core.get_supported_tasks()

    def add_request(self, request: EngineCoreRequest) -> None:
        req, request_wave = self.engine_core.preprocess_add_request(request)
        self.engine_core.add_request(req, request_wave)

    def abort_requests(self, request_ids: list[str]) -> None:
        if len(request_ids) > 0:
            self.engine_core.abort_requests(request_ids)

    # DYAD-NOTE(vllm0.24): 0.24 gave shutdown a `timeout` argument and passes it by keyword;
    # accept and forward it.
    def shutdown(self, timeout: float | None = None) -> None:
        self.engine_core.shutdown(timeout=timeout)

    # DYAD-NOTE(vllm0.24): follows 0.24's new signature (new arguments accepted at upstream's
    # defaults and forwarded).
    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        self.engine_core.profile(is_start, profile_prefix)

    def reset_mm_cache(self) -> None:
        self.engine_core.reset_mm_cache()

    # DYAD-NOTE(vllm0.24): follows 0.24's new signature (new arguments accepted at upstream's
    # defaults and forwarded).
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.engine_core.reset_prefix_cache(reset_running_requests, reset_connector)

    # DYAD-NOTE(vllm0.24): follows 0.24's new signature (new arguments accepted at upstream's
    # defaults and forwarded).
    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        self.engine_core.sleep(level, mode)

    def wake_up(self, tags: list[str] | None = None) -> None:
        self.engine_core.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def execute_dummy_batch(self) -> None:
        self.engine_core.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.engine_core.pin_lora(lora_id)

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        self.engine_core.save_sharded_state(path, pattern, max_size)

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def dp_engines_running(self) -> bool:
        return False


@dataclass
class BackgroundResources:
    """Used as a finalizer for clean shutdown, avoiding
    circular reference back to the client object."""

    ctx: zmq.Context
    # If CoreEngineProcManager, it manages local engines;
    # if CoreEngineActorManager, it manages all engines.
    engine_manager: CoreEngineProcManager | CoreEngineActorManager | None = None
    coordinator: DPCoordinator | None = None
    output_socket: zmq.Socket | zmq.asyncio.Socket | None = None
    input_socket: zmq.Socket | zmq.asyncio.Socket | None = None
    first_req_send_socket: zmq.asyncio.Socket | None = None
    first_req_rcv_socket: zmq.asyncio.Socket | None = None
    stats_update_socket: zmq.asyncio.Socket | None = None
    output_queue_task: asyncio.Task | None = None
    stats_update_task: asyncio.Task | None = None
    shutdown_path: str | None = None

    # Set if any of the engines are dead. Here so that the output
    # processing threads can access it without holding a ref to the client.
    engine_dead: bool = False

    def __call__(self):
        """Clean up background resources."""

        self.engine_dead = True
        if self.engine_manager is not None:
            self.engine_manager.close()
        if self.coordinator is not None:
            self.coordinator.close()

        if isinstance(self.output_socket, zmq.asyncio.Socket):
            # Async case.
            loop = self.output_queue_task._loop if self.output_queue_task else None

            sockets = (
                self.output_socket,
                self.input_socket,
                self.first_req_send_socket,
                self.first_req_rcv_socket,
                self.stats_update_socket,
            )

            tasks = (self.output_queue_task, self.stats_update_task)

            def close_sockets_and_tasks():
                close_sockets(sockets)
                for task in tasks:
                    if task is not None and not task.done():
                        with contextlib.suppress(Exception):
                            task.cancel()

            if loop is not None:
                if in_loop(loop):
                    close_sockets_and_tasks()
                elif not loop.is_closed():
                    loop.call_soon_threadsafe(close_sockets_and_tasks)
            else:
                # Loop has been closed, try to clean up directly.
                del tasks
                del close_sockets_and_tasks
                close_sockets(sockets)
                del self.output_queue_task
                del self.stats_update_task
        else:
            # Sync case.

            # ZMQ context termination can hang if the sockets
            # aren't explicitly closed first.
            close_sockets((self.output_socket, self.input_socket))

            if self.shutdown_path is not None:
                # We must ensure that the sync output socket is
                # closed cleanly in its own thread.
                with self.ctx.socket(zmq.PAIR) as shutdown_sender:
                    shutdown_sender.connect(self.shutdown_path)
                    # Send shutdown signal.
                    shutdown_sender.send(b"")

    def validate_alive(self, frames: Sequence[zmq.Frame]):
        if len(frames) == 1 and (frames[0].buffer == DyadEngineCoreProc.ENGINE_CORE_DEAD):
            self.engine_dead = True
            raise EngineDeadError()


class DyadMPClient(DyadEngineCoreClient):
    """
    MPClient: base class of the client for a multi-process EngineCore.

    The EngineCore behind it runs in a background process and keeps spinning in a busy loop:
    - receive new EngineCoreRequests
    - run scheduling / inference
    - produce EngineCoreOutputs

    The responsibilities of this client are:
    - send requests to the background EngineCore through input_socket
    - pull outputs back from the background EngineCore through output_socket

    Subclasses:
    - AsyncMPClient: used by AsyncLLM
    - SyncMPClient: used by the synchronous LLM
    """

    def __init__(
        self,
        asyncio_mode: bool,                  # whether to use the asyncio flavour of socket/context
        vllm_config: VllmConfig,             # global vLLM configuration
        executor_class: type[Executor],      # executor type used inside the background engine process
        log_stats: bool,                     # whether to record statistics
        client_addresses: dict[str, str] | None = None,  # if given, the engine was started externally
    ):
        # Keep the config around, many places below need it
        self.vllm_config = vllm_config

        # ---------------------------
        # serialization / deserialization components
        # ---------------------------

        # client -> engine: Python objects must be encoded into byte frames when sending requests
        self.encoder = MsgpackEncoder()

        # engine -> client: byte frames must be decoded back into EngineCoreOutputs when returning outputs
        self.decoder = MsgpackDecoder(DyadEngineCoreOutputs)

        # ---------------------------
        # ZMQ context initialization
        # ---------------------------

        # Create a synchronous zmq Context; io_threads=2 is the number of ZMQ's own internal IO threads
        # This thread count is not about Python business threads, but the threads ZMQ uses for low-level IO.
        sync_ctx = zmq.Context(io_threads=2)

        # In asyncio mode, wrap another asyncio context around the synchronous context,
        # so that the resulting socket supports async usage such as await socket.recv_multipart().
        # Otherwise the synchronous context is used directly.
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx

        # ---------------------------
        # resource cleaner initialization
        # ---------------------------

        # BackgroundResources is a resource container:
        # - manages ctx / sockets / engine_manager / coordinator / tasks, etc.
        # - closes all of them together when the client is destroyed
        #
        # Note that sync_ctx is passed here, not self.ctx,
        # because the synchronous context is what really owns/manages the underlying resources.
        self.resources = BackgroundResources(ctx=sync_ctx)

        # Create a finalizer:
        # when the MPClient object is garbage collected, self.resources() is called automatically to clean up.
        #
        # This way, even if __init__ raises halfway through, the already created sockets / managers
        # are still cleaned up as much as possible.
        self._finalizer = weakref.finalize(self, self.resources)

        # success=False means initialization has not finished successfully yet;
        # if it fails halfway, the finally block runs the cleanup logic automatically.
        success = False

        try:
            # ---------------------------
            # data parallel runtime state
            # ---------------------------

            # In DP mode, record whether the engines are currently running.
            # This value means little outside DP, but is kept uniformly anyway.
            self.engines_running = False

            # stats_update_address is the address on which the frontend receives DP stats broadcasts.
            # It may be unused in the plain single-engine mode, so it starts as None.
            self.stats_update_address: str | None = None

            # ---------------------------
            # Decide: connect to an "externally existing engine", or start the engine ourselves
            # ---------------------------

            if client_addresses:
                # Case 1: the engine was already started outside of the client
                # Here the client is only responsible for "connecting to these addresses"

                input_address = client_addresses["input_address"]

                output_address = client_addresses["output_address"]

                # Optional: address on which stats updates are received
                self.stats_update_address = client_addresses.get("stats_update_address")

            else:
                # Case 2: the engine is started by the current client itself

                # launch_core_engines(...) is a context manager;
                # it starts the background EngineCore processes / actors and returns:
                # - engine_manager: the object managing those engines
                # - coordinator: the DP coordinator (if any)
                # - addresses: the various ZMQ addresses
                with launch_core_engines(vllm_config, executor_class, log_stats) as (
                    engine_manager,
                    coordinator,
                    addresses,
                ):
                    # Keep the coordinator, it must be closed during cleanup later
                    self.resources.coordinator = coordinator

                    # Keep the engine manager, it must be closed later too
                    self.resources.engine_manager = engine_manager

                # addresses.inputs / outputs look like a tuple/list here,
                # but this MPClient only takes a single frontend input/output address.
                (input_address,) = addresses.inputs
                (output_address,) = addresses.outputs

                # The publish address on which the coordinator broadcasts stats to the frontend
                self.stats_update_address = addresses.frontend_stats_publish_address

                # If a coordinator exists, the stats address it reports itself
                # must match the address given in addresses.
                if coordinator is not None:
                    assert self.stats_update_address == (
                        coordinator.get_stats_publish_address()
                    )

            # ---------------------------
            # create the input / output sockets
            # ---------------------------

            # input_socket:
            # the frontend client uses it to send requests to the background engine.
            #
            # Why ROUTER:
            # - ROUTER can tell apart the identity of each engine
            # - the client can target a specific engine when sending a request
            #
            # bind=True means the frontend "listens/binds" on this address,
            # and the engine side connects to it.
            self.input_socket = self.resources.input_socket = make_zmq_socket(
                self.ctx, input_address, zmq.ROUTER, bind=True
            )

            # output_socket:
            # the frontend client uses it to receive the outputs sent back by the background engine.
            #
            # PULL is used, which implies the background engine most likely PUSHes to it.
            # There is no explicit bind here, so make_zmq_socket's default rule applies.
            self.resources.output_socket = make_zmq_socket(
                self.ctx, output_address, zmq.PULL
            )

            # ---------------------------
            # parse the data parallel configuration
            # ---------------------------

            parallel_config = vllm_config.parallel_config

            # Global DP size, e.g. 8 means there are 8 DP ranks in total
            dp_size = parallel_config.data_parallel_size

            # The starting DP rank of the current frontend/client
            dp_rank = parallel_config.data_parallel_rank

            # The number of DP engines that can be managed "locally"
            # For example under hybrid/external LB this client may only handle the few ranks on this machine
            dp_local_size = parallel_config.data_parallel_size_local

            # How offline_mode is decided:
            # if data_parallel_rank_local is not None, the local rank concept is already explicit in
            # the current runtime environment, which usually means the client targets one fixed local rank.
            offline_mode = parallel_config.data_parallel_rank_local is not None

            # local_engines_only:
            # whether this client manages only the local engines instead of all engines globally.
            #
            # It is True in two cases:
            # 1) hybrid LB: part of the load balancing happens locally
            # 2) external LB: an external load balancer already split traffic across DP ranks
            local_engines_only = (
                parallel_config.data_parallel_hybrid_lb
                or parallel_config.data_parallel_external_lb
            )

            # How many ranks to manage:
            # - if only local engines are managed, the count is dp_local_size
            # - otherwise all global DP ranks are managed, i.e. dp_size
            num_ranks = dp_local_size if local_engines_only else dp_size

            # engine_ranks_managed:
            # the list of engine ranks this client is actually responsible for communicating with
            #
            # Two cases:
            # 1) offline_mode: only one fixed rank is managed, i.e. [dp_rank]
            # 2) otherwise: num_ranks consecutive ranks starting at dp_rank
            self.engine_ranks_managed = (
                [dp_rank] if offline_mode else list(range(dp_rank, dp_rank + num_ranks))
            )

            # Safety check:
            # the number of local engines must not exceed the number of actually managed ranks
            assert parallel_config.data_parallel_size_local <= len(
                self.engine_ranks_managed
            )

            # ---------------------------
            # generate the ZMQ identity of each engine
            # ---------------------------

            # A ROUTER socket needs to know the peer identity when sending messages to different peers.
            #
            # The convention here is:
            # - the identity of each engine = the two-byte little-endian representation of its rank
            #
            # For instance rank=3 -> b'\x03\x00'
            self.core_engines: list[EngineIdentity] = [
                rank.to_bytes(2, "little") for rank in self.engine_ranks_managed
            ]

            # ---------------------------
            # wait for each engine to send its "ready" message
            # ---------------------------

            # identities is the set of engine identities that are not ready yet
            identities = set(self.core_engines)

            # Make a synchronous shadow socket for input_socket:
            # this way, even when self.input_socket is an asyncio socket, we can poll/recv synchronously here.
            #
            # The purpose:
            # during __init__, block and wait directly until every engine has started, which keeps the logic simpler.
            sync_input_socket = zmq.Socket.shadow(self.input_socket)

            # Keep waiting as long as some engine has not sent its ready message
            while identities:
                # Wait at most 600_000 ms = 600 seconds = 10 minutes
                if not sync_input_socket.poll(timeout=600_000):
                    raise TimeoutError(
                        "Timed out waiting for engines to send"
                        "initial message on input socket."
                    )

                # When a ROUTER receives a message, it first gets the peer identity, then the message body
                identity, _ = sync_input_socket.recv_multipart()

                identities.remove(identity)

            # ---------------------------
            # default core_engine
            # ---------------------------

            # By default the first engine is chosen as the "base target engine".
            # In single-engine mode it is naturally that one;
            # in multi-engine mode, subclasses can override the scheduling logic as needed.
            self.core_engine: EngineIdentity = self.core_engines[0]

            # ---------------------------
            # utility call result table
            # ---------------------------

            # utility_results: call_id -> future
            #
            # When the client issues a utility RPC (e.g. get_supported_tasks / add_lora / sleep),
            # it generates a call_id and stores the future here.
            # Once the background engine sends back the UtilityOutput, the future is looked up by call_id
            # and filled with the result.
            self.utility_results: dict[int, AnyFuture] = {}

            # ---------------------------
            # pending_messages: keep references to objects that are still being sent
            # ---------------------------

            # Some requests may carry the underlying buffer of a pytorch tensor.
            # With ZMQ zero-copy / asynchronous sends, the underlying buffer may not be fully sent yet.
            #
            # If Python releases the original object too early, the underlying buffer can become invalid.
            # So the following must be kept additionally:
            # - MessageTracker: used to decide whether this message finished sending
            # - the original object: keep the reference alive until tracker.done
            self.pending_messages = deque[tuple[zmq.MessageTracker, Any]]()

            # ---------------------------
            # start the death monitor for the background engine processes
            # ---------------------------

            # If some engine core process dies unexpectedly,
            # this monitor thread sets resources.engine_dead to True,
            # and makes subsequent client calls raise EngineDeadError.
            self.start_engine_core_monitor()

            # Reaching here means initialization completed successfully
            success = True

        finally:
            # If any step above failed:
            # - success is still False
            # - call the finalizer to clean up the already created resources
            if not success:
                self._finalizer()

    # DYAD-NOTE(vllm0.24): as above, accept `timeout`. Cleanup here is done by a finalizer so the
    # argument has nowhere to go, but the signature must still match or the call raises
    # TypeError: shutdown() got an unexpected keyword argument 'timeout'.
    def shutdown(self, timeout: float | None = None) -> None:
        # Terminate background resources.
        self._finalizer()

    def _format_exception(self, e: Exception) -> Exception:
        """If errored, use EngineDeadError so root cause is clear."""
        return (
            EngineDeadError(suppress_context=True) if self.resources.engine_dead else e
        )

    def ensure_alive(self):
        if self.resources.engine_dead:
            raise EngineDeadError()

    def add_pending_message(self, tracker: zmq.MessageTracker, msg: Any):
        if not tracker.done:
            self.pending_messages.appendleft((tracker, msg))

    def free_pending_messages(self):
        while self.pending_messages and self.pending_messages[-1][0].done:
            self.pending_messages.pop()

    def dp_engines_running(self) -> bool:
        return self.engines_running

    def start_engine_core_monitor(self):
        """Start a monitor thread for engine core processes."""
        engine_manager = self.resources.engine_manager
        if (
            engine_manager is None
            or not hasattr(engine_manager, "processes")
            or not engine_manager.processes
        ):
            # No engine processes to monitor
            return

        engine_processes = engine_manager.processes
        self_ref = weakref.ref(self)

        # Monitor engine core process liveness. If any die unexpectedly,
        # logs an error, shuts down the client and invokes the failure
        # callback to inform the engine.
        def monitor_engine_cores():
            sentinels = [proc.sentinel for proc in engine_processes]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or _self.resources.engine_dead:
                return
            _self.resources.engine_dead = True
            proc_name = next(
                proc.name for proc in engine_processes if proc.sentinel == died[0]
            )
            logger.error(
                "Engine core proc %s died unexpectedly, shutting down client.",
                proc_name,
            )
            _self.shutdown()
            # Note: For MPClient, we don't have a failure callback mechanism
            # like MultiprocExecutor, but we set engine_dead flag which will
            # cause subsequent operations to raise EngineDeadError

        Thread(
            target=monitor_engine_cores, daemon=True, name="MPClientEngineMonitor"
        ).start()


def _process_utility_output(
    output: UtilityOutput, utility_results: dict[int, AnyFuture]
):
    """Set the result from a utility method in the waiting future."""
    future = utility_results.pop(output.call_id)
    failure_message = output.failure_message
    try:
        if failure_message is not None:
            future.set_exception(Exception(failure_message))
        else:
            assert output.result is not None
            future.set_result(output.result.result)
    except asyncio.InvalidStateError:
        # This can happen if the future is cancelled due to the
        # original calling task being cancelled.
        if failure_message is not None:
            logger.error(
                "Cancelled call to utility method failed with error: %s",
                failure_message,
            )


class SyncMPClient(DyadMPClient):
    """Synchronous client for multi-proc EngineCore."""

    def __init__(
        self, vllm_config: VllmConfig, executor_class: type[Executor], log_stats: bool
    ):
        super().__init__(
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )
        self.is_dp = self.vllm_config.parallel_config.data_parallel_size > 1
        self.outputs_queue = queue.Queue[EngineCoreOutputs | Exception]()

        # Ensure that the outputs socket processing thread does not have
        # a ref to the client which prevents gc.
        ctx = self.ctx
        out_socket = self.resources.output_socket
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue

        shutdown_path = get_open_zmq_inproc_path()
        resources = self.resources
        resources.shutdown_path = shutdown_path

        def process_outputs_socket():
            assert isinstance(out_socket, zmq.Socket)
            shutdown_socket = ctx.socket(zmq.PAIR)
            try:
                shutdown_socket.bind(shutdown_path)
                poller = zmq.Poller()
                poller.register(shutdown_socket, zmq.POLLIN)
                poller.register(out_socket, zmq.POLLIN)
                while True:
                    socks = poller.poll()
                    if not socks:
                        continue
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        # shutdown signal, exit thread.
                        break

                    frames = out_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        _process_utility_output(outputs.utility_output, utility_results)
                    else:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            finally:
                # Close sockets.
                shutdown_socket.close(linger=0)
                out_socket.close(linger=0)

        # Process outputs from engine in separate thread.
        self.output_queue_thread = Thread(
            target=process_outputs_socket,
            name="EngineCoreOutputQueueThread",
            daemon=True,
        )
        self.output_queue_thread.start()

        # The thread takes on responsibility for closing the socket.
        self.resources.output_socket = None

    def get_output(self) -> EngineCoreOutputs:
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        outputs = self.outputs_queue.get()
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        if outputs.wave_complete is not None:
            self.engines_running = False
        return outputs

    def _send_input(self, request_type: EngineCoreRequestType, request: Any):
        """
        Send one input message to the EngineCore.

        Arguments:
        - request_type:
            the request type, describing what operation this message is.
            For example it may be add_request / abort_request / profile / shutdown, etc.
        - request:
            the concrete request object to send; its type is unconstrained and the encoder
            is responsible for serializing it.

        The core flow of this function:
        1. confirm the backend engine is still alive
        2. release the previously completed "messages awaiting confirmation"
        3. encode the current request into a ZMQ multipart message
        4. send the message
        5. if the message contains extra backing buffers (such as the memory blocks of a tensor),
           track the send-completion event to prevent the related objects from being released too early
        """

        # ---------------------------------------------------------
        # 1) Make sure the backend engine of this client is still alive
        # ---------------------------------------------------------
        # If the engine process already died, the connection broke, or the client state is abnormal,
        # this usually raises instead of proceeding to send.
        self.ensure_alive()

        # ---------------------------------------------------------
        # 2) Release the previously tracked messages that already finished sending
        # ---------------------------------------------------------
        # Some messages are sent with track=True,
        # so that ZMQ returns a tracker used to confirm whether the low-level send really completed.
        #
        # Before those messages finish sending, the reference to the original request may need to be kept,
        # otherwise the underlying buffer / memoryview / tensor backing memory the request depends on
        # could be reclaimed early by Python.
        #
        # free_pending_messages() therefore typically:
        # - checks which trackers have completed
        # - releases the pending references held for those messages
        self.free_pending_messages()

        # ZMQ frames contain the engine identity, request type and serialized body, followed
        # by any auxiliary zero-copy buffers returned by the encoder.
        msg = (self.core_engine, request_type.value, *self.encoder.encode(request))

        # Three or fewer frames have no auxiliary buffers requiring lifetime tracking.
        if len(msg) <= 3:
            # copy=False means avoiding extra copies as much as possible,
            # letting ZMQ reference the existing buffer directly for better performance.
            self.input_socket.send_multipart(msg, copy=False)
            return

        # ---------------------------------------------------------
        # 5) If the message length is > 3, extra buffers are attached
        # ---------------------------------------------------------
        # This usually means the request is not just an ordinary small-object serialization,
        # but also carries some large memory blocks (e.g. tensor / ndarray / memoryview).
        #
        # In that case, if the reference to request is dropped right after send,
        # the underlying buffer may not have been fully sent yet while the memory is already
        # released/reused, which causes problems.
        #
        # So track=True is used here, making ZMQ return a MessageTracker.
        tracker = self.input_socket.send_multipart(msg, copy=False, track=True)

        # ---------------------------------------------------------
        # 6) Record the tracker and the request as a "pending message"
        # ---------------------------------------------------------
        # The client will then, at some later point (via free_pending_messages),
        # check whether the tracker has completed.
        #
        # Until the tracker completes, the reference to request is retained,
        # to ensure the backing buffers associated with request are not released too early.
        self.add_pending_message(tracker, request)

    def call_utility(self, method: str, *args) -> Any:
        call_id = uuid.uuid1().int >> 64
        future: Future[Any] = Future()
        self.utility_results[call_id] = future
        self._send_input(EngineCoreRequestType.UTILITY, (0, call_id, method, args))

        return future.result()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.call_utility("get_supported_tasks")

    def add_request(self, request: EngineCoreRequest) -> None:
        if self.is_dp:
            self.engines_running = True
        self._send_input(EngineCoreRequestType.ADD, request)

    def abort_requests(self, request_ids: list[str]) -> None:
        if request_ids and not self.resources.engine_dead:
            self._send_input(EngineCoreRequestType.ABORT, request_ids)

    def profile(self, is_start: bool = True) -> None:
        self.call_utility("profile", is_start)

    def reset_mm_cache(self) -> None:
        self.call_utility("reset_mm_cache")

    def reset_prefix_cache(self, reset_running_requests: bool = False) -> bool:
        return self.call_utility("reset_prefix_cache", reset_running_requests)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.call_utility("add_lora", lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.call_utility("remove_lora", lora_id)

    def list_loras(self) -> set[int]:
        return self.call_utility("list_loras")

    def pin_lora(self, lora_id: int) -> bool:
        return self.call_utility("pin_lora", lora_id)

    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        # DYAD-NOTE(vllm0.24): LLM.sleep unconditionally forwards the upstream mode argument.
        self.call_utility("sleep", level, mode)

    def wake_up(self, tags: list[str] | None = None) -> None:
        self.call_utility("wake_up", tags)

    def is_sleeping(self) -> bool:
        return self.call_utility("is_sleeping")

    def execute_dummy_batch(self) -> None:
        self.call_utility("execute_dummy_batch")

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.call_utility("collective_rpc", method, timeout, args, kwargs)

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        self.call_utility("save_sharded_state", path, pattern, max_size)


class AsyncMPClient(DyadMPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        super().__init__(
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            client_addresses=client_addresses,
        )

        self.client_count = client_count
        self.client_index = client_index
        self.outputs_queue = asyncio.Queue[DyadEngineCoreOutputs | Exception]()
        try:
            # If we are running in an asyncio event loop, start the queue task.
            # Otherwise, it will be started lazily. If it is not started here,
            # we could miss EXECUTOR_FAILED messages from engine core if they
            # occur prior to any requests being sent.
            asyncio.get_running_loop()
            self._ensure_output_queue_task()
        except RuntimeError:
            pass

    def _ensure_output_queue_task(self):
        resources = self.resources
        if resources.output_queue_task is not None:
            return

        # Perform IO in separate task to parallelize as much as possible.
        # Avoid task having direct reference back to the client.
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue
        output_handler: (
            Callable[[AsyncMPClient, DyadEngineCoreOutputs], Awaitable[None]] | None
        ) = getattr(self.__class__, "process_engine_outputs", None)
        _self_ref = weakref.ref(self) if output_handler else None
        output_socket = resources.output_socket
        assert output_socket is not None

        async def process_outputs_socket():
            try:
                while True:
                    frames = await output_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: DyadEngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        _process_utility_output(outputs.utility_output, utility_results)
                        continue

                    if output_handler is not None:
                        assert _self_ref is not None
                        _self = _self_ref()
                        if not _self:
                            # Client has been garbage collected, abort.
                            return
                        await output_handler(_self, outputs)

                    if outputs.outputs or outputs.scheduler_stats:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            except asyncio.CancelledError:
                outputs_queue.put_nowait(EngineDeadError())

        resources.output_queue_task = asyncio.create_task(
            process_outputs_socket(), name="EngineCoreOutputQueueTask"
        )

    async def get_output_async(self) -> DyadEngineCoreOutputs:
        self._ensure_output_queue_task()
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        assert self.outputs_queue is not None
        outputs = await self.outputs_queue.get()
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        return outputs

    def _send_input(
        self,
        request_type: EngineCoreRequestType,
        request: Any,
        engine: EngineIdentity | None = None,
    ) -> Awaitable[Any]:
        if engine is None:
            engine = self.core_engine

        message = (request_type.value, *self.encoder.encode(request))
        return self._send_input_message(message, engine, request)

    def _send_input_message(
        self, message: tuple[bytestr, ...], engine: EngineIdentity, objects: Any
    ) -> Awaitable[Any]:
        """
        objects is a reference to retain until zmq is finished with the
        buffers, in case they were extracted from tensors in the request.
        """
        self.ensure_alive()
        self.free_pending_messages()

        msg = (engine,) + message
        if not objects or len(msg) <= 3:
            # No auxiliary buffers => no tensor backing buffers in request.
            return self.input_socket.send_multipart(msg, copy=False)

        future: asyncio.Future[zmq.MessageTracker]
        future = self.input_socket.send_multipart(msg, copy=False, track=True)

        def add_pending(f: asyncio.Future[zmq.MessageTracker]):
            with contextlib.suppress(BaseException):
                self.add_pending_message(f.result(), objects)

        future.add_done_callback(add_pending)
        return future

    async def call_utility_async(self, method: str, *args) -> Any:
        return await self._call_utility_async(method, *args, engine=self.core_engine)

    async def _call_utility_async(
        self, method: str, *args, engine: EngineIdentity
    ) -> Any:
        call_id = uuid.uuid1().int >> 64
        future = asyncio.get_running_loop().create_future()
        self.utility_results[call_id] = future
        message = (
            EngineCoreRequestType.UTILITY.value,
            *self.encoder.encode((self.client_index, call_id, method, args)),
        )
        await self._send_input_message(message, engine, args)
        self._ensure_output_queue_task()
        return await future

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
        return await self.call_utility_async("get_supported_tasks")

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        request.client_index = self.client_index
        await self._send_input(EngineCoreRequestType.ADD, request)
        self._ensure_output_queue_task()

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if request_ids and not self.resources.engine_dead:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)

    async def profile_async(self, is_start: bool = True) -> None:
        await self.call_utility_async("profile", is_start)

    async def reset_mm_cache_async(self) -> None:
        await self.call_utility_async("reset_mm_cache")

    # DYAD-NOTE(vllm0.24): 0.24 gave reset_prefix_cache a second argument, reset_connector. The
    # abstract base picked up the new signature along with the pure-copy sync, but this concrete
    # implementation was missed, so callers pass 2 arguments and the implementation takes 1:
    # TypeError: reset_prefix_cache_async() takes from 1 to 2 positional arguments but 3 were given.
    async def reset_prefix_cache_async(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return await self.call_utility_async(
            "reset_prefix_cache", reset_running_requests, reset_connector
        )

    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:
        # AsyncLLM.sleep likewise forwards both level and mode.
        await self.call_utility_async("sleep", level, mode)

    async def wake_up_async(self, tags: list[str] | None = None) -> None:
        await self.call_utility_async("wake_up", tags)

    async def is_sleeping_async(self) -> bool:
        return await self.call_utility_async("is_sleeping")

    async def execute_dummy_batch_async(self) -> None:
        await self.call_utility_async("execute_dummy_batch")

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        return await self.call_utility_async("add_lora", lora_request)

    async def remove_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("remove_lora", lora_id)

    async def list_loras_async(self) -> set[int]:
        return await self.call_utility_async("list_loras")

    async def pin_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("pin_lora", lora_id)

    async def save_sharded_state_async(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        await self.call_utility_async("save_sharded_state", path, pattern, max_size)

    async def collective_rpc_async(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return await self.call_utility_async(
            "collective_rpc", method, timeout, args, kwargs
        )


class DPAsyncMPClient(AsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Assumes external load-balancing by default."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        self.current_wave = 0

        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )

        # List of [waiting, running] pair per engine.
        # Used only by DPLBAsyncMPClient subclass.
        self.lb_engines: list[list[int]] = [[0, 0] for _ in self.core_engines]

        self.first_req_sock_addr = get_open_zmq_inproc_path()
        self.first_req_send_socket = self.resources.first_req_send_socket = (
            make_zmq_socket(self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=True)
        )
        try:
            # If we are running in an asyncio event loop, start the stats task.
            # Otherwise, it will be started lazily.
            asyncio.get_running_loop()
            self._ensure_stats_update_task()
        except RuntimeError:
            pass

    def _ensure_stats_update_task(self):
        resources = self.resources
        if resources.stats_update_task is not None:
            return

        assert self.stats_update_address is not None
        stats_addr: str = self.stats_update_address
        assert len(self.engine_ranks_managed) > 0
        # NOTE: running and waiting counts are all global from
        # the Coordinator include all global EngineCores. This
        # slice includes just the cores managed by this client.
        count_slice = slice(
            self.engine_ranks_managed[0], self.engine_ranks_managed[-1] + 1
        )

        async def run_engine_stats_update_task():
            with (
                make_zmq_socket(self.ctx, stats_addr, zmq.XSUB, linger=0) as socket,
                make_zmq_socket(
                    self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=False, linger=0
                ) as first_req_rcv_socket,
            ):
                assert isinstance(socket, zmq.asyncio.Socket)
                assert isinstance(first_req_rcv_socket, zmq.asyncio.Socket)
                self.resources.stats_update_socket = socket
                self.resources.first_req_rcv_socket = first_req_rcv_socket
                # Send subscription message.
                await socket.send(b"\x01")

                poller = zmq.asyncio.Poller()
                poller.register(socket, zmq.POLLIN)
                poller.register(first_req_rcv_socket, zmq.POLLIN)

                while True:
                    events = await poller.poll()
                    if (
                        not self.engines_running
                        and len(events) == 2
                        or (events[0][0] == first_req_rcv_socket)
                    ):
                        # Check if this is a regular request notification or
                        # scale up notification
                        buf = first_req_rcv_socket.recv(flags=zmq.NOBLOCK).result()

                        decoded = msgspec.msgpack.decode(buf)
                        if (
                            isinstance(decoded, (list, tuple))
                            and len(decoded) == 2
                            and decoded[0] == "SCALE_ELASTIC_EP"
                        ):
                            # Extract new engine count from the decoded message
                            new_engine_count = decoded[1]
                            # Send scale up notification to coordinator
                            scale_msg = msgspec.msgpack.encode(
                                ("SCALE_ELASTIC_EP", new_engine_count)
                            )
                            await socket.send(scale_msg)
                            continue

                        # we're sending a request while the engines are
                        # paused, so that it can wake the others up
                        # (to run dummy EP loop).
                        assert decoded[0] == "FIRST_REQ"
                        target_eng_index = decoded[1]
                        self.engines_running = True
                        msg = msgspec.msgpack.encode(
                            (target_eng_index, self.current_wave)
                        )
                        await socket.send(msg)

                    buf = None
                    while True:
                        # Drain all stats events (we only care about latest).
                        future: asyncio.Future[bytes] = socket.recv(flags=zmq.NOBLOCK)
                        if isinstance(future.exception(), zmq.Again):
                            break
                        buf = future.result()
                    if buf is None:
                        continue

                    # Update local load-balancing state.
                    counts, wave, running = msgspec.msgpack.decode(buf)
                    self.current_wave = wave
                    self.engines_running = running
                    if counts is not None:
                        sliced_counts = counts[count_slice]
                        self.lb_engines = sliced_counts
                        logger.debug(
                            "Received counts: %s (%s)", sliced_counts, count_slice
                        )

        resources.stats_update_task = asyncio.create_task(
            run_engine_stats_update_task()
        )

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        self._ensure_stats_update_task()

        request.current_wave = self.current_wave
        request.client_index = self.client_index

        chosen_engine = self.get_core_engine_for_request(request)
        to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
        if not self.engines_running:
            # Notify coordinator that we're sending a request
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))
            await self.first_req_send_socket.send(req_msg)

        await to_await

        self._ensure_output_queue_task()

    def get_core_engine_for_request(self, request: EngineCoreRequest):
        return self.core_engine


class DPLBAsyncMPClient(DPAsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Load-balances between multiple engine processes."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        self.client_count = client_count

        # To route aborts to the correct engine.
        self.reqs_in_flight: dict[str, EngineIdentity] = {}

        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )

        assert len(self.core_engines) > 1

        self.eng_start_index = (
            len(self.core_engines) * self.client_index
        ) // client_count

    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        # Engines are in rank order.
        if (eng_index := request.data_parallel_rank) is None:
            current_counts = self.lb_engines
            # TODO use P2C alg for larger DP sizes
            num_engines = len(current_counts)
            min_score = sys.maxsize
            eng_index = 0
            for i in range(num_engines):
                # Start from client_index to help with balancing when engines
                # are empty.
                idx = (self.eng_start_index + i) % num_engines
                waiting, running = current_counts[idx]
                score = waiting * 4 + running
                if score < min_score:
                    min_score = score
                    eng_index = idx
            # Increment local waiting count for better balancing between stats
            # updates from the coordinator (which happen every 100ms).
            current_counts[eng_index][0] += self.client_count

        chosen_engine = self.core_engines[eng_index]
        # Record which engine is chosen for this request, to handle aborts.
        self.reqs_in_flight[request.request_id] = chosen_engine
        return chosen_engine

    async def call_utility_async(self, method: str, *args) -> Any:
        # Only the result from the first engine is returned.
        return (
            await asyncio.gather(
                *[
                    self._call_utility_async(method, *args, engine=engine)
                    for engine in self.core_engines
                ]
            )
        )[0]

    @staticmethod
    async def process_engine_outputs(
        self: "DPLBAsyncMPClient", outputs: EngineCoreOutputs
    ):
        if outputs.finished_requests and self.reqs_in_flight:
            for req_id in outputs.finished_requests:
                self.reqs_in_flight.pop(req_id, None)

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if not request_ids or self.resources.engine_dead:
            return

        if len(request_ids) == 1:
            # Fast-path common case.
            if engine := self.reqs_in_flight.get(request_ids[0]):
                await self._abort_requests(request_ids, engine)
            return

        by_engine = defaultdict[EngineIdentity, list[str]](list)
        for req_id in request_ids:
            if engine := self.reqs_in_flight.get(req_id):
                by_engine[engine].append(req_id)
        for engine, req_ids in by_engine.items():
            await self._abort_requests(req_ids, engine)

    async def _abort_requests(
        self, request_ids: list[str], engine: EngineIdentity
    ) -> None:
        await self._send_input(EngineCoreRequestType.ABORT, request_ids, engine)

    async def scale_elastic_ep(self, new_data_parallel_size: int) -> None:
        """Scale elastic EP data parallel size"""
        cur_data_parallel_size = len(self.core_engines)

        assert new_data_parallel_size != cur_data_parallel_size, (
            f"new_data_parallel_size {new_data_parallel_size} must be "
            f"different from cur_data_parallel_size {cur_data_parallel_size}"
        )

        assert self.vllm_config.parallel_config.data_parallel_backend == "ray", (
            "Only ray DP backend supports scaling elastic EP"
        )

        scale_up = new_data_parallel_size > cur_data_parallel_size

        if scale_up:
            await self._scale_up_elastic_ep(
                cur_data_parallel_size, new_data_parallel_size
            )
        else:
            await self._scale_down_elastic_ep(
                cur_data_parallel_size, new_data_parallel_size
            )

    async def _scale_up_elastic_ep(
        self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None:
        """Scale up the data parallel size by creating new engine cores
        and reconfiguring existing ones."""
        cur_data_parallel_size = len(self.core_engines)

        # Phase 1: Send reconfigure messages to all existing engines and wait
        # for them to be sent
        reconfig_futures = []
        self.vllm_config.parallel_config.data_parallel_master_port = get_open_port()
        for engine in self.core_engines:
            reconfig_request = ReconfigureDistributedRequest(
                new_data_parallel_size=new_data_parallel_size,
                new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_master_ip=self.vllm_config.parallel_config.data_parallel_master_ip,
                new_data_parallel_master_port=self.vllm_config.parallel_config.data_parallel_master_port,
            )
            coro = self._call_utility_async(
                "reinitialize_distributed", reconfig_request, engine=engine
            )
            reconfig_futures.append(asyncio.create_task(coro))

        logger.info("All reconfigure messages sent, starting engine creation")

        # Phase 2: Create new engines now that reconfig messages have been sent
        # self.resources.engine_manager is guaranteed to be
        # CoreEngineActorManager for RayDPClient
        assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
        self.resources.engine_manager.scale_up_elastic_ep(
            self.vllm_config, new_data_parallel_size
        )

        # Create new CoreEngine objects for the new engines
        new_engine_identities = set()
        for i in range(cur_data_parallel_size, new_data_parallel_size):
            new_engine = i.to_bytes(2, "little")
            self.core_engines.append(new_engine)
            new_engine_identities.add(new_engine)

        # Wait for ready messages from new engines on the input socket
        sync_input_socket = zmq.Socket.shadow(self.input_socket)
        while new_engine_identities:
            if not sync_input_socket.poll(timeout=600_000):
                raise TimeoutError(
                    "Timed out waiting for new engines to send initial "
                    "message on input socket."
                )
            identity, _ = sync_input_socket.recv_multipart()
            new_engine_identities.discard(identity)

        # Phase 3: Wait for all existing engines to complete reconfiguration
        logger.info("Waiting for existing engines to complete reconfiguration")
        await asyncio.gather(*reconfig_futures)

        # Notify coordinator about scale up through existing
        # stats_update_task connection
        self._ensure_stats_update_task()
        scale_up_marker = msgspec.msgpack.encode(
            ("SCALE_ELASTIC_EP", new_data_parallel_size)
        )
        await self.first_req_send_socket.send(scale_up_marker)

        # Update the parallel config
        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        logger.info(
            "[Elastic EP] Scale up completed, new data parallel size: %s",
            new_data_parallel_size,
        )

    async def _scale_down_elastic_ep(
        self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None:
        """Scale down the data parallel size by shutting down and
        reconfiguring existing engine cores."""
        cur_data_parallel_size = len(self.core_engines)

        self.vllm_config.parallel_config.data_parallel_master_port = get_open_port()

        reconfig_futures = []
        for cur_dp_rank, engine in enumerate(self.core_engines):
            reconfig_request = ReconfigureDistributedRequest(
                new_data_parallel_size=new_data_parallel_size,
                new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_master_ip=self.vllm_config.parallel_config.data_parallel_master_ip,
                new_data_parallel_master_port=self.vllm_config.parallel_config.data_parallel_master_port,
            )
            if cur_dp_rank >= new_data_parallel_size:
                reconfig_request.new_data_parallel_rank = (
                    ReconfigureRankType.SHUTDOWN_CURRENT_RANK
                )
            coro = self._call_utility_async(
                "reinitialize_distributed", reconfig_request, engine=engine
            )
            reconfig_futures.append(asyncio.create_task(coro))

        for _ in range(new_data_parallel_size, cur_data_parallel_size):
            self.core_engines.pop()

        await asyncio.gather(*reconfig_futures)

        assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
        self.resources.engine_manager.scale_down_elastic_ep(
            cur_data_parallel_size, new_data_parallel_size
        )

        self._ensure_stats_update_task()
        scale_down_marker = msgspec.msgpack.encode(
            ("SCALE_ELASTIC_EP", new_data_parallel_size)
        )
        await self.first_req_send_socket.send(scale_down_marker)

        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        logger.info(
            "[Elastic EP] Scale down completed, new data parallel size: %s",
            new_data_parallel_size,
        )
