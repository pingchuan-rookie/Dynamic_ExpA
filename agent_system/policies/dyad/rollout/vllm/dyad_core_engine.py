# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing import Process, connection
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING
from unittest.mock import patch

import msgspec
import zmq

from vllm import envs
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.ray.ray_env import get_env_vars_to_copy
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.utils.system_utils import get_mp_context
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.executor import Executor
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

STARTUP_POLL_PERIOD_MS = 10000


class CoreEngineState(Enum):
    NEW = auto()
    CONNECTED = auto()
    READY = auto()


class CoreEngine:
    """One per data parallel rank, used to track state during handshaking."""

    def __init__(self, index: int = 0, local: bool = True):
        self.local = local
        self.identity = index.to_bytes(2, "little")

        self.state = CoreEngineState.NEW


@dataclass
class EngineZmqAddresses:
    # ZMQ input socket addresses for each front-end client (requests)
    inputs: list[str]
    # ZMQ output socket addresses for each front-end client (responses)
    outputs: list[str]
    # ZMQ input socket address of DP coordinator if applicable
    coordinator_input: str | None = None
    # ZMQ output socket address of DP coordinator if applicable
    coordinator_output: str | None = None
    # ZMQ socket for front-end to connect to DP coordinator.
    # Not used by engine, just relayed to front-end in handshake response.
    # Only required for external DP LB case.
    frontend_stats_publish_address: str | None = None


@dataclass
class EngineHandshakeMetadata:
    """Metadata sent to each engine process during startup handshake,
    including addresses of the front-end ZMQ queues that they should
    connect to.
    """

    addresses: EngineZmqAddresses
    parallel_config: dict[str, int | str | list[int]]
    parallel_config_hash: str | None = None


class CoreEngineProcManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of background processes used by the AsyncLLM and LLMEngine.
    """
    # Manage EngineCore process startup, failure cleanup and liveness for AsyncLLM/LLMEngine.


    def __init__(
        self,
        target_fn: Callable,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
    ):
        # Start local_engine_count processes running target_fn with the supplied vLLM executor.
        # start_index and local_start_index determine global and local DP ranks, respectively.
        # The engine handshake address and optional client handshake address serve separate peers.


        # Obtain the multiprocessing context.
        # This is an important step, because the process start method may differ
        # across platforms/configurations (spawn / fork / forkserver, etc.).
        context = get_mp_context()

        # Build the set of kwargs shared by all child processes.
        # These arguments are identical for every process, so they are hoisted into common_kwargs.
        common_kwargs = {
            "vllm_config": vllm_config,
            "local_client": local_client,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
        }

        # If client_handshake_address was given,
        # add it to the shared arguments of all child processes as well.
        if client_handshake_address:
            common_kwargs["client_handshake_address"] = client_handshake_address

        # self.processes:
        # holds every background process object that was created.
        self.processes: list[BaseProcess] = []

        # local_dp_ranks:
        # holds the local data-parallel rank corresponding to each local process.
        # It is used later at start time to configure devices/environment variables.
        local_dp_ranks = []

        # Create the background processes in a loop, according to local_engine_count
        for index in range(local_engine_count):
            # Compute the local index of this process
            local_index = local_start_index + index

            # Compute the global index of this process
            global_index = start_index + index

            # Start EngineCore in background process.
            #
            # What local_dp_ranks collects here is the local data-parallel rank;
            # device control environment variables may need to be set based on it before proc.start().
            local_dp_ranks.append(local_index)

            # Create a new background process object, not started yet at this point
            self.processes.append(
                context.Process(
                    # child process entry function
                    target=target_fn,

                    # name the process for easier logging/debugging
                    # e.g. EngineCore_DP0 / EngineCore_DP1 ...
                    name=f"DyadEngineCore_DP{global_index}",

                    # keyword arguments passed to the child process target_fn
                    kwargs=common_kwargs
                    | {
                        "dp_rank": global_index,       # global data parallel rank
                        "local_dp_rank": local_index,  # local data parallel rank
                    },
                )
            )

        # Register a finalizer:
        # when this CoreEngineProcManager object is destroyed,
        # shutdown(self.processes) is called automatically to clean up all background processes.
        #
        # The benefit of weakref.finalize is:
        # even if the caller forgets to call close() manually,
        # the processes still get wound down as long as this manager object is garbage collected.
        self._finalizer = weakref.finalize(self, shutdown, self.processes)

        # Decide whether data parallel is enabled
        data_parallel = vllm_config.parallel_config.data_parallel_size > 1

        try:
            # Iterate over all processes to start and their corresponding local_dp_rank
            for proc, local_dp_rank in zip(self.processes, local_dp_ranks):

                # DP workers on non-CUDA platforms or Ray need temporary device environment settings.
                # Plain CUDA multiprocessing selects its device directly.
                with (
                    set_device_control_env_var(vllm_config, local_dp_rank)
                    if (
                        data_parallel
                        and (
                            not current_platform.is_cuda_alike()
                            or vllm_config.parallel_config.use_ray
                        )
                    )
                    else contextlib.nullcontext()
                ):
                    # actually start the child process
                    proc.start()

        finally:
            # The point of finally is:
            # regardless of whether the startup flow raised, check whether some processes already finished.
            #
            # Typical problem:
            # - some processes started successfully
            # - others failed to start / exited immediately
            #
            # In that case the already started processes must not be left running;
            # they must all be cleaned up to avoid a half-broken state.
            if self.finished_procs():
                self.close()


    def close(self):
        """Shutdown all procs."""
        # Shut down all background processes.
        #
        # This does not hand-write terminate/join, but calls the finalizer registered above.
        # The finalizer calls shutdown(self.processes) internally.
        #
        # This keeps close() and the destructor cleanup logic identical.
        self._finalizer()


    def join_first(self):
        """Wait for any process to exit."""
        # Wait until "any one" process exits.
        #
        # proc.sentinel is a wait handle provided by multiprocessing:
        # when the corresponding process exits, that sentinel becomes ready.
        #
        # connection.wait(...) blocks until any one of the passed sentinels becomes ready.
        #
        # So the meaning of this function is:
        #   "block and wait until some background EngineCore process exits"
        #
        # This interface is commonly used for:
        # - monitoring whether a background process crashed early
        # - letting the main process notice the exit of any worker/core as soon as possible
        connection.wait(proc.sentinel for proc in self.processes)


    def sentinels(self) -> list:
        # Return the list of sentinels of all processes.
        #
        # A sentinel can be understood as:
        #   "the waitable exit-signal handle of this process"
        #
        # Higher layers may integrate these sentinels into their own event loop/wait logic.
        return [proc.sentinel for proc in self.processes]


    def finished_procs(self) -> dict[str, int]:
        """Returns dict of proc name -> exit code for any finished procs."""
        # Return all processes that "already finished", in the form:
        #   {process name: exit code}
        #
        # The semantics of proc.exitcode:
        # - None: the process has not finished yet
        # - an integer: the process finished, and that integer is the exit code
        #
        # So only processes with exitcode is not None are selected here.
        return {
            proc.name: proc.exitcode
            for proc in self.processes
            if proc.exitcode is not None
        }


@contextlib.contextmanager
def set_device_control_env_var(
    vllm_config: VllmConfig, local_dp_rank: int
) -> Iterator[None]:
    """
    Temporarily set CUDA_VISIBLE_DEVICES or equivalent
    for engine subprocess.
    """
    world_size = vllm_config.parallel_config.world_size
    local_world_size = vllm_config.parallel_config.local_world_size
    evar = current_platform.device_control_env_var

    value = get_device_indices(evar, local_dp_rank, world_size, local_world_size)
    with patch.dict(os.environ, values=((evar, value),)):
        yield


def get_device_indices(
    device_control_env_var: str,
    local_dp_rank: int,
    world_size: int,
    local_world_size: int | None = None,
):
    """
    Returns a comma-separated string of device indices for the specified
    data parallel rank.

    For example, if world_size=2 and local_dp_rank=1, and there are 4 devices,
    this will select devices 2 and 3 for local_dp_rank=1.
    """
    if local_world_size is None:
        local_world_size = world_size
    try:
        value = ",".join(
            str(current_platform.device_id_to_physical_device_id(i))
            for i in range(
                local_dp_rank * world_size,
                local_dp_rank * world_size + local_world_size,
            )
        )
    except IndexError as e:
        raise Exception(
            f"Error setting {device_control_env_var}: "
            f"local range: [{local_dp_rank * world_size}, "
            f"{(local_dp_rank + 1) * world_size}) "
            "base value: "
            f'"{os.getenv(device_control_env_var)}"'
        ) from e
    return value


class CoreEngineActorManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of core engine Ray actors used by the AsyncLLM and LLMEngine.

    Different from CoreEngineProcManager, this class manages
    core engines for both local and remote nodes.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        placement_groups: list["PlacementGroup"] | None = None,
        local_dp_ranks: list[int] | None = None,
    ):
        import copy

        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        # >>> DYAD-BEGIN(vllm0.24): import dyad's own DPEngineCoreActor instead
        # 0.24 deleted vllm.v1.engine.core.DPEngineCoreActor and folded its job into
        # DPEngineCoreProc (all that remains in 0.24's core.py is a process name of the form
        # "DPEngineCoreActor_DP{rank}"). dyad already forks a class of the same name in
        # dyad_core.py -- DPEngineCoreActor(DyadDPEngineCoreProc) -- so that is what is imported.
        #
        # WARNING: this change alters semantics and needs confirming in the equivalence tests.
        # On 0.12 this line imported **vllm's original**, so ray.remote() wrapped an actor with no
        # Dyad logic at all; dyad's DPEngineCoreActor inherits DyadDPEngineCoreProc and does have it.
        # In other words ExpA_sys was most likely not running the Dyad version on the DP path --
        # either deliberately, or as an import mistake that went unnoticed for a long time. On 0.24
        # there is no second option: only dyad's class exists. If the DP path is genuinely meant to
        # run without Dyad logic, that should be a policy LLM backbone derived explicitly from vllm's
        # DPEngineCoreProc, not something achieved by importing the wrong name.
        from agent_system.policies.dyad.rollout.vllm.dyad_core import DPEngineCoreActor
        # <<< DYAD-END

        self.local_engine_actors: list[ray.ActorHandle] = []
        self.remote_engine_actors: list[ray.ActorHandle] = []

        env_vars_list = get_env_vars_to_copy(destination="DPEngineCoreActor")
        self.env_vars_dict = {
            name: os.environ[name] for name in env_vars_list if name in os.environ
        }
        runtime_env = RuntimeEnv(env_vars=self.env_vars_dict)

        self.addresses = addresses
        self.executor_class = executor_class
        self.log_stats = log_stats
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_engine_count = vllm_config.parallel_config.data_parallel_size_local
        world_size = vllm_config.parallel_config.world_size

        if ray.is_initialized():
            logger.info("Ray is already initialized. Skipping Ray initialization.")
        else:
            ray.init()

        if placement_groups is not None:
            assert local_dp_ranks is not None, (
                "local_dp_ranks must be provided if placement_groups is provided"
            )
            assert len(placement_groups) == len(local_dp_ranks), (
                "placement_groups and local_dp_ranks must have the same length"
            )
            logger.info("Using provided placement groups")
            # TODO(rui): validate passed-in placement groups
            self.created_placement_groups = []
        else:
            placement_groups, local_dp_ranks = (
                CoreEngineActorManager.create_dp_placement_groups(vllm_config)
            )
            self.created_placement_groups = placement_groups
        assert len(placement_groups) == dp_size, (
            "Number of placement groups must match data parallel size"
        )

        self.placement_group_is_local = []
        refs = []
        for index, local_index, pg in zip(
            range(dp_size), local_dp_ranks, placement_groups
        ):
            dp_vllm_config = copy.deepcopy(vllm_config)
            dp_vllm_config.parallel_config.placement_group = pg
            local_client = index < local_engine_count

            # Ray XPU known issue: dpctl initializes the GPU runtime early, so
            # setting device env vars in Ray actor's initialization method
            # will not affect device selection. See:
            # https://github.com/ray-project/ray/blob/master/python/ray/_private/accelerators/intel_gpu.py#L56 # noqa: E501
            if current_platform.is_xpu():
                device_evar = current_platform.device_control_env_var
                device_indices = get_device_indices(
                    device_evar, local_index, world_size
                )
                actor_env_vars = self.env_vars_dict.copy()
                actor_env_vars[device_evar] = device_indices
                runtime_env = RuntimeEnv(env_vars=actor_env_vars)

            actor = (
                ray.remote(DPEngineCoreActor)
                .options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=world_size,
                    ),
                    runtime_env=runtime_env,
                )
                .remote(
                    vllm_config=dp_vllm_config,
                    executor_class=executor_class,
                    log_stats=log_stats,
                    local_client=local_client,
                    addresses=addresses,
                    dp_rank=index,
                    local_dp_rank=local_index,
                )
            )
            if local_client:
                self.local_engine_actors.append(actor)
            else:
                self.remote_engine_actors.append(actor)
            self.placement_group_is_local.append(local_client)
            refs.append(actor.wait_for_init.remote())

        ray.get(refs)
        self.run_refs = []
        for actor in self.local_engine_actors + self.remote_engine_actors:
            self.run_refs.append(actor.run.remote())

    @staticmethod
    def create_dp_placement_groups(
        vllm_config: VllmConfig,
    ) -> tuple[list["PlacementGroup"], list[int]]:
        """
        Create placement groups for data parallel.
        """

        import ray
        from ray._private.state import available_resources_per_node

        logger.info("Creating placement groups for data parallel")
        dp_master_ip = vllm_config.parallel_config.data_parallel_master_ip
        dp_size = vllm_config.parallel_config.data_parallel_size
        dp_size_local = vllm_config.parallel_config.data_parallel_size_local

        available_resources = available_resources_per_node()
        world_size = vllm_config.parallel_config.world_size
        placement_groups: list[PlacementGroup] = []
        local_dp_ranks: list[int] = []

        dp_master_ip_key = f"node:{dp_master_ip}"
        nodes = sorted(
            available_resources.values(), key=lambda x: dp_master_ip_key not in x
        )
        assert len(nodes) > 0, "No nodes with resources found in Ray cluster."
        assert dp_master_ip_key in nodes[0], (
            f"The DP master node (ip: {dp_master_ip}) is missing or dead"
        )
        device_str = current_platform.ray_device_key
        n_node_devices: list[int] = [
            int(node_resources[device_str])
            for node_resources in nodes
            if device_str in node_resources
        ]
        assert n_node_devices, f"No {device_str} found in Ray cluster."
        max_device_per_node = max(n_node_devices)

        pack_strategy = envs.VLLM_RAY_DP_PACK_STRATEGY
        _supported_pack_strategies = ("strict", "fill", "span")
        if pack_strategy not in _supported_pack_strategies:
            raise ValueError(
                f"{envs.VLLM_RAY_DP_PACK_STRATEGY} is not supported. "
                "Make sure to set `VLLM_RAY_DP_PACK_STRATEGY` "
                f"to one of {_supported_pack_strategies}"
            )

        all2all_backend = vllm_config.parallel_config.all2all_backend
        if pack_strategy == "fill" and (
            all2all_backend == "deepep_high_throughput"
            or all2all_backend == "deepep_low_latency"
        ):
            raise ValueError(
                "DeepEP kernels require EP ranks [0,7] (same for [8,15], ...) "
                "to be on the same node, but VLLM_RAY_DP_PACK_STRATEGY=fill "
                "does not guarantee that. "
                "Please use VLLM_RAY_DP_PACK_STRATEGY=strict instead."
            )

        if pack_strategy in ("strict", "fill"):
            placement_strategy = "STRICT_PACK"
        else:
            placement_strategy = "PACK"
            assert world_size > max_device_per_node, (
                f"World size {world_size} is smaller than the "
                "maximum number of devices per node "
                f"{max_device_per_node}. Make sure to set "
                "`VLLM_RAY_DP_PACK_STRATEGY` to `strict` or `fill`"
            )

            # if we need multiple nodes per dp group, we require for now that
            # available nodes are homogenous
            assert set(n_node_devices) == {max_device_per_node}, (
                f"Nodes are not homogenous, {nodes}"
            )
            assert world_size % max_device_per_node == 0, (
                f"For multi-node data parallel groups, world_size ({world_size}) must "
                f"be a multiple of number of devices per node ({max_device_per_node})."
            )
            assert len(n_node_devices) * max_device_per_node >= world_size * dp_size, (
                f"Not enough total available nodes ({len(n_node_devices)}) "
                f"and devices per node ({max_device_per_node}) "
                f"to satisfy required world size {world_size} and data parallel size "
                f"{dp_size}"
            )
            assert dp_size_local == 1, (
                f"data-parallel-size-local {dp_size_local} should be set as the "
                "default (1) for VLLM_RAY_DP_PACK_STRATEGY=span. "
                "The actual data-parallel-size-local will be auto determined."
            )

        # bundles collected for a single DP rank from multiple nodes,
        # for "span" pack strategy
        collected_bundles = []
        for node_resources in nodes:
            node_ip_keys = [
                key
                for key in node_resources
                if key != "node:__internal_head__" and key.startswith("node:")
            ]
            assert len(node_ip_keys) == 1, (
                f"Zero or multiple node IP keys found in node resources: {node_ip_keys}"
            )
            node_ip_key = node_ip_keys[0]
            node_ip = node_ip_key.split(":")[1]

            n_device_on_node = int(node_resources.get(device_str, 0))
            if pack_strategy == "span" and n_device_on_node != 0:
                # Strictly speaking,
                # dp_size_available = n_device_on_node / world_size
                # and is a fraction, but we use 1 for easier processing
                dp_size_available = 1
            else:
                dp_size_available = n_device_on_node // world_size

            if node_ip == dp_master_ip:
                if dp_size_available < dp_size_local:
                    raise ValueError(
                        f"Not enough resources to allocate {dp_size_local} DP ranks "
                        f"on DP master node {dp_master_ip}, possible to fit "
                        f"{dp_size_available} DP ranks."
                    )
                dp_size_to_allocate = dp_size_local
            elif pack_strategy == "strict":
                if dp_size_available < dp_size_local:
                    logger.info(
                        "Skipping node %s as %s DP ranks could not fit, "
                        "possible to fit %s DP ranks",
                        node_ip,
                        dp_size_local,
                        dp_size_available,
                    )
                    continue
                dp_size_to_allocate = dp_size_local
            else:
                # for "pack_strategy" in "fill" and "span"
                # we always take everything that's available
                dp_size_to_allocate = dp_size_available

            for i in range(dp_size_to_allocate):
                device_bundle = [{device_str: 1.0, "node:" + node_ip: 0.001}]
                if pack_strategy == "span":
                    collected_bundles += device_bundle * n_device_on_node
                    assert len(collected_bundles) <= world_size, (
                        "collected_bundles should be <= world_size, "
                        f"but got {len(collected_bundles)=} and {world_size=}"
                    )

                    # we only create a placement group if we collected enough devices
                    if len(collected_bundles) < world_size:
                        continue

                    bundles = collected_bundles + [{"CPU": 1.0}]
                    collected_bundles = []
                else:
                    bundles = device_bundle * world_size + [{"CPU": 1.0}]

                pg = ray.util.placement_group(
                    name=f"dp_rank_{len(placement_groups)}",
                    strategy=placement_strategy,
                    bundles=bundles,
                )
                placement_groups.append(pg)
                local_dp_ranks.append(i)
                if len(placement_groups) == dp_size:
                    break

        if len(placement_groups) < dp_size:
            raise ValueError(
                f"Not enough resources to allocate {dp_size} "
                "placement groups, only created "
                f"{len(placement_groups)} placement groups. "
                "Available resources: "
                f"{available_resources}"
            )
        assert len(placement_groups) == dp_size, (
            f"Created {len(placement_groups)} DP placement groups, expected {dp_size}"
        )
        assert len(local_dp_ranks) == dp_size, (
            f"local_dp_ranks length {len(local_dp_ranks)} does not match "
            f"expected {dp_size}"
        )
        return placement_groups, local_dp_ranks

    @staticmethod
    def add_dp_placement_groups(
        old_vllm_config: VllmConfig, new_data_parallel_size: int
    ) -> tuple[list["PlacementGroup"], list[int]]:
        """
        Add placement groups for new data parallel size.
        """
        import ray
        from ray._private.state import (
            available_resources_per_node,
            total_resources_per_node,
        )
        from ray.util.state import list_nodes

        old_dp_size = old_vllm_config.parallel_config.data_parallel_size
        num_pg_to_create = new_data_parallel_size - old_dp_size

        if num_pg_to_create <= 0:
            return [], []

        dp_master_ip = old_vllm_config.parallel_config.data_parallel_master_ip
        world_size = old_vllm_config.parallel_config.world_size

        nodes = list_nodes()
        nodes = sorted(nodes, key=lambda node: node.node_ip != dp_master_ip)
        assert nodes[0].node_ip == dp_master_ip, "The first node must be the head node"
        assert len(nodes) == 1 or nodes[1].node_ip != dp_master_ip, (
            "There can only be one head node"
        )

        available_resources = available_resources_per_node()
        total_resources = total_resources_per_node()

        placement_groups = []
        local_dp_ranks = []
        num_pg_created = 0

        device_str = current_platform.ray_device_key
        for node in nodes:
            if num_pg_created >= num_pg_to_create:
                break

            node_ip = node.node_ip
            node_id = node.node_id
            available_gpus = int(available_resources[node_id][device_str])

            # Get total GPUs on this node from the node's resources
            # Ray stores node resources with node ID as key
            total_gpus = int(total_resources[node_id][device_str])

            # Calculate used GPUs and used engines on this node
            used_gpus = max(0, total_gpus - available_gpus)
            used_engines_on_node = used_gpus // world_size

            # Calculate how many new engines this node can accommodate
            available_engine_count = available_gpus // world_size

            # Create placement groups for new engines on this node
            for i in range(available_engine_count):
                if num_pg_created >= num_pg_to_create:
                    break

                rank = old_dp_size + num_pg_created

                # Create bundles with node constraint for master node
                if node_ip == dp_master_ip:
                    bundles = [
                        {device_str: 1.0, "node:" + dp_master_ip: 0.001}
                    ] * world_size + [{"CPU": 1.0}]
                else:
                    bundles = [{device_str: 1.0}] * world_size + [{"CPU": 1.0}]

                pg = ray.util.placement_group(
                    name=f"dp_rank_{rank}",
                    strategy="STRICT_PACK",
                    bundles=bundles,
                )
                placement_groups.append(pg)

                # Local rank starts from the number of engines already used
                # on this node
                local_rank = used_engines_on_node + i
                local_dp_ranks.append(local_rank)
                num_pg_created += 1

        return placement_groups, local_dp_ranks

    def scale_up_elastic_ep(
        self, cur_vllm_config: VllmConfig, new_data_parallel_size: int
    ) -> None:
        import copy

        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        # >>> DYAD-BEGIN(vllm0.24): import dyad's own DPEngineCoreActor instead
        # 0.24 deleted vllm.v1.engine.core.DPEngineCoreActor and folded its job into
        # DPEngineCoreProc (all that remains in 0.24's core.py is a process name of the form
        # "DPEngineCoreActor_DP{rank}"). dyad already forks a class of the same name in
        # dyad_core.py -- DPEngineCoreActor(DyadDPEngineCoreProc) -- so that is what is imported.
        #
        # WARNING: this change alters semantics and needs confirming in the equivalence tests.
        # On 0.12 this line imported **vllm's original**, so ray.remote() wrapped an actor with no
        # Dyad logic at all; dyad's DPEngineCoreActor inherits DyadDPEngineCoreProc and does have it.
        # In other words ExpA_sys was most likely not running the Dyad version on the DP path --
        # either deliberately, or as an import mistake that went unnoticed for a long time. On 0.24
        # there is no second option: only dyad's class exists. If the DP path is genuinely meant to
        # run without Dyad logic, that should be a policy LLM backbone derived explicitly from vllm's
        # DPEngineCoreProc, not something achieved by importing the wrong name.
        from agent_system.policies.dyad.rollout.vllm.dyad_core import DPEngineCoreActor
        # <<< DYAD-END

        cur_data_parallel_size = len(self.local_engine_actors) + len(
            self.remote_engine_actors
        )

        assert new_data_parallel_size > cur_data_parallel_size, (
            f"New data parallel size {new_data_parallel_size} must be greater "
            f"than current data parallel size {cur_data_parallel_size} "
            "for scale up"
        )

        placement_groups, local_dp_ranks = self.add_dp_placement_groups(
            cur_vllm_config, new_data_parallel_size
        )

        world_size = cur_vllm_config.parallel_config.world_size
        dp_master_ip = cur_vllm_config.parallel_config.data_parallel_master_ip
        new_local_engines = 0

        runtime_env = RuntimeEnv(
            env_vars=self.env_vars_dict | {"VLLM_ELASTIC_EP_SCALE_UP_LAUNCH": "1"}
        )
        for i, (pg, local_rank) in enumerate(zip(placement_groups, local_dp_ranks)):
            rank = cur_data_parallel_size + i
            dp_vllm_config = copy.deepcopy(cur_vllm_config)
            dp_vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
            dp_vllm_config.parallel_config.placement_group = pg

            # Check if this placement group is on the head node
            local_client = any(
                bundle.get("node:" + dp_master_ip, 0) > 0 for bundle in pg.bundle_specs
            )

            if local_client:
                new_local_engines += 1
                # Update data_parallel_size_local
                dp_vllm_config.parallel_config.data_parallel_size_local = (
                    cur_vllm_config.parallel_config.data_parallel_size_local
                    + new_local_engines
                )

            actor = (
                ray.remote(DPEngineCoreActor)
                .options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=world_size,
                    ),
                    runtime_env=runtime_env,
                )
                .remote(
                    vllm_config=dp_vllm_config,
                    executor_class=self.executor_class,
                    log_stats=self.log_stats,
                    local_client=local_client,
                    addresses=self.addresses,
                    dp_rank=rank,
                    local_dp_rank=local_rank,
                )
            )

            if local_client:
                self.local_engine_actors.append(actor)
            else:
                self.remote_engine_actors.append(actor)
            self.created_placement_groups.append(pg)
            self.placement_group_is_local.append(local_client)

        ray.get(
            [
                actor.wait_for_init.remote()
                for actor in (
                    self.local_engine_actors[-new_local_engines:]
                    if new_local_engines > 0
                    else []
                )
                + self.remote_engine_actors[
                    -(len(placement_groups) - new_local_engines) :
                ]
            ]
        )

        actors = (
            self.local_engine_actors[-new_local_engines:]
            if new_local_engines > 0
            else []
        ) + self.remote_engine_actors[-(len(placement_groups) - new_local_engines) :]

        for actor in actors:
            self.run_refs.append(actor.run.remote())

        cur_vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        # Update old_vllm_config with new data_parallel_size_local if any new
        # local engines were added
        if new_local_engines > 0:
            cur_vllm_config.parallel_config.data_parallel_size_local += (
                new_local_engines
            )

    def scale_down_elastic_ep(
        self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None:
        import ray

        assert cur_data_parallel_size > new_data_parallel_size, (
            f"cur_data_parallel_size {cur_data_parallel_size} must be greater "
            f"than new_data_parallel_size {new_data_parallel_size} "
            "for scale down"
        )
        for _ in range(cur_data_parallel_size - new_data_parallel_size):
            pg = self.created_placement_groups.pop()
            is_local = self.placement_group_is_local.pop()
            if is_local:
                self.local_engine_actors.pop()
            else:
                self.remote_engine_actors.pop()
            ray.util.remove_placement_group(pg)

    def get_run_refs(self):
        return self.run_refs

    def close(self):
        import ray

        for actor in self.local_engine_actors + self.remote_engine_actors:
            ray.kill(actor)
        for pg in self.created_placement_groups:
            ray.util.remove_placement_group(pg)


@contextlib.contextmanager
def launch_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    num_api_servers: int = 1,
) -> Iterator[
    tuple[
        CoreEngineProcManager | CoreEngineActorManager | None,
        DPCoordinator | None,
        EngineZmqAddresses,
    ]
]:
    """
    Launch the Core Engine and the DP Coordinator (if needed).

    This is a context manager, responsible for:
    1. deciding whether a coordinator must be started, based on the data parallel configuration
    2. building the frontend <-> engine ZMQ addresses
    3. choosing, based on the backend:
       - ray actor mode
       - local multi-process mode
    4. in local multi-process mode, waiting via the handshake mechanism until every engine is ready

    What is yielded out:
    - engine manager (manages the engine processes or ray actors)
    - coordinator (if any)
    - addresses (all communication addresses)
    """

    # -----------------------------
    # read the parallel configuration
    # -----------------------------
    parallel_config = vllm_config.parallel_config

    # global data parallel replica count
    dp_size = parallel_config.data_parallel_size

    # how many engines this frontend/node is responsible for starting "locally"
    local_engine_count = parallel_config.data_parallel_size_local

    # in offline mode, the local starting rank of the local engines
    local_start_index = parallel_config.data_parallel_rank_local

    # the data parallel rank of this frontend/node itself
    dp_rank = parallel_config.data_parallel_rank

    # DP master node address
    host = parallel_config.data_parallel_master_ip

    # local_engines_only=True means this frontend only talks to "local/colocated" engines
    # Typical scenarios:
    # - hybrid_lb: hybrid load balancing
    # - external_lb: external load balancing
    local_engines_only = (
        parallel_config.data_parallel_hybrid_lb
        or parallel_config.data_parallel_external_lb
    )

    # -----------------------------
    # decide whether this is offline mode
    # -----------------------------
    # As the comment says:
    # in offline mode each DP rank has its own independent LLM instance,
    # and each LLM corresponds to one core engine.
    #
    # local_start_index is not None usually means:
    # we are in the per-rank-independent-instance mode of "offline inference"
    offline_mode = local_start_index is not None

    # -----------------------------
    # decide whether the frontend only reaches local engines
    # -----------------------------
    # client_local_only=True means:
    # this front-end only sends requests to colocated engines.
    #
    # It is True when:
    # 1. offline mode: each LLM instance only serves its own engine
    # 2. local_engines_only: explicitly only serving local engines
    # 3. local_engine_count == dp_size: meaning all engines are local
    client_local_only = (
        offline_mode or local_engines_only or (local_engine_count == dp_size)
    )

    # -----------------------------
    # build the input/output socket addresses for frontend <-> engine
    # -----------------------------
    # num_api_servers is how many API servers / frontends will connect to the engine
    # Each frontend needs one pair of input/output addresses
    addresses = EngineZmqAddresses(
        inputs=[
            get_engine_client_zmq_addr(client_local_only, host)
            for _ in range(num_api_servers)
        ],
        outputs=[
            get_engine_client_zmq_addr(client_local_only, host)
            for _ in range(num_api_servers)
        ],
    )

    # -----------------------------
    # whether a DP Coordinator must be started
    # -----------------------------
    # It is only started when:
    # 1. dp_size > 1: there really are multiple DP replicas
    # 2. not offline_mode: we are in online mode, not the per-rank independent offline mode
    # 3. dp_rank == 0: the coordinator is only started on rank 0
    run_coordinator = dp_size > 1 and not offline_mode and dp_rank == 0

    if run_coordinator:
        # create the coordinator process
        coordinator = DPCoordinator(parallel_config)

        # the coordinator exposes its own input/output socket addresses
        addresses.coordinator_input, addresses.coordinator_output = (
            coordinator.get_engine_socket_addresses()
        )

        # the coordinator also has an address for publishing frontend statistics
        addresses.frontend_stats_publish_address = (
            coordinator.get_stats_publish_address()
        )

        logger.info("Started DP Coordinator process (PID: %d)", coordinator.proc.pid)
    else:
        coordinator = None

    # -----------------------------
    # if the backend is ray, take the policy LLM backbone path
    # -----------------------------
    if parallel_config.data_parallel_backend == "ray":
        logger.info("Starting ray-based data parallel backend")

        engine_actor_manager = CoreEngineActorManager(
            vllm_config=vllm_config,
            addresses=addresses,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        # hand the ray actor manager straight back to the caller
        yield engine_actor_manager, coordinator, addresses
        return

    # -----------------------------
    # non-ray mode: decide next "who must take part in the handshake"
    # -----------------------------
    #
    # "handshake" means that an engine registers itself with some ROUTER socket after starting.
    # Through the handshake the frontend learns:
    # - which engines have started
    # - their rank/index
    # - their communication addresses
    #
    # engines_to_handshake is "the list of engines expected to complete the handshake successfully"

    if offline_mode:
        # In offline mode each LLM instance corresponds to exactly one engine of one DP rank,
        # so there can only be one local engine
        assert local_engine_count == 1

        # The index of that engine is the current dp_rank, and it is necessarily local=True
        engines_to_handshake = [CoreEngine(index=dp_rank, local=True)]

    elif dp_rank == 0:
        # In online mode, besides its own local engines, rank 0
        # is also responsible for completing the handshake with "all core engines".
        #
        # Because rank 0 owns the Coordinator, it needs to know every engine.
        # Even if rank 0 has no local engine at all (headless),
        # it must still wait for every engine to come and handshake.
        engines_to_handshake = [
            CoreEngine(index=i, local=(i < local_engine_count)) for i in range(dp_size)
        ]

    else:
        # A frontend with rank > 0 only handshakes with the engines it manages locally
        # This requires that we are in local_engines_only mode,
        # because internal DPLB does not support initiating such core_engine startup outside rank 0
        assert local_engines_only, (
            "Attempting to launch core_engines from dp_rank > 0, but "
            "found internal DPLB, which is incompatible."
        )

        # The current rank manages a contiguous span of local engines
        # Note: the index here starts at dp_rank,
        # meaning each rank owns the engine index range that corresponds to it.
        engines_to_handshake = [
            CoreEngine(index=i, local=True)
            for i in range(dp_rank, dp_rank + local_engine_count)
        ]

    # -----------------------------
    # determine whether the handshake address is "locally visible only"
    # -----------------------------
    #
    # handshake_local_only=True means:
    # the started engine only needs to handshake with the local/colocated frontend.
    #
    # Under external_dp_lb, engines on rank > 0
    # also need to reach the rank 0 frontend, so it cannot be local-only.
    handshake_local_only = offline_mode or local_engine_count == dp_size

    handshake_address = get_engine_client_zmq_addr(
        handshake_local_only, host, parallel_config.data_parallel_rpc_port
    )

    # -----------------------------
    # In some scenarios the handshake address must be set separately for the "local process manager"
    # and for the "client-visible address"
    # -----------------------------
    #
    # Condition:
    # local_engines_only and dp_rank > 0
    #
    # This means:
    # - the local process manager can handshake with local engines over a local-only IPC address
    # - but what the engine reports outward to clients/remote frontends is still the global handshake_address
    if local_engines_only and dp_rank > 0:
        assert not handshake_local_only

        # the local manager listens on an IPC address, for fast handshakes with local engines
        local_handshake_address = get_open_zmq_ipc_path()

        # client_handshake_address is passed to the engine so that it knows
        # which handshake address to expose to clients, on top of the local IPC handshake address
        client_handshake_address = local_handshake_address
    else:
        # the general case: the local listen address = the outward handshake address
        local_handshake_address = handshake_address
        client_handshake_address = None

    # -----------------------------
    # create a ROUTER socket to receive the engines' handshake messages
    # -----------------------------
    with zmq_socket_ctx(
        local_handshake_address, zmq.ROUTER, bind=True
    ) as handshake_socket:

        from .dyad_core import DyadEngineCoreProc

        # -----------------------------
        # start the local engine processes (if this node is responsible for starting them)
        # -----------------------------
        if local_engine_count:
            local_engine_manager = CoreEngineProcManager(
                # entry function of each child process: actually run the EngineCore
                DyadEngineCoreProc.run_engine_core,

                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,

                # the address the engine process must finally handshake with
                handshake_address=handshake_address,

                # in certain special modes, the handshake address handed to the client
                client_handshake_address=client_handshake_address,

                # the current manager is a local client
                local_client=True,

                # how many engines to start locally
                local_engine_count=local_engine_count,

                # the starting point of the global engine index
                start_index=dp_rank,

                # the starting point of the local rank; offline mode may have a dedicated local_start_index
                local_start_index=local_start_index or 0,
            )
        else:
            # the current rank may be headless and start no local engine
            local_engine_manager = None

        # -----------------------------
        # hand "manager / coordinator / addresses" to the upper layer first
        # -----------------------------
        yield local_engine_manager, coordinator, addresses

        # -----------------------------
        # before leaving the with block, wait for every engine to finish starting and handshaking
        # -----------------------------
        #
        # This step is crucial:
        # it blocks until all expected engines have registered,
        # otherwise it means startup failed, the address is unreachable, a process crashed, etc.
        wait_for_engine_startup(
            handshake_socket,
            addresses,
            engines_to_handshake,
            parallel_config,
            vllm_config.cache_config,
            local_engine_manager,
            coordinator.proc if coordinator else None,
        )


def wait_for_engine_startup(
    handshake_socket: zmq.Socket,
    addresses: EngineZmqAddresses,
    core_engines: list[CoreEngine],
    parallel_config: ParallelConfig,
    cache_config: CacheConfig,
    proc_manager: CoreEngineProcManager | None,
    coord_process: Process | None,
):
    """
    Wait until all expected core engines have completed the startup handshake.

    Arguments:
    - handshake_socket:
        the ZMQ socket (usually a ROUTER) the frontend/manager uses to receive engine handshake messages
    - addresses:
        the set of communication addresses between frontend, engines and coordinator;
        it may be supplemented/updated during the handshake
    - core_engines:
        the list of engines expected to take part in the handshake
    - parallel_config:
        the parallel configuration, in particular the data parallel related settings
    - cache_config:
        the cache configuration; num_gpu_blocks is aggregated here once each engine is READY
    - proc_manager:
        the local engine process manager; used to detect whether a local engine exited early
    - coord_process:
        the coordinator process; if present, it must also be monitored for abnormal exit

    What this function does:
    1. count how many local and remote engines are still not connected / not ready
    2. poll and wait for HELLO / READY messages on handshake_socket
    3. at the same time monitor whether the local engine processes and the coordinator process died early
    4. on HELLO, send back the initialization information (addresses, DP config, config hash)
    5. on READY, receive the engine's initialization results (such as num_gpu_blocks)
    6. check configuration consistency across all engines, until every one of them is READY
    """

    # ---------------------------------------------------------
    # 1) count the local and remote engines
    # ---------------------------------------------------------
    # local_count: the number of engines this frontend/manager is "locally responsible" for
    local_count = parallel_config.data_parallel_size_local

    # remote_count: total engines expected to handshake - local engines = remote engines
    remote_count = len(core_engines) - local_count

    # conn_pending: the [local, remote] engine counts that have not finished the first "connect/HELLO" stage
    # start_pending: the [local, remote] engine counts that already sent HELLO but are not READY yet
    #
    # Initial state:
    # - no engine has sent HELLO yet, so conn_pending = [local_count, remote_count]
    # - no engine has entered the "waiting for READY" stage, so start_pending = [0, 0]
    #
    # The two [local, remote] slots track them separately.
    conn_pending, start_pending = [local_count, remote_count], [0, 0]

    # ---------------------------------------------------------
    # 2) create the poller and watch the handshake socket
    # ---------------------------------------------------------
    poller = zmq.Poller()
    poller.register(handshake_socket, zmq.POLLIN)

    # ---------------------------------------------------------
    # 3) decide whether "remote engines should be headless"
    # ---------------------------------------------------------
    # remote_should_be_headless=True when:
    # it is neither hybrid_lb nor external_lb
    #
    # That is, under "plain internal DPLB", remote workers usually do not need to carry a frontend,
    # so they should be headless.
    #
    # Under external/hybrid dp lb mode, remote engines usually must not be headless,
    # because they also have to cooperate with the frontend/load balancing logic.
    remote_should_be_headless = (
        not parallel_config.data_parallel_hybrid_lb
        and not parallel_config.data_parallel_external_lb
    )

    # ---------------------------------------------------------
    # 4) have the poller also watch the sentinels of the local engine child processes
    # ---------------------------------------------------------
    # A sentinel is the "process termination notification handle" provided by multiprocessing.Process.
    # As soon as a child process ends, it becomes readable.
    #
    # This way the poller is not only waiting for handshake messages, but can simultaneously detect:
    # "did some engine process die before it managed to handshake?"
    if proc_manager is not None:
        for sentinel in proc_manager.sentinels():
            poller.register(sentinel, zmq.POLLIN)

    # ---------------------------------------------------------
    # 5) if there is a coordinator, watch the coordinator's sentinel too
    # ---------------------------------------------------------
    if coord_process is not None:
        poller.register(coord_process.sentinel, zmq.POLLIN)

    # ---------------------------------------------------------
    # 6) main loop: keep waiting as long as some engine has not finished HELLO or READY
    # ---------------------------------------------------------
    while any(conn_pending) or any(start_pending):
        # poll once, waiting for a fixed period
        events = poller.poll(STARTUP_POLL_PERIOD_MS)

        # -----------------------------------------------------
        # 6.1) no event at all: print the debug log of "who are we still waiting for"
        # -----------------------------------------------------
        if not events:
            if any(conn_pending):
                print("Waiting for %d local, %d remote core engine proc(s) to connect.")
                logger.debug(
                    "Waiting for %d local, %d remote core engine proc(s) to connect.",
                    *conn_pending,
                )
            if any(start_pending):
                print("Waiting for %d local, %d remote core engine proc(s) to start.")
                logger.debug(
                    "Waiting for %d local, %d remote core engine proc(s) to start.",
                    *start_pending,
                )
            continue

        # Startup expects handshake messages only. A child/coordinator sentinel indicates
        # initialization failure, so stop waiting and raise.
        if len(events) > 1 or events[0][0] != handshake_socket:
            # read the local core processes that already finished, together with their exit codes
            finished = proc_manager.finished_procs() if proc_manager else {}

            # if the coordinator also exited, add it to the failure list
            if coord_process is not None and coord_process.exitcode is not None:
                finished[coord_process.name] = coord_process.exitcode

            raise RuntimeError(
                "Dyad Engine core initialization failed. "
                "See root cause above. "
                f"Failed core proc(s): {finished}"
            )

        # -----------------------------------------------------
        # 6.3) receive a message from the handshake socket
        # -----------------------------------------------------
        # A ROUTER socket recv_multipart() yields:
        # - eng_identity: the engine's identity (byte string)
        # - ready_msg_bytes: the message body sent by the engine (msgpack encoded)
        eng_identity, ready_msg_bytes = handshake_socket.recv_multipart()

        # the identity is decoded little-endian into an integer, which is usually the engine index / dp rank
        eng_index = int.from_bytes(eng_identity, "little")

        # find the engine that sent this message in the expected core_engines list
        engine = next((e for e in core_engines if e.identity == eng_identity), None)

        # if this identity is not in the expected list, an "unexpected" engine showed up
        if engine is None:
            raise RuntimeError(
                f"Message from engine with unexpected data parallel rank: {eng_index}"
            )

        # deserialize the handshake message
        msg = msgspec.msgpack.decode(ready_msg_bytes)

        # the handshake message carries at least these fields:
        # - status: "HELLO" or "READY"
        # - local: whether this engine is local
        # - headless: whether this engine is headless
        status, local, headless = msg["status"], msg["local"], msg["headless"]

        # -----------------------------------------------------
        # 6.4) verify the local/remote identity matches expectation
        # -----------------------------------------------------
        # If some engine reports itself as local while we expected it to be remote (or vice versa),
        # our understanding of the system topology is wrong, so raise immediately.
        if local != engine.local:
            raise RuntimeError(
                f"{status} message from "
                f"{'local' if local else 'remote'} "
                f"engine {eng_index}, expected it to be "
                f"{'local' if engine.local else 'remote'}"
            )

        # -----------------------------------------------------
        # 6.5) verify the headless mode of remote engines matches expectation
        # -----------------------------------------------------
        # Rules:
        # - under internal DPLB (neither hybrid nor external), a remote engine must be headless
        # - under external/hybrid dp lb, a remote engine must not be headless
        if not local and headless != remote_should_be_headless:
            if headless:
                raise RuntimeError(
                    f"Remote engine {eng_index} must not use "
                    f"--headless in external or hybrid dp lb "
                    f"mode"
                )
            else:
                raise RuntimeError(
                    f"Remote engine {eng_index} must use "
                    f"--headless unless in external or hybrid "
                    f"dp lb mode"
                )

        # -----------------------------------------------------
        # 6.6) first stage: handle HELLO
        # -----------------------------------------------------
        if status == "HELLO" and engine.state == CoreEngineState.NEW:
            # On receiving HELLO, the frontend/manager sends back an init_message.
            #
            # This init_message contains:
            # - addresses: the communication addresses of frontend/backend, coordinator, etc.
            # - parallel_config: the key settings DP needs
            # - parallel_config_hash: the config hash, used to guarantee all DP workers agree
            #
            # Only after receiving this init_message can the engine continue its initialization.
            init_message = msgspec.msgpack.encode(
                EngineHandshakeMetadata(
                    addresses=addresses,
                    parallel_config={
                        k: getattr(parallel_config, k)
                        for k in (
                            "data_parallel_master_ip",
                            "data_parallel_master_port",
                            "_data_parallel_master_port_list",
                            "data_parallel_size",
                        )
                    },
                    parallel_config_hash=parallel_config.compute_hash()
                    if parallel_config.data_parallel_size > 1
                    else None,
                )
            )

            # send it back to the corresponding engine (a ROUTER socket needs the identity)
            handshake_socket.send_multipart((eng_identity, init_message), copy=False)

            # this engine has finished the "connect/HELLO" stage
            conn_pending[0 if local else 1] -= 1

            # but it is not READY yet, so it enters the "waiting for startup completion" stage
            start_pending[0 if local else 1] += 1

            # update the state: NEW -> CONNECTED
            engine.state = CoreEngineState.CONNECTED

        # -----------------------------------------------------
        # 6.7) second stage: handle READY
        # -----------------------------------------------------
        elif status == "READY" and engine.state == CoreEngineState.CONNECTED:
            # >>> DYAD-BEGIN(vllm0.24): the READY handshake no longer carries num_gpu_blocks
            # On 0.12 the engine returned the num_gpu_blocks it computed during init, and in the DP
            # case the frontend summed the engines' values back into cache_config.num_gpu_blocks.
            # 0.24 moved kv cache sizing entirely inside the engine (settled by
            # initialize_from_config); the field is gone from the READY payload and the frontend no
            # longer aggregates it -- reading it anyway is
            # KeyError: 'num_gpu_blocks'.
            #
            # 0.24's READY branch also dropped the frontend_stats_publish_address passthrough, so
            # that goes too, to avoid reading an equally absent dp_stats_address.
            # Compare: wait_for_engine_startup in vllm/v1/engine/utils.py.
            # <<< DYAD-END

            # -------------------------------------------------
            # verify the DP worker config hashes agree
            # -------------------------------------------------
            # When multiple DP workers take part in collective communication, some settings must be identical.
            # Otherwise communication errors, inconsistent behaviour or deadlocks are likely.
            #
            # So each worker reports its own parallel_config_hash when it becomes READY,
            # and it is compared here against the locally expected value.
            if parallel_config.data_parallel_size > 1:
                worker_config_hash = msg.get("parallel_config_hash")
                expected_hash = parallel_config.compute_hash()
                if worker_config_hash != expected_hash:
                    raise RuntimeError(
                        f"Configuration mismatch detected for engine "
                        f"{eng_index}. All DP workers must have identical "
                        f"configurations for parameters that affect collective "
                        f"communication (e.g., enable_eplb, "
                        f"eplb_config.log_balancedness). "
                        f"Worker hash: {worker_config_hash}, "
                        f"Expected hash: {expected_hash}. "
                        f"Please ensure all workers are started with the same "
                        f"command-line arguments."
                    )

            # this engine reached READY, so decrement the corresponding pending counter
            start_pending[0 if local else 1] -= 1

            # update the state: CONNECTED -> READY
            engine.state = CoreEngineState.READY

        # -----------------------------------------------------
        # 6.8) everything else is an illegal state transition
        # -----------------------------------------------------
        else:
            raise RuntimeError(
                f"Unexpected {status} message for "
                f"{'local' if local else 'remote'} engine "
                f"{eng_index} in {engine.state} state."
            )

        # -----------------------------------------------------
        # 6.9) emit a debug log line
        # -----------------------------------------------------
        logger.debug(
            "%s from %s core engine process %s.",
            status,
            "local" if local else "remote",
            eng_index,
        )
