import enum
import time
from collections.abc import Mapping
from typing import Any

import msgspec
import torch

from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import LogprobsLists, LogprobsTensors
from vllm.v1.serial_utils import UtilityResult
from vllm.v1.engine import EngineCoreOutput,UtilityOutput


class DyadEngineCoreOutputs(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    # NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout

    engine_index: int = 0

    # [num_reqs]
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: SchedulerStats | None = None
    timestamp: float = 0.0

    utility_output: UtilityOutput | None = None
    finished_requests: set[str] | None = None

    # In DP case, used to signal that the current wave of requests
    # has finished and the engines are paused.
    wave_complete: int | None = None
    # In DP case, used to signal that a request was received for an
    # "old" wave, so the next wave needs to be started in other engines.
    start_wave: int | None = None

    action_content:dict | None = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()
