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


"""Forward the project worker environment while retaining explicit config overrides.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations


def forward_worker_environment(runtime_env_kwargs: dict):
    # Forward project worker settings through the existing Ray runtime initialization.
    from verl.trainer.main_ppo import (
        os,
    )

    _diag_env_vars = {
        k: v
        for k, v in os.environ.items()
        # Forward whole prefixes so new options cannot silently disappear in workers.
        if k.startswith(
            ("DYAD_", "GRPO_", "CALC_ENV", "ALFWORLD_ENV", "CODEGYM_", "WEBSHOP_", "REACT_TOOL_NAME", "VERL_", "VLLM_")
        )
    }
    if _diag_env_vars:
        runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
        for _k, _v in _diag_env_vars.items():
            runtime_env_vars.setdefault(_k, _v)
        runtime_env_kwargs["env_vars"] = runtime_env_vars
    # DYAD-TRAPI: Forward only nonsecret environment-LLM settings to the
    # actors that create isolated environment pools. Tokens are worker-local.
    from agent_system.environments.env_package.dive.model_api.providers.trapi import (
        TRAPI_ENV_VARS as DIVE_TRAPI_ENV_VARS,
    )
    from agent_system.environments.env_package.t2bench.trapi import TRAPI_ENV_VARS as T2BENCH_TRAPI_ENV_VARS

    runtime_env_vars = runtime_env_kwargs.setdefault("env_vars", {})
    for name in (
        *dict.fromkeys((*DIVE_TRAPI_ENV_VARS, *T2BENCH_TRAPI_ENV_VARS)),
        "DIVE_JUDGE_PROVIDER",
        "DIVE_JUDGE_MODEL",
        "DIVE_JUDGE_BASE_URL",
        "BROWSE_LLM_PROVIDER",
        "BROWSE_LLM_MODEL",
        "BROWSE_LLM_BASE_URL",
    ):
        if name in os.environ:
            runtime_env_vars.setdefault(name, os.environ[name])
