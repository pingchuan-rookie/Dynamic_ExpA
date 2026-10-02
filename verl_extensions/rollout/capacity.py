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


"""Reserve environment sessions for the actual dispatched shard.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations


async def reserve_batch_capacity(self, validate: bool, batch):
    # Configure environment capacity for the dispatched sampling shard.
    from verl.experimental.agent_loop.agent_loop import (
        os,
    )

    capacity_tools = self.tools
    if validate and os.environ.get("DYAD_VAL_TOOL_CONFIG"):
        from verl.tools.tool_registry import initialize_tools_from_config

        capacity_tools = initialize_tools_from_config(os.environ["DYAD_VAL_TOOL_CONFIG"])
    for tool in capacity_tools:
        if getattr(tool, "DEFAULT_ENV_TYPE", None) in {"alfworld", "codegym"} and len(batch):
            await tool.pool.ensure_capacity(len(batch))
