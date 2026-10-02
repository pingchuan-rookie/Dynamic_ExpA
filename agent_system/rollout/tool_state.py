# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


"""Keep trajectory environment state and diagnostics outside the upstream tool implementation.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations


def initialize_environment_state(self, extra_info: dict | None):
    from verl.experimental.agent_loop.tool_agent_loop import (
        Any,
        Optional,
    )

    self.extra_info: dict[str, Any] = extra_info or {}
    # Use environment-reported success rather than inferring it from a reward threshold.
    self.env_won: bool = False
    # CodeGym format validity = fmt_valid / fmt_attempts.
    self.fmt_valid = 0
    self.fmt_attempts = 0
    # Diagnostics only: preserve termination reasons and action/observation spans for replay inspection.
    self.termination_reason: Optional[str] = None
    self.turn_records: list[dict[str, Any]] = []
    self.span_records: list[dict[str, Any]] = []
    # Preserve the train/validation identity in trajectory summaries.
    self.is_validate: bool = False
    # Termination reason is diagnostic only.
    self.termination_reason: Optional[str] = None
    # Full diagnostic mode writes turn and span records.
    self.turn_records: list[dict[str, Any]] = []
    self.span_records: list[dict[str, Any]] = []
    # DYAD-GIGPO: training state, independent of diagnostic collection.
    self.gigpo_enabled = False
    self.gigpo_steps: list[dict[str, Any]] = []
    self.gigpo_anchor: Optional[str] = None
    self.gigpo_initial = True


def bind_native_tool_parser(self):
    # Prefer REACT_TOOL_NAME, then the sole registered tool; a mismatched name would skip execution.
    from verl.experimental.agent_loop.tool_agent_loop import (
        os,
    )

    if getattr(self.tool_parser, "TOOL_NAME", None) is not None:
        _react_tool_name = os.getenv("REACT_TOOL_NAME") or (next(iter(self.tools)) if len(self.tools) == 1 else None)
        if _react_tool_name:
            self.tool_parser.TOOL_NAME = _react_tool_name
