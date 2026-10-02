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


"""Project defaults consumed by the upstream structured configuration fields.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations


def gigpo_defaults() -> dict:
    """Return fresh gigpo defaults for each algorithm config."""
    return {
        "step_advantage_w": 1.0,
        "mode": "mean_std_norm",
        "rollout_layout": "trajectory",
        "history_length": 2,
        "invalid_action_penalty": 0.1,
        "enable_similarity": False,
        "similarity_thresh": 0.95,
    }


def step_rollout_defaults() -> dict:
    """Return fresh step_rollout defaults for each algorithm config."""
    return {
        "enabled": False,
        "protocol_version": 2,
        "profile": "alfworld_official_v2",
        "action_interface": "text",
        "history_length": 2,
        "max_steps": 50,
        "invalid_action_penalty": 0.1,
        "resampling": "reference_copy",
        "loss_reduction": "reference_microbatch",
        "compute_mean_std_cross_steps": True,
    }
