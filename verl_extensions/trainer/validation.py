# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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


"""Map native environment scores onto the shared validation panels.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations


def environment_metric_names(reward_extra_infos_dict: dict):
    # Map acc := trajectory reward and reward := sum(turn_scores), retaining raw won/score.
    # Apply only to environment outputs so single-turn baseline metrics remain unchanged.
    if "won" in reward_extra_infos_dict or "score" in reward_extra_infos_dict:
        reward_extra_infos_dict = dict(reward_extra_infos_dict)
        if "reward" in reward_extra_infos_dict:
            reward_extra_infos_dict["acc"] = list(reward_extra_infos_dict["reward"])
        if "score" in reward_extra_infos_dict:
            reward_extra_infos_dict["reward"] = list(reward_extra_infos_dict["score"])
    return reward_extra_infos_dict
