# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
# Copyright 2026 Dyad contributors.
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
"""Prompt preparation shared by independent-decision collectors."""


def strip_thinking_prefill(prompt_ids: list[int], tokenizer) -> list[int]:
    """Leave tag generation to the policy, as required by official projection.

    Qwen3.5 templates append an empty or open think block after the assistant
    header. Remove only an exact token suffix before generation, never from a
    generated response or from the training sample after sampling.
    """
    for text in ("<think>\n\n</think>\n\n", "<think>\n"):
        suffix = tokenizer.encode(text, add_special_tokens=False)
        if suffix and prompt_ids[-len(suffix):] == suffix:
            return prompt_ids[:-len(suffix)]
    return prompt_ids
