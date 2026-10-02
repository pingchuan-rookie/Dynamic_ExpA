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
"""ALFWorld action projection shared by current and compatibility collectors."""
import re


def project_alfworld_action(text: str, *, no_thinking: bool = False) -> tuple[str, bool]:
    """Official projection, including executing malformed outputs' fallback.

    Admissibility is intentionally not a validity condition in the official
    implementation. Action tags are case-insensitive; think tags are not.
    ``no_thinking`` is only the frozen legacy action-only protocol, not a model mode.
    """
    lowered = text.lower()
    start, end = lowered.find("<action>"), lowered.find("</action>")
    if start == -1 or end == -1:
        return lowered[-30:], False
    action = lowered[start + len("<action>"):end].strip().lower()
    if no_thinking:
        # No post-generation stripping: generated reasoning is an invalid action-only format.
        valid = re.fullmatch(r"\s*<action>[^<>]+</action>\s*", text, re.IGNORECASE) is not None
    else:
        valid = "<think>" in text and "</think>" in text
    return action, valid and re.search(r"[一-鿿]", text) is None
