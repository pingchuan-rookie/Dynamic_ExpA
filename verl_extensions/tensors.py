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


"""Preserve the true jagged axis through TensorDict serialization.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verl.utils.tensordict_utils import TensorDict


def maybe_fix_3d_position_ids(data: TensorDict):
    # DYAD-NESTED-POSITION-IDS: TQ legitimately packs equal-length [C, S]
    # rows on axis 1 as [B*C, S], whereas TensorDict consolidation can lose
    # axis 2 metadata for [C, sum(S)] storage. Never relabel the former as
    # axis 2: its offsets count channels, not tokens.
    # Normalize positional tensor layouts without corrupting ordinary or shared-step batch dimensions.
    from verl.utils.tensordict_utils import (
        nested_tensor_from_tensor_list,
        torch,
    )

    position_ids = data.get("position_ids")
    if not isinstance(position_ids, torch.Tensor) or not position_ids.is_nested or position_ids.dim() != 3:
        return
    if position_ids.layout != torch.jagged:
        data["position_ids"] = nested_tensor_from_tensor_list(list(position_ids.unbind()), ragged_idx=2)
        return
    if position_ids._ragged_idx == 2:
        return

    values, offsets = position_ids.values(), position_ids.offsets()
    packed_size = int(offsets[-1])
    channel_packed = packed_size == values.shape[0]
    token_packed = packed_size == values.shape[1]
    if channel_packed and token_packed:
        # Square buffers are ambiguous after consolidation. Token lengths
        # distinguish them without guessing a model's number of RoPE channels.
        input_ids = data.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or not input_ids.is_nested:
            raise ValueError("Ambiguous nested position_ids layout requires nested input_ids")
        token_packed = torch.equal(offsets.diff(), input_ids.offsets().diff())
        channel_packed = not token_packed
    if token_packed:
        # Rebuild all shape/stride metadata, not only the private axis flag.
        data["position_ids"] = torch.nested.nested_tensor_from_jagged(
            values=values, offsets=offsets, lengths=position_ids.lengths(), jagged_dim=2
        )
    elif channel_packed:
        data["position_ids"] = nested_tensor_from_tensor_list(list(position_ids.unbind()), ragged_idx=2)
    else:
        raise ValueError("Nested position_ids offsets do not match either packed dimension")
