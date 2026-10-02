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


"""Bind resume positions to the actual filtered dataset content.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations


def dataset_checkpoint_identity(self):
    import pyarrow as pa

    from verl.utils.dataset.rl_dataset import (
        hashlib,
    )

    cache = getattr(self, "_checkpoint_identity_cache", None)
    if cache is None or cache[0] != id(self.dataframe):
        digest = hashlib.sha256()
        # Rebuild bounded logical batches: combine_chunks alone can retain
        # unselected child buffers inside sliced nested arrays.
        # Arrow values avoid decoding image/audio features during hashing.
        for table in self.dataframe.with_format("arrow").iter(batch_size=1024):
            table = pa.Table.from_pylist(table.to_pylist(), schema=table.schema.remove_metadata())
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            digest.update(sink.getvalue())
        self._checkpoint_identity_cache = (id(self.dataframe), digest.hexdigest())
    return {
        "version": 1,
        "rows_sha256": self._checkpoint_identity_cache[1],
        "num_rows": len(self.dataframe),
        "seed": self.seed,
        "shuffle": self.shuffle,
        "max_samples": self.max_samples,
    }
