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


"""Measure prompt lengths once for filtering and diagnostics.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations


def filter_and_report_prompt_lengths(self, dataframe, doc2len):
    # Reuse a temporary length column; counts already include tool-schema tokens when configured.
    import numpy as np

    dataframe = dataframe.map(
        lambda doc: {"__prompt_len__": doc2len(doc)},
        num_proc=self.num_workers,
        desc="Measuring prompt token length",
    )
    _lens = np.asarray(dataframe["__prompt_len__"], dtype=np.int64)
    _n = int(_lens.size)
    _over = int((_lens > self.max_prompt_length).sum())
    print(
        f"[prompt length] n={_n} "
        f"min={int(_lens.min()) if _n else 0} "
        f"max={int(_lens.max()) if _n else 0} "
        f"mean={float(_lens.mean()) if _n else 0:.1f} "
        f"p50={int(np.percentile(_lens, 50)) if _n else 0} "
        f"p90={int(np.percentile(_lens, 90)) if _n else 0} "
        f"p99={int(np.percentile(_lens, 99)) if _n else 0} "
        f"| max_prompt_length={self.max_prompt_length} "
        f"tools_counted={self.tool_schemas is not None} "
        f"over={_over} ({(_over / _n * 100) if _n else 0:.1f}%) keep={_n - _over}"
    )

    dataframe = dataframe.filter(
        lambda doc: doc["__prompt_len__"] <= self.max_prompt_length,
        num_proc=self.num_workers,
        desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
    )
    dataframe = dataframe.remove_columns("__prompt_len__")
    return dataframe
