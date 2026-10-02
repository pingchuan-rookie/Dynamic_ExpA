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


"""Preserve the existing project tracking cleanup behavior.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations


def tracking_finish(self, exit_code: int = 0):
    """Flush and finalize every configured backend exactly once."""
    if getattr(self, "_finished", False):
        return
    self._finished = True
    loggers = getattr(self, "logger", {})

    for name, call in (
        ("wandb", lambda lg: lg.finish(exit_code=exit_code)),
        ("swanlab", lambda lg: lg.finish()),
        ("vemlp_wandb", lambda lg: lg.finish(exit_code=exit_code)),
        ("tensorboard", lambda lg: lg.finish()),
        ("clearml", lambda lg: lg.finish()),
        ("trackio", lambda lg: lg.finish()),
        ("file", lambda lg: lg.finish()),
        ("rl_insight", lambda lg: lg.finish()),
    ):
        if name not in loggers:
            continue
        try:
            call(loggers[name])
        except (BrokenPipeError, OSError, ConnectionError) as e:
            print(f"[WARN] {name}.finish() failed during shutdown: {e}")
        except Exception as e:
            print(f"[WARN] unexpected {name}.finish() error: {e}")
