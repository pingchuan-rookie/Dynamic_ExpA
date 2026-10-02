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


"""Close loaders, trackers and TransferQueue in the original task-runner finally boundary.

Derived from verl v0.9.0, commit 483b8a009ba3a97563edee3a19887e4862b8094a.
DYAD-ADD: explicit calls at the original upstream stage delegate here.
The trainer main loop and dispatch decorators remain in verl.
"""

from __future__ import annotations


def finish_task(trainer, succeeded: bool):
    import transfer_queue as tq

    try:
        # DYAD-LOADER-LIFECYCLE: join children while the task actor is alive,
        # including initialization/fit failures and validation-only returns.
        # Close owned loaders and tracking services while the shared task actor is still alive.
        if trainer is not None:
            try:
                trainer.close_dataloaders()
            except BaseException:
                succeeded = False
                raise
    finally:
        # Close owned loaders and tracking services while the shared task actor is still alive.
        try:
            tracking = getattr(trainer, "logger", None)
            if tracking is not None:
                exit_code = 0 if succeeded else 1
                try:
                    tracking.finish(exit_code=exit_code)
                finally:
                    # DYAD-TRACKING-LIFECYCLE: finish the process-owned service
                    # before Ray exits; run.finish() leaves its atexit hook alive.
                    wandb = getattr(tracking, "logger", {}).get("wandb")
                    if wandb is not None:
                        wandb.teardown(exit_code=exit_code)
        finally:
            tq.close()
