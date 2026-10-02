# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Pointer note for the ALFWorld env "session" (this file holds no logic, it only points).

Unlike calc, ALFWorld has no in-repo `AlfworldSession` class: its "one trajectory = one session"
state *is* the **external TextWorld env instance** held by the worker, and the env engine comes from
the **external package `alfworld`**. This placeholder only keeps the `agent_system/environments/` directory
symmetric with calc and makes it obvious where the session logic lives.

The real things are in:
  - env engine (external): `alfworld.agents.environment.alfred_tw_env.AlfredTWEnv` (pip package alfworld)
  - session wrapper / lifecycle: `agent_system/environments/backends/alfworld/worker.py`
      · `_build_single_game_env()` -- builds an AlfredTWEnv that loads a single game only
      · `AlfworldEnvWorker.reset(game_idx)/step(action)/close()` -- one worker holds one env and is re-reset to different games
  - game data / index: `agent_system/environments/configs/alfworld_base_config.yaml` + `configs/alfworld_mappings_{train,test}.json` (ALFWORLD_DATA points at the game files)

Compare `agent_system/environments/backends/calc/session.py`: calc has no external env package, so its session logic
(CalcSession) lives in-repo; the alfworld equivalent is the external `alfworld` package, hence there
is no implementation here, only a pointer placeholder.
"""

# Pointers for quick lookup, programmatic or by eye (no runtime effect).
ENV_ENGINE = "alfworld.agents.environment.alfred_tw_env.AlfredTWEnv"  # external pip package
WRAPPED_BY = "agent_system.environments.backends.alfworld.worker.AlfworldEnvWorker"
