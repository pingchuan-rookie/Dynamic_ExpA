# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Pointer note for the CodeGym env "session" (this file holds no logic, it only points).

Unlike calc, CodeGym has no in-repo `CodeGymSession` class: its "one trajectory = one session"
state *is* the **external env instance** held by the worker, and the env class comes from an
**external env file** (gymnasium-based) that is loaded dynamically at runtime from the env directory
by `env_str`. This placeholder only keeps the `agent_system/environments/` directory symmetric with calc and
makes it obvious where the session logic lives.

The real things are in:
  - env class (external): the env classes in `data/codegym/dataset/envs/codegym_v1/*.py`
      (the env directory can be overridden by the tool config's `envs_dir` or the `CODEGYM_ENVS_DIR` environment variable)
  - session loading / wrapper / lifecycle: `agent_system/environments/backends/codegym/worker.py`
      · `parse_codegym_env_str()` -- parses env_str into (filename, class_name, inner_env_str)
      · `_load_env_class()`       -- dynamically loads the env class from envs_dir by file name (cached inside the worker)
      · `CodeGymEnvWorker.reset(env_str)/step(action)/close()` -- one worker holds one env and is re-reset to different env_str values

Compare `agent_system/environments/backends/calc/session.py`: calc has no external env package, so its session logic
(CalcSession) lives in-repo; the codegym equivalent is the external env file, hence there is no
implementation here, only a pointer placeholder.
"""

# Pointers for quick lookup, programmatic or by eye (no runtime effect).
ENV_ENGINE = "data/codegym/dataset/envs/codegym_v1/*.py"  # external env file (loaded at runtime)
WRAPPED_BY = "agent_system.environments.backends.codegym.worker.CodeGymEnvWorker"
