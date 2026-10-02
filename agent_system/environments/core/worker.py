# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Shared base class for single-env workers (Calc / ALFWorld / CodeGym).

The contract itself is already stated in `agent_system/environments/core/pool.py`: a worker actor must implement
reset(**reset_spec) / step(action) / close() / health_check(*health_args), and reset/step return a
dict shaped {observation, reward, available_actions, done, step_count, ...}. This class turns that
prose contract into code so the three workers cannot drift apart silently.

**Import-light is the hard constraint** (see agent_system/environments/README.md and
dyad_test/test_envworker_import_light.py): the standard library only, never verl / torch / ray.
One stray import here is inherited by all three workers, so every actor process pays for it.

Deliberately a plain class rather than an ABC: subclasses get wrapped by `ray.remote(...)`, and a
plain class keeps ABCMeta out of that path entirely. This also matches the hook style of
`BaseEnvPool`, which is a plain class for the same family of reasons.

Subclasses override the following:
  - IMPORT_PROBES : module names debug_import_state() reports on (extend, do not replace)
  - reset / step / close / health_check : the whole env-specific surface
"""

from __future__ import annotations


class BaseEnvWorker:
    """One env instance, leased by one trajectory at a time, resettable to different tasks.

    Subclasses hold whatever "session" their env needs: an in-repo object (calc), an external
    package's env (alfworld), or a class loaded from an env file at runtime (codegym).
    """

    # ---- specified by subclasses ----
    # Probed by debug_import_state(). torch/verl are the two that must stay False; a subclass adds
    # its own heavy optional dependency when knowing whether it loaded is diagnostically useful.
    IMPORT_PROBES: tuple[str, ...] = ("torch", "verl")

    # =====================================================================
    # Env-specific surface (implemented by subclasses)
    # =====================================================================
    def reset(self, **reset_spec) -> dict:
        """Bind this worker to a task and return the initial observation dict.

        The accepted keys are env-specific: calc takes ground_truth/max_turns, alfworld takes
        game_idx/world_type, codegym takes env_str.
        """
        raise NotImplementedError

    def step(self, action) -> dict:
        """Apply an environment action (text or a JSON-safe structured value)."""
        raise NotImplementedError

    def finalize(self, reason: str) -> dict:
        """Optional terminal-result protocol; closing must not implicitly score."""
        raise NotImplementedError

    def close(self) -> dict:
        """Release the underlying env. Must tolerate being called when nothing is open."""
        raise NotImplementedError

    def health_check(self, *health_args) -> dict:
        """Round-trip self-check; returns a dict containing 'ok'.

        codegym needs an env_str to have something to open, the other two take no argument.
        """
        raise NotImplementedError

    # =====================================================================
    # Shared diagnostics
    # =====================================================================
    def debug_import_state(self) -> dict:
        """Diagnostic: did the policy LLM backbone process get dragged into torch/verl (import-light expects False)."""
        import sys

        state = {f"{name}_imported": name in sys.modules for name in self.IMPORT_PROBES}
        state["num_modules"] = len(sys.modules)
        return state
