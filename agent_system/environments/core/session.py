# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Protocol declaration for an env session (the per-trajectory state a worker holds).

This file declares a contract; it carries no implementation. Only calc has an in-repo session class
(`CalcSession`) -- for alfworld and codegym the "session" *is* an external env instance, so
`backends/alfworld/session.py` / `backends/codegym/session.py` stay pointer placeholders. See those files, and
`agent_system/environments/README.md` on why the asymmetry is intentional rather than missing work.

Import-light applies here as everywhere in this package: standard library only, never verl / torch / ray.

An ABC here, unlike the plain-class `BaseEnvWorker` next door: nothing wraps a session in
`ray.remote(...)`, so ABCMeta stays out of the actor path either way, and for a class whose entire
purpose is enforcing a protocol, failing at instantiation beats failing at first call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEnvSession(ABC):
    """One trajectory's env state: reset to a task, stepped, then closed.

    Same four verbs as `BaseEnvWorker`, one layer down -- a worker is the Ray-facing wrapper, a
    session is the env logic itself.
    """

    @abstractmethod
    def reset(self, **reset_spec: Any) -> dict:
        """Bind this session to a task and return the initial observation dict."""

    @abstractmethod
    def step(self, action: str) -> dict:
        """Apply one action and return the resulting observation dict."""

    @abstractmethod
    def close(self) -> dict:
        """Release env resources. Must tolerate being called when nothing is open."""

    @abstractmethod
    def health_check(self) -> dict:
        """Round-trip self-check; returns a dict containing 'ok'."""
