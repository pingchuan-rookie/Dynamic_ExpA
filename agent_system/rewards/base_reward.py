# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Shared base for env reward functions (ALFWorld / CodeGym).

gsm8k is deliberately absent: it scores through verl's upstream `reward_score/gsm8k.py`, and
../dynamic-expa-design/reference/architecture.md calls that decoupling intentional. Do not pull it in here.

What is genuinely common is only the *shape tolerance* -- an env flag arrives as a bare bool, a
`[True]` from TextWorld's batch API, or a numpy/torch scalar. The lookup *cascades* are not common
and are left in the subclasses:

  - ALFWorld consults the first container that is present and stops there, even when that container
    holds neither key (an elif chain, so `infos` is unreachable once `info` is a dict).
  - CodeGym walks every container, and additionally honours `turn_scores` first and numeric
    reward fields last.

Those differences decide real reward values, so they are transcribed rather than harmonised. Key
order differs too: ALFWorld reads success-then-won, CodeGym won-then-success.

`compute_score` stays a module-level callable in each subclass module -- verl's reward dispatch in
verl/utils/reward_score/__init__.py imports the module and calls that name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence


class BaseEnvReward(ABC):
    """Scores one trajectory from the env metrics the agent loop collected into extra_info."""

    # ---- specified by subclasses ----
    SUCCESS_KEYS: tuple[str, ...] = ()
    CONTAINERS: tuple[str, ...] = ()

    @staticmethod
    def to_bool(value: Any) -> bool:
        """Normalise the various shapes the env may return into a bool."""
        # ALFWorld/TextWorld sometimes wraps the flag in a list/tuple, e.g. [True].
        if isinstance(value, (list, tuple)):
            return bool(value[0]) if len(value) else False
        try:
            # Also accepts numpy / torch scalars, e.g. tensor(True).
            if hasattr(value, "item"):
                return bool(value.item())
        except Exception:
            pass
        return bool(value)

    @staticmethod
    def to_float(value: Any) -> float:
        """Same shape tolerance as to_bool, for numeric reward fields."""
        if isinstance(value, (list, tuple)):
            return float(value[0]) if len(value) else 0.0
        try:
            if hasattr(value, "item"):
                return float(value.item())
        except Exception:
            pass
        return float(value)

    @classmethod
    def find_flag(cls, mapping: dict, keys: Sequence[str]) -> Optional[bool]:
        """First present key in `keys`, normalised to bool. None when the mapping holds none of them."""
        for key in keys:
            if key in mapping:
                return cls.to_bool(mapping[key])
        return None

    @abstractmethod
    def compute_score(self, solution_str, ground_truth=None, extra_info=None, **kwargs) -> float:
        """Return this trajectory's episode reward."""
