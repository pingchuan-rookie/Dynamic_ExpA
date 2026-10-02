"""Compose policy implementations without importing unused model backends."""
from __future__ import annotations

from agent_system.policies.base import StepPolicy


def make_step_policy(action_interface: str) -> StepPolicy:
    if action_interface == "text":
        from agent_system.policies.text import TextStepPolicy

        return TextStepPolicy()
    if action_interface == "dyad":
        from agent_system.policies.dyad.policy import DyadStepPolicy

        return DyadStepPolicy()
    raise ValueError(f"Unsupported action interface: {action_interface!r}; expected text or dyad")
