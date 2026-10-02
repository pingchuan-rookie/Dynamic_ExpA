"""Policy-side contract for one independently sampled environment decision."""
from __future__ import annotations

from typing import Any, Protocol


class StepPolicy(Protocol):
    """Supply sampling context and decision identity without owning the environment."""

    async def prepare(self, loop: Any, session: Any, kwargs: dict[str, Any]) -> None:
        ...

    def sampling_params(self, params: dict[str, Any], step_index: int, max_steps: int) -> dict[str, Any]:
        ...

    def trace(self, generated: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
        ...

    def selected_action_names(self, payload: dict[str, Any] | None, environment: str) -> tuple[str, ...] | None:
        """Return verified selections, or None when the policy imposes no selection constraint."""
        ...

    @property
    def extra_fields(self) -> dict[str, Any]:
        ...
