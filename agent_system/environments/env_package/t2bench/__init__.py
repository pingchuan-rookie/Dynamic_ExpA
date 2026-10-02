"""Standalone t2bench environment."""

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

_SOURCE = Path(__file__).resolve().parent / "source"
_PROJECT = Path(__file__).resolve().parents[4]
_DATA = Path(os.environ.get("TAU2_DATA_DIR", _PROJECT / "data/t2bench/source/data"))
os.environ.setdefault("TAU2_DATA_DIR", str(_DATA))
if str(_SOURCE / "src") not in sys.path:
    sys.path.insert(0, str(_SOURCE / "src"))


if TYPE_CHECKING:
    from .envs import T2BenchEnv


def build_env(**kwargs: Any) -> "T2BenchEnv":
    """Construct one environment without allocating rollout workers."""
    from .envs import T2BenchEnv

    return T2BenchEnv(**kwargs)
