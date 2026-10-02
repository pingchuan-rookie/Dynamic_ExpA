"""Compatibility entry for the shared inference export_policy implementation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent_system.inference.export_policy import *  # noqa: F403

if __name__ == "__main__":
    raise SystemExit(main())
