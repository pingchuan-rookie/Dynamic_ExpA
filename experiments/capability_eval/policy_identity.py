"""Compatibility entry for the shared inference policy_identity implementation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent_system.inference.policy_identity import *  # noqa: F403
