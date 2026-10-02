"""Verify packaged environment sources without requiring an external Git checkout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def source_identity(root: str | Path, *, verify: bool = True) -> dict[str, Any]:
    """Return the recorded upstream identity and verify the exact distributed files."""
    root = Path(root).resolve()
    manifest = json.loads((root / "source_manifest.json").read_text())
    commit = manifest.get("source_commit")
    if not isinstance(commit, str) or len(commit) != 40 or not manifest.get("files"):
        raise ValueError(f"Invalid environment source manifest: {root}")
    if verify:
        for name, expected in manifest["files"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Missing or invalid environment source file: {name}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Environment source hash mismatch: {name}")
    return {
        "commit": commit,
        "tracked_changes": manifest.get("local_changes", ""),
        "source": "packaged_manifest",
        "runtime_data_dirty": False if verify else None,
    }
