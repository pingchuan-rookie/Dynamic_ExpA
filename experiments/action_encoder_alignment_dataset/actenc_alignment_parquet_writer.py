"""Publish prepared Alignment rows atomically with round-trip verification."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from agent_system.policies.dyad.data.actenc_alignment_parquet import SCHEMA, SPLIT_POLICY, _validate_rows, read_dataset


def write_dataset(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    """Write compressed Parquet atomically, after exact round-trip verification."""
    path = Path(path)
    rows = list(rows)
    _validate_rows(rows)
    table = pa.Table.from_pylist(rows, schema=SCHEMA.with_metadata({
        b"split_policy": SPLIT_POLICY.encode("utf-8"),
    }))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp = Path(temporary)
    try:
        pq.write_table(table, temp, compression="zstd")
        if read_dataset(temp) != rows:
            raise ValueError("Parquet round trip changed Alignment samples")
        # mkstemp is private by default; published datasets must remain readable by cluster jobs.
        temp.chmod((path.stat().st_mode & 0o777) if path.exists() else 0o644)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return len(rows)

