"""Resolve prepared data, checkpoints, and metrics for supervised pretraining."""
from __future__ import annotations

import os
from pathlib import Path

from agent_system.utils.artifact_paths import artifact_root, run_site

PROJECT = Path(__file__).resolve().parents[5]


def data_root() -> Path:
    override = os.environ.get("DYAD_ALIGNMENT_DATA", "").strip()
    return Path(override) if override else PROJECT / "data" / "actenc_alignment"



def final_dir() -> Path:
    return data_root() / "final"



def dataset_path() -> Path:
    return final_dir() / "dataset.parquet"



def records_dir() -> Path:
    """Training metrics and analysis inputs; legacy checkpoint runs use --runs-dir explicitly."""
    return artifact_root(PROJECT) / "outputs" / run_site() / "alignment"



def runs_dir() -> Path:
    """Checkpoint root; historical sources remain readable through explicit paths."""
    return artifact_root(PROJECT) / "ckpt" / run_site() / "alignment"


def evaluation_dir() -> Path:
    return artifact_root(PROJECT) / "outputs" / run_site() / "alignment_test"
