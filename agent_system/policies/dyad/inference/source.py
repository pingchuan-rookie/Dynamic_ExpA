"""Checkpoint provenance shared by standalone inference clients and servers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(value):
    wire_value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return hashlib.sha256(
        json.dumps(wire_value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_metadata(args):
    checkpoint = getattr(args, "checkpoint", None)
    projector = getattr(args, "projector_init", None)
    if bool(checkpoint) == bool(projector):
        raise ValueError(
            "Dyad requires exactly one of --checkpoint (native global_step_N) "
            "or --projector-init (exact Alignment source)"
        )
    explicit = getattr(args, "model_config", None)
    if getattr(args, "model_path", None):
        raise ValueError("Dyad policy model is restored from model_config.json; unset MODEL_PATH/--model-path")
    if checkpoint:
        source = Path(checkpoint).expanduser().resolve()
        import re

        if not re.fullmatch(r"global_step_\d+", source.name) or not (source / "actor").is_dir():
            raise ValueError(
                "Dyad --checkpoint must be native global_step_N containing actor/, not HF export or Alignment projector"
            )
        shards = sorted((source / "actor").glob("model_world_size_*_rank_*.pt"))
        if not shards or any(p.stat().st_size == 0 for p in shards):
            raise ValueError("Native checkpoint has no nonempty model shards")
        topology = [
            tuple(map(int, re.fullmatch(r"model_world_size_(\d+)_rank_(\d+)\.pt", p.name).groups())) for p in shards
        ]
        sizes = {size for size, _ in topology}
        if len(sizes) != 1 or sorted(rank for _, rank in topology) != list(range(next(iter(sizes)))):
            raise ValueError("Native checkpoint model shard topology is incomplete or mixed")
        config_path = Path(explicit).expanduser().resolve() if explicit else source.parent / "model_config.json"
        kind = "native_stage2"
        files = {str(p.relative_to(source)): file_digest(p) for p in shards}
    else:
        source = Path(projector).expanduser()
        if not source.is_absolute() and not source.exists():
            project = Path(__file__).resolve().parents[4]
            import sys

            if str(project) not in sys.path:
                sys.path.insert(0, str(project))
            from agent_system.utils.artifact_paths import artifact_root

            root = artifact_root(project) / "ckpt"
            site = source.parts[0] if source.parts[0] in ("local", "lucia") else "lucia"
            run = Path(*source.parts[1:]) if source.parts[0] in ("local", "lucia") else source
            candidates = [root / site / phase / run for phase in ("alignment", "stage1")]
            hits = [path for path in candidates if path.is_file() or (path / "projector.pt").is_file()]
            if len(hits) != 1:
                raise ValueError(
                    "Alignment run must identify exactly one checkpoint, including historical stage1 paths"
                )
            source = hits[0]
        if source.is_dir():
            source = source / "projector.pt"
        source = source.resolve()
        if not source.is_file() or not source.stat().st_size:
            raise ValueError("Exact Alignment projector.pt is missing/empty; no latest-run fallback")
        if not explicit:
            raise ValueError(
                "--projector-init requires --model-config to identify the two models and projector configuration"
            )
        config_path = Path(explicit).expanduser().resolve()
        kind = "stage1_projector"
        files = {"projector.pt": file_digest(source)}
    from agent_system.policies.dyad.checkpoint_compat import read_saved_model_config

    config = read_saved_model_config(config_path)
    if kind == "native_stage2" and config.get("model", {}).get("DYAD_ENCODER_TRAINING") == "projector_and_encoder_lm":
        encoder = source / "actor/encoder_backbone.pt"
        if not encoder.is_file():
            raise ValueError("Native checkpoint is missing trained encoder weights")
        files["actor/encoder_backbone.pt"] = file_digest(encoder)
    if config.get("algo") not in {"dyad", "dyad-grpo", "dyad-gigpo"} or not isinstance(config.get("model"), dict):
        raise ValueError("model_config.json must describe native Dyad, not a baseline")
    from agent_system.inference.policy_identity import training_identity
    training_identity(config)
    identity = {
        "kind": kind,
        "source": str(source),
        "files_sha256": files,
        "model_config_sha256": file_digest(config_path),
        "training_benchmark": config.get("benchmark"),
    }
    return {
        "source": str(source),
        "identity": identity,
        "identity_sha256": digest(identity),
        "model_config": str(config_path),
        "local_files_verified": True,
        "local_weights_verified": True,
    }
