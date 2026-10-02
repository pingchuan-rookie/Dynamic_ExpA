"""Export trusted native FSDP checkpoints as lossless, standalone HF policies.

No action encoder or projector is constructed. Only the policy architecture from
checkpoint metadata is instantiated, on meta, to validate tensor names and shapes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore_tensor(name, values):
    """Reconstruct a one-dimensional FSDP mesh without casting or guessing layouts."""
    import torch
    from torch.distributed.tensor import DTensor

    first = values[0]
    size = len(values)
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise ValueError(f"{name}: checkpoint value is not a tensor")
    if any(
        value.dtype != first.dtype or tuple(value.shape) != tuple(first.shape) or value.layout != torch.strided
        for value in values
    ):
        raise ValueError(f"{name}: rank dtype, global shape or dense layout mismatch")
    if isinstance(first, DTensor):
        if not all(isinstance(value, DTensor) for value in values):
            raise ValueError(f"{name}: mixed tensor layouts")
        placement = first.placements
        if len(placement) != 1:
            raise ValueError(f"{name}: only one-dimensional FSDP placement is supported")
        for rank, value in enumerate(values):
            mesh = value.device_mesh
            if (
                mesh.mesh.tolist() != list(range(size))
                or mesh.mesh_dim_names != ("fsdp",)
                or getattr(mesh, "_rank", None) != rank
                or value.placements != placement
            ):
                raise ValueError(f"{name}: rank mesh or placement mismatch")
        tensors = [value._local_tensor.detach().cpu() for value in values]
        rule = placement[0]
        if rule.is_shard():
            dim = rule.dim
            if dim < 0 or dim >= first.ndim:
                raise ValueError(f"{name}: invalid shard dimension")
            width = (first.shape[dim] + size - 1) // size
            for rank, tensor in enumerate(tensors):
                shape = list(first.shape)
                shape[dim] = max(0, min(width, first.shape[dim] - rank * width))
                if tuple(tensor.shape) != tuple(shape):
                    raise ValueError(f"{name}: local shard shape mismatch")
            result = torch.cat(tensors, dim=dim).contiguous()
        elif rule.is_replicate():
            if any(not torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
                raise ValueError(f"{name}: replicated tensor values disagree")
            result = tensors[0].contiguous()
        else:
            raise ValueError(f"{name}: partial or unknown placement is unsupported")
    else:
        if not all(type(value) is torch.Tensor for value in values):
            raise ValueError(f"{name}: unsupported checkpoint tensor type")
        if any(not torch.equal(first, value) for value in values[1:]):
            raise ValueError(f"{name}: plain replicated tensor values disagree")
        result = first.detach().cpu().contiguous()
    if result.dtype != first.dtype or tuple(result.shape) != tuple(first.shape):
        raise ValueError(f"{name}: reconstruction changed dtype or shape")
    return result


def checkpoint_files(checkpoint):
    actor = checkpoint / "actor"
    metadata = actor / "fsdp_config.json"
    size = json.loads(metadata.read_text()).get("world_size")
    if type(size) is not int or size < 1:
        raise ValueError("fsdp_config.json must declare a positive integer world_size")
    files = [actor / f"model_world_size_{size}_rank_{rank}.pt" for rank in range(size)]
    if set(actor.glob("model_world_size_*_rank_*.pt")) != set(files):
        raise ValueError("Checkpoint must contain exactly one model shard for every declared rank")
    if not all(path.is_file() for path in files):
        raise ValueError("Checkpoint model shard is missing")
    return files, metadata


def export_policy(checkpoint, output, *, max_shard_bytes=1024**3):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; select a fresh directory")
    if output.is_relative_to(checkpoint) or checkpoint.is_relative_to(output):
        raise ValueError("Output must be separate from the source checkpoint")
    if type(max_shard_bytes) is not int or max_shard_bytes < 1:
        raise ValueError("max_shard_bytes must be positive")
    shards, fsdp_config = checkpoint_files(checkpoint)
    hf = checkpoint / "actor/huggingface"
    if not (hf / "config.json").is_file():
        raise ValueError("Checkpoint must include actor/huggingface/config.json")
    if not any((hf / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise ValueError("Checkpoint must include its tokenizer, not a substituted base tokenizer")
    metadata_files = [
        p
        for p in hf.iterdir()
        if p.is_file() and p.suffix in (".json", ".model", ".txt", ".jinja") and not p.name.endswith(".index.json")
    ]
    config_path = checkpoint.parent / "model_config.json"
    training_config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    from agent_system.inference.policy_identity import training_identity

    identity = training_identity(training_config) if training_config else {}
    source_files = [*shards, fsdp_config, *metadata_files]
    if config_path.is_file():
        source_files.append(config_path)
    source_hashes = {str(path): file_digest(path) for path in source_files}
    # Native FSDP DTensors contain Python metadata; only trusted checkpoints may be loaded.
    ranks = [torch.load(path, map_location="cpu", weights_only=False, mmap=True) for path in shards]
    if any(not isinstance(rank, dict) or set(rank) != set(ranks[0]) for rank in ranks):
        raise ValueError("Native rank tensor key sets disagree")
    config = AutoConfig.from_pretrained(hf, local_files_only=True, trust_remote_code=False)
    architecture = (config.architectures or [""])[0]
    cls = AutoModelForImageTextToText if "ForConditionalGeneration" in architecture else AutoModelForCausalLM
    with torch.device("meta"):
        policy = cls.from_config(config, trust_remote_code=False)
    expected = policy.state_dict()
    missing = set(expected) - set(ranks[0])
    extras = set(ranks[0]) - set(expected)
    dyad_keys = {key for key in extras if key.startswith("dyad_residual_head.") or key == "action_head.weight"}
    if missing or extras != dyad_keys:
        raise ValueError(
            f"Policy tensor contract mismatch: missing={sorted(missing)}, unexpected={sorted(extras - dyad_keys)}"
        )
    if dyad_keys and identity.get("action_interface") != "dyad":
        raise ValueError("Dyad-only tensors require saved model_config.json with an Dyad algorithm identity")
    state, excluded = {}, {}
    for name in sorted(ranks[0]):
        tensor = restore_tensor(name, [rank[name] for rank in ranks])
        if name in expected:
            if tuple(tensor.shape) != tuple(expected[name].shape):
                raise ValueError(f"{name}: tensor shape differs from checkpoint policy architecture")
            state[name] = tensor
        else:
            excluded[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    del ranks
    policy.load_state_dict(state, strict=True, assign=True)
    if getattr(config, "tie_word_embeddings", False):
        if not torch.equal(policy.get_input_embeddings().weight, policy.get_output_embeddings().weight):
            raise ValueError("Checkpoint tied input/output embedding values disagree")
    output.mkdir(parents=True, exist_ok=False)
    try:
        for path in metadata_files:
            shutil.copyfile(path, output / path.name)
        groups, group, size = [], {}, 0
        for name, tensor in sorted(state.items()):
            nbytes = tensor.numel() * tensor.element_size()
            if group and size + nbytes > max_shard_bytes:
                groups.append(group)
                group, size = {}, 0
            group[name] = tensor
            size += nbytes
        if group:
            groups.append(group)
        index = {
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())},
            "weight_map": {},
        }
        for number, group in enumerate(groups, 1):
            filename = f"model-{number:05d}-of-{len(groups):05d}.safetensors"
            # Clones avoid shared storage for tied embeddings without dropping checkpoint keys.
            save_file(
                {name: tensor.clone() for name, tensor in group.items()},
                str(output / filename),
                metadata={"format": "pt"},
            )
            with safe_open(output / filename, framework="pt", device="cpu") as saved:
                if set(saved.keys()) != set(group):
                    raise ValueError("Exported safetensor key set differs from native policy")
                for name, tensor in group.items():
                    value = saved.get_tensor(name)
                    if value.dtype != tensor.dtype or not torch.equal(value, tensor):
                        raise ValueError(f"{name}: exported policy tensor differs from checkpoint")
            index["weight_map"].update({name: filename for name in group})
        (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
        if any(file_digest(path) != digest for path, digest in source_hashes.items()):
            raise ValueError("Checkpoint source changed during export")
        manifest = {
            "version": 1,
            "source_checkpoint": str(checkpoint),
            "source_files_sha256": source_hashes,
            "training_algorithm": training_config.get("algo"),
            **identity,
            "world_size": len(shards),
            "policy_only": True,
            "action_encoder_loaded": False,
            "projector_loaded": False,
            "dtype_preserved": True,
            "policy_strict_load": True,
            "serialized_tensor_equality_verified": True,
            "policy_tensor_count": len(state),
            "excluded_non_policy_tensors": excluded,
            "artifact_files_sha256": {p.name: file_digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        }
        (output / "policy_export.json").write_text(json.dumps(manifest, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted native global_step_N directory")
    parser.add_argument("--output", type=Path, required=True, help="Fresh HF policy export directory")
    args = parser.parse_args(argv)
    try:
        manifest = export_policy(args.checkpoint, args.output)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Policy export failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "policy": str(args.output.resolve()),
                "tensor_count": manifest["policy_tensor_count"],
                "dtype_preserved": True,
                "action_encoder_loaded": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
