# DYAD-DATASET-RESUME: shared identity checks for text and expanded-action training.
"""Bind RLHF dataloader positions to the actual ordered, filtered dataset."""

from __future__ import annotations

_IDENTITY_KEY = "_dyad_dataset_identity"


def _sampler_identity(sampler):
    """Describe traversal semantics, not the mutable cursor/RNG checkpoint state."""
    identity = {"class": f"{type(sampler).__module__}.{type(sampler).__qualname__}", "length": len(sampler)}
    for key in (
        "num_samples",
        "replacement",
        "size",
        "padded_size",
        "batch_size",
        "drop_last",
        "num_replicas",
        "rank",
        "shuffle",
        "seed",
    ):
        if hasattr(sampler, key):
            identity[key] = getattr(sampler, key)
    return identity


def _dataloader_identity(dataloader):
    import copy

    from omegaconf import OmegaConf

    settings = getattr(dataloader.dataset, "config", {}).get("sampler")
    if OmegaConf.is_config(settings):
        settings = OmegaConf.to_container(settings, resolve=True)
    return {
        "dataset": dataloader.dataset.checkpoint_identity(),
        "batch_size": dataloader.batch_size,
        "drop_last": dataloader.drop_last,
        "num_workers": dataloader.num_workers,
        "sampler": _sampler_identity(dataloader.sampler),
        "batch_sampler": _sampler_identity(dataloader.batch_sampler),
        "sampler_settings": copy.deepcopy(settings),
    }


def dataloader_state_dict(dataloader):
    state = dataloader.state_dict()
    identity = getattr(dataloader.dataset, "checkpoint_identity", None)
    # Trainers save immediately after the last batch, before the iterator gets
    # its final next()/StopIteration. Preserve the sampler RNG but mark that
    # exhausted epoch complete, so resume does not spend an epoch on zero batches.
    yielded = state.get("_num_yielded")
    if yielded is None and "_snapshot" in state:
        yielded = state["_snapshot"]["_snapshot_step"] + state["_steps_since_snapshot"]
    if yielded is not None and yielded == len(dataloader):
        state["_iterator_finished"] = True
    if identity is not None:
        state[_IDENTITY_KEY] = _dataloader_identity(dataloader)
    return state


def load_dataloader_checkpoint(dataloader, path):
    import os

    import torch

    if not os.path.isfile(path):
        raise ValueError(f"Exact training resume requires dataloader state: {path}")
    return validate_dataloader_state(dataloader, torch.load(path, weights_only=False))


def validate_dataloader_state(dataloader, state):
    """Validate before restoring the sampler, including at an epoch boundary.

    Historical RLHFDataset checkpoints contain no subset/filter identity.
    A seed supplied now cannot prove which rows were used then, even if counts
    match, so these checkpoints can be evaluated but not resumed for training.
    Custom datasets without this identity protocol retain their existing behavior.
    """
    identity = getattr(dataloader.dataset, "checkpoint_identity", None)
    if identity is not None:
        saved = state.get(_IDENTITY_KEY)
        if saved is None:
            raise ValueError(
                "Cannot verify training dataset identity: this legacy data.pt has no "
                "dataset identity metadata. Setting data.seed now cannot recover the "
                "historical sampled/filtered rows. Use the checkpoint for evaluation "
                "or start a new training run; exact training resume is unsupported."
            )
        expected = _dataloader_identity(dataloader)
        if saved != expected:
            raise ValueError(
                "Training dataset identity or dataloader configuration changed since the checkpoint. "
                "Exact resume requires the same ordered sampled/filtered rows, data.seed, "
                "sampling limits, sampler semantics/settings, batch size and worker count; "
                "restore the original data/configuration."
            )
    elif _IDENTITY_KEY in state:
        raise ValueError("Checkpoint requires dataset identity verification, but the current dataset cannot provide it")
    state = dict(state)
    state.pop(_IDENTITY_KEY, None)
    return state
