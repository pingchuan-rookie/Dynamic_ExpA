"""Best-effort JSONL diagnostics for the experimental Dyad training path."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


_WRITE_LOCK = threading.Lock()
_WARNED = False
_DEFAULT_MAX_VALUES = 64
_DEFAULT_DUMP_LIMIT = 64
_DEFAULT_OBS_MAXLEN = 2000
_DYAD_KEYS = ("response_dyad", "tool_mask", "dyad_action_mask", "seq_mask")

OFF = "off"
NORMAL = "normal"
FULL = "full"

# Marks a list or dict that was cut down to `_max_values()` entries on the way into the jsonl.
# It is part of the **on-disk contract**, not an implementation detail: every consumer under
# experiments/shared/analysis/ has to skip it, because in a list it sits where a value would.
# A consumer that does not know it either crashes on it (decode_raw did) or counts it as one more
# admissible action, which is worse -- a 65-way choice reported where the run actually faced 83.
TRUNCATION_KEY = "__truncated_items__"
_FULL_ALIASES = {"full", "debug", "verbose", "2"}
_RUN_META_DONE: set[str] = set()


def _enabled() -> bool:
    return os.getenv("DYAD_DIAG_ENABLED", "0").lower() not in {"0", "false", "no", "off"}


def diag_level() -> str:
    """`off` / `normal` / `full`. See ../dynamic-expa-design/reference/diagnostics_contract.md §7.

    `full` adds the per-token payload (token ids, masks, decoded text, decision traces) that the
    offline decode layer needs. It is expensive -- a gsm8k debug run writes ~1 MB of jsonl per 64
    trajectories -- so only the `*_debug.sh` entrypoints ask for it; a production run that turns
    diagnostics on gets `normal`, which is scalars and action strings only.
    """
    if not _enabled():
        return OFF
    return FULL if os.getenv("DYAD_DIAG_LEVEL", NORMAL).strip().lower() in _FULL_ALIASES else NORMAL


def full_dump() -> bool:
    """Gate for a per-token payload field inside an otherwise `normal` event."""
    return diag_level() == FULL


def log_event_full(component: str, event: str, **data: Any) -> None:
    """Emit an event that exists only to serve the offline decode layer."""
    if diag_level() != FULL:
        return
    log_event(component, event, **data)


def record_span(spans: list[dict[str, Any]], kind: str, start: int, length: int) -> None:
    """Append one `[start, start+length)` span in response-mask coordinates.

    The offline aligned view can already split the mask into runs of 1s and 0s, but then "3 GEN
    runs" and "assistant_turns == 3" are the same inference twice and agree even when both are
    wrong. Recorded at the moment the mask is appended to, these spans are an independent source.
    """
    if length <= 0 or diag_level() != FULL:
        return
    spans.append({"kind": kind, "start": start, "end": start + length})


def log_run_meta(component: str, tokenizer: Any = None, **extra: Any) -> None:
    """Record once per process what the offline decoder cannot recover from the dump alone.

    Without it the decode layer has to be handed $MODEL_PATH by hand and *guess* which ids are
    special; both guesses go stale as soon as the checkpoint moves, and a guessed special-id set
    silently decodes the model's own literal `<|im_start|>` identically to token 151644.
    """
    if diag_level() != FULL or component in _RUN_META_DONE:
        return
    _RUN_META_DONE.add(component)
    meta: dict[str, Any] = {"diag_level": diag_level(), **extra}
    if tokenizer is not None:
        try:
            added = {str(int(v)): str(k) for k, v in (tokenizer.get_added_vocab() or {}).items()}
            ids = {int(i) for i in (tokenizer.all_special_ids or [])} | {int(i) for i in added}
            meta.update(
                tokenizer_path=str(getattr(tokenizer, "name_or_path", "") or ""),
                # The path above is where the tokenizer was loaded from *during the run*, which is
                # not necessarily where it will be afterwards: with N_GPUS_PER_NODE>=4 the launch
                # scripts copy the checkpoint to $LOCAL_RUNTIME_ROOT/models/<tag>-<snap>-<pid> and
                # delete it on exit, so `tokenizer_path` points at a directory that is guaranteed to
                # be gone by the time anyone decodes the dump. Measured: decode_raw.py on a Qwen3-4B
                # run died with `Repo id must be in the form 'repo_name'...` -- transformers falling
                # back to treating the dead path as a Hub id.
                # The launch scripts export the pre-staging path here; it survives the run.
                tokenizer_source=str(os.getenv("DYAD_MODEL_SOURCE_PATH", "") or ""),
                vocab_size=int(getattr(tokenizer, "vocab_size", 0) or 0),
                # Comma / JSON strings, not lists: _serialize truncates lists at DYAD_DIAG_MAX_VALUES
                # and a silently truncated special-id set is worse than none.
                special_ids=",".join(str(i) for i in sorted(ids)),
                added_vocab=json.dumps(added, ensure_ascii=True),
            )
        except Exception as exc:  # noqa: BLE001
            meta["tokenizer_error"] = repr(exc)
    log_event(component, "run_meta", **meta)


def _max_values() -> int:
    try:
        return max(0, int(os.getenv("DYAD_DIAG_MAX_VALUES", str(_DEFAULT_MAX_VALUES))))
    except ValueError:
        return _DEFAULT_MAX_VALUES


def obs_maxlen() -> int:
    """Cap on a dumped env observation. ALFWorld observations run to thousands of characters."""
    try:
        return max(0, int(os.getenv("DIAG_OBS_MAXLEN", str(_DEFAULT_OBS_MAXLEN))))
    except ValueError:
        return _DEFAULT_OBS_MAXLEN


def dump_limit(legacy_env: str | None = None) -> int:
    """How many trajectories this process may dump. See ../dynamic-expa-design/reference/diagnostics_contract.md §4.

    The cap is per process -- agent-loop runs several workers and each keeps its own counter, so the
    volume on disk is limit x worker count. A true global cap needs IPC, which is not worth it; the
    contract instead requires every event to carry `dump_seq` so offline code can count exactly what
    landed. `legacy_env` is the path's deprecated variable, honoured only when explicitly set.
    """
    for name, default in ((legacy_env, None), ("DIAG_DUMP_LIMIT", _DEFAULT_DUMP_LIMIT)):
        if name is None:
            continue
        raw = os.getenv(name)
        if raw is None:
            if default is None:
                continue
            raw = str(default)
        try:
            return max(0, int(raw))
        except ValueError:
            continue
    return _DEFAULT_DUMP_LIMIT


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach()
    result: dict[str, Any] = {
        "type": "tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "numel": value.numel(),
    }
    if value.numel() == 0:
        return result

    result["sample"] = value.reshape(-1)[: _max_values()].to("cpu").tolist()
    try:
        numeric = value.float()
        finite = torch.isfinite(numeric)
        # Count in integers: float32 mean can round below 1 even when all values are finite.
        finite_count = finite.sum().item()
        result["finite_fraction"] = finite_count / finite.numel()
        result["nonzero_fraction"] = (numeric != 0).float().mean().item()
        if finite.any():
            finite_values = numeric[finite]
            result.update(
                {
                    "min": finite_values.min().item(),
                    "max": finite_values.max().item(),
                    "mean": finite_values.mean().item(),
                }
            )
    except Exception:
        pass
    return result


def _serialize(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return repr(value)[:512]
    if torch.is_tensor(value):
        return _tensor_summary(value)
    if isinstance(value, np.ndarray):
        items = value.reshape(-1)
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sample": [_serialize(item, depth + 1) for item in items[: _max_values()]],
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        items = list(value.items())
        limit = max(_max_values(), 16)
        result = {str(k): _serialize(v, depth + 1) for k, v in items[:limit]}
        if len(items) > limit:
            result[TRUNCATION_KEY] = len(items) - limit
        return result
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        limit = _max_values()
        result = [_serialize(item, depth + 1) for item in items[:limit]]
        if len(items) > limit:
            result.append({TRUNCATION_KEY: len(items) - limit})
        return result
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)[:1024]


def log_event(component: str, event: str, **data: Any) -> None:
    """Append one diagnostic event. Failures never interrupt training."""
    global _WARNED
    if not _enabled():
        return
    try:
        # On the normal path the trainer exports DYAD_DIAG_DIR pointing at this run's directory
        # (outputs/<algo>/<env>/<run>/, see experiments/shared/train_eval/scripts/run_layout.sh).
        # This fallback is only reached when log_event is called outside a trainer; _unscoped keeps
        # those events visibly apart from a real run directory.
        from agent_system.utils.artifact_paths import artifact_root

        configured = os.getenv("DYAD_DIAG_DIR")
        root = Path(configured) if configured else artifact_root() / "outputs" / "_unscoped"
        safe_component = "".join(c if c.isalnum() or c in "-_" else "_" for c in component)
        path = root / f"{safe_component}_pid{os.getpid()}.jsonl"
        record = {
            "time": time.time(),
            "component": component,
            "event": event,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            **{key: _serialize(value) for key, value in data.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=True, allow_nan=True)
        with _WRITE_LOCK, path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
    except Exception as exc:
        if not _WARNED:
            _WARNED = True
            print(f"[Dyad diagnostics] write failure, continuing training: {exc!r}")


def batch_contract(batch: Any) -> dict[str, Any]:
    """Summarize DataProto/TensorDict/dict fields relevant to Dyad alignment."""
    tensor_batch = getattr(batch, "batch", batch)
    keys = list(tensor_batch.keys()) if hasattr(tensor_batch, "keys") else []
    summary: dict[str, Any] = {
        "keys": sorted(str(key) for key in keys),
        "dyad_fields_present": {key: key in keys for key in _DYAD_KEYS},
    }
    for key in (
        "responses",
        "response_mask",
        "response_dyad",
        "tool_mask",
        "dyad_action_mask",
        "seq_mask",
        "old_log_probs",
        "ref_log_prob",
        "advantages",
        "token_level_scores",
        "token_level_rewards",
        "rollout_log_probs",
    ):
        if key not in keys:
            continue
        value = tensor_batch[key]
        summary[key] = _tensor_summary(value) if torch.is_tensor(value) else _serialize(value)
        if torch.is_tensor(value) and value.ndim >= 2:
            try:
                reduce_dims = tuple(range(1, value.ndim))
                summary[key]["per_sample_sum"] = value.float().sum(dim=reduce_dims).to("cpu").tolist()
                summary[key]["per_sample_nonzero"] = (value != 0).sum(dim=reduce_dims).to("cpu").tolist()
                if key.endswith("mask"):
                    summary[key]["per_sample_active"] = value.bool().sum(dim=reduce_dims).to("cpu").tolist()
            except Exception:
                pass
    return summary


def module_parameter_summary(module: Any) -> dict[str, Any]:
    if module is None:
        return {"exists": False}
    result: dict[str, Any] = {"exists": True, "class": type(module).__name__}
    try:
        parameters = list(module.named_parameters())
        result["parameter_names"] = [name for name, _ in parameters]
        result["parameter_count"] = sum(param.numel() for _, param in parameters)
        result["parameters"] = {
            name: {
                "value": _tensor_summary(param),
                "grad": _tensor_summary(param.grad) if param.grad is not None else None,
                "requires_grad": param.requires_grad,
            }
            for name, param in parameters
        }
    except Exception as exc:
        result["summary_error"] = repr(exc)
    return result


def action_head_summary(weight) -> str:
    """A dtype-tolerant summary of a materialised `action_head`, as `rows=A norm=.. mean=.. absmax=..`.

    Both sides print this after every re-derive so that "the head that sampled" and "the head that
    computed log-probs" can be compared after the fact instead of argued about. They are supposed to
    describe the same distribution (AGENTS.md section 1); with a trainable projector that stops being
    automatic, and a divergence produces no error at all -- just a policy gradient computed against
    a distribution nobody sampled from.

    Statistics rather than a hash, because a hash cannot answer this question. The trainer derives
    the head from an fp32 `lm_head` and the engine from vLLM's bf16 copy, so the two are *never*
    bit-identical -- a hash comparison would report a divergence on every step of every run,
    including runs that are perfectly correct. Summary statistics agree to bf16 precision when the
    heads agree and separate by orders of magnitude when they do not, which is the distinction worth
    reporting.
    """
    import torch

    with torch.no_grad():
        data = weight.detach().float()
        return (
            f"rows={data.shape[0]} norm={float(data.norm()):.6g} "
            f"mean={float(data.mean()):.6g} absmax={float(data.abs().max()):.6g}"
        )
