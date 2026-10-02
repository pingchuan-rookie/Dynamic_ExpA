"""Runtime-only model identity and chat-template thinking policy.

Qwen3.5 policies always use a closed thinking prefix, including renamed local
checkpoints. Identification reads local metadata only, never remote HF resources.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from os import PathLike
from pathlib import Path
import re
from typing import Any


_QWEN35 = re.compile(r"qwen3[._-]?5(?=$|[^0-9])", re.IGNORECASE)
_IDENTITY_KEYS = (
    "model_type", "architectures", "text_config", "hf_config", "config",
    "model", "path", "tokenizer_path", "name_or_path", "_name_or_path",
    "MODEL_PATH", "MODEL_NAME", "tokenizer", "init_kwargs",
)


def is_qwen35_model(*sources: Any) -> bool:
    """Recognize Qwen3.5 from names, local config files, configs or tokenizers."""
    seen: set[int] = set()

    def matches(source: Any) -> bool:
        if source is None or id(source) in seen:
            return False
        seen.add(id(source))
        if isinstance(source, (str, PathLike)):
            path = Path(source).expanduser()
            try:
                paths = [path / "config.json", path / "tokenizer_config.json"] if path.is_dir() else [path]
                for config_path in paths:
                    if config_path.name not in ("config.json", "tokenizer_config.json") or not config_path.is_file():
                        continue
                    metadata = json.loads(config_path.read_text())
                    if isinstance(metadata, Mapping):
                        # An explicit local architecture is authoritative over directory names
                        # and stale tokenizer/source metadata from another model family.
                        family = {key: metadata[key] for key in ("model_type", "architectures", "text_config")
                                  if metadata.get(key)}
                        if family:
                            return matches(family)
                    if matches(metadata):
                        return True
            except (OSError, ValueError):
                pass
            # Never inspect arbitrary parent names (e.g. pytest's test_qwen35_identity).
            # HF snapshot directories are the exception: the models-- component is a
            # structured repository identity rather than a user-chosen ancestor label.
            model_name = path.name
            if model_name in ("config.json", "tokenizer_config.json"):
                model_name = path.parent.name
            if _QWEN35.search(model_name):
                return True
            for parent in path.parents:
                if parent.name == "snapshots" and parent.parent.name.startswith("models--"):
                    return bool(_QWEN35.search(parent.parent.name.split("--", 2)[-1]))
            return False
        if isinstance(source, Mapping):
            return any(matches(source.get(key)) for key in _IDENTITY_KEYS)
        if isinstance(source, (list, tuple)):
            return any(matches(item) for item in source)
        return any(matches(getattr(source, key, None)) for key in _IDENTITY_KEYS)

    return any(matches(source) for source in sources)


def resolve_chat_template_kwargs(
    kwargs: Mapping[str, Any] | None = None,
    *,
    model: Any = None,
    tokenizer: Any = None,
    processor: Any = None,
) -> dict[str, Any]:
    """Copy kwargs and enforce Qwen3.5 no-thinking without changing other models.

    Explicit true values are configuration errors, not silently overridden.
    Other template options and the caller's mapping are preserved.
    """
    result = dict(kwargs or {})
    if is_qwen35_model(model, tokenizer, processor):
        value = result.get("enable_thinking")
        enabled = value is True or value == 1 or (
            isinstance(value, str) and value.strip().lower() in {"true", "on", "1", "yes"}
        )
        if enabled:
            raise ValueError("Qwen3.5 requires no-thinking: enable_thinking=True is not supported")
        result["enable_thinking"] = False
    return result
