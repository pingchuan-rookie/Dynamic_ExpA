"""Resolve inference-only evaluation settings from the public experiment selection."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parents[2]

# Compatibility is limited to controls that change inference or evaluation.
OVERRIDES = {
    "data.val_max_samples": ("limit", int),
    "data.val_batch_size": ("concurrency", int),
    "data.max_prompt_length": ("prompt_length", int),
    "data.max_response_length": ("max_tokens", int),
    "actor_rollout_ref.rollout.val_kwargs.n": ("repeats", int),
    "actor_rollout_ref.rollout.val_kwargs.temperature": ("temperature", float),
    "actor_rollout_ref.rollout.val_kwargs.top_p": ("top_p", float),
    "actor_rollout_ref.rollout.tensor_model_parallel_size": ("tensor_parallel_size", int),
    "actor_rollout_ref.rollout.gpu_memory_utilization": ("gpu_memory_utilization", float),
    "actor_rollout_ref.rollout.max_model_len": ("context_length", int),
    "algorithm.step_rollout.history_length": ("history_length", int),
    "algorithm.step_rollout.max_steps": ("max_steps", int),
}


def from_environment(
    env: dict[str, str],
    benchmark: str,
    overrides: Sequence[str] = (),
    *,
    dataset: str | None = None,
    task_config: dict | None = None,
) -> dict[str, Any]:
    """Read resolved model/data settings; never construct actor, optimizer or trainer config."""
    task = task_config or {}
    prompt_length = int(task.get("max_prompt_length", env.get("MAX_PROMPT_LENGTH", 8192)))
    max_tokens = int(task.get("max_tokens", env.get("MAX_RESPONSE_LENGTH", 2048)))
    tool_name = "calc" if benchmark == "gsm8k" else benchmark
    tool_path = env.get("DIVE_TOOL_CONFIG_PATH") if benchmark == "dive" else env.get("TOOL_CONFIG_PATH")
    if benchmark in {"t2bench", "swebench_verified"}:
        tool_path = str(Path(env["RUN_DIR"]) / "environment_tool.yaml")
    model_name = env.get("MODEL_NAME") or task.get("model")
    if not model_name:
        raise ValueError("Evaluation requires a model name")
    algorithm = env.get("RUN_ALGO", task.get("algorithm", env.get("RUN_ALGO_BASE", "grpo_react")))
    config = {
        "version": 1,
        "backend": "inference_api",
        "benchmark": benchmark,
        "algorithm": algorithm,
        "debug": env.get("RUN_IS_DEBUG") == "1",
        "task_prompt_protocol": "environment_react_v3",
        "dataset_identity": {
            name: json.loads(env.get(key, "{}"))
            for name, key in (
                ("alfworld_eval_data", "ALFWORLD_EVAL_DATA"),
                ("webshop_data", "WEBSHOP_DATA"),
                ("codegym_coverage", "CODEGYM_COVERAGE"),
                ("dive_data", "DIVE_DATA"),
            )
        },
        "action_interface": "dyad"
        if env.get("RUN_ACTION_INTERFACE") == "dyad" or algorithm.startswith("dyad")
        else "text",
        "model": model_name,
        "model_path": env.get("MODEL_PATH", model_name),
        "checkpoint": env.get("EVAL_CHECKPOINT") or env.get("RESUME_CKPT"),
        "model_config": env.get("EVAL_MODEL_CONFIG"),
        "model_settings": {
            key: value
            for key, value in env.items()
            if key.startswith("DYAD_ENCODER_")
            or key
            in {
                "MODEL_NAME",
                "MODEL_PATH",
                "DYAD_ADV_ESTIMATOR",
                "DYAD_TRAINING_SCHEDULE",
                "DYAD_PROJECTOR_LR",
                "DYAD_VALUES",
                "DYAD_ACTION_YAML",
            }
        },
        "projector_init": env.get("DYAD_ENCODER_PROJECTOR_INIT"),
        "schema_name": env.get("DYAD_ACTION_YAML"),
        "dataset": dataset or env.get("TEST_DATA"),
        "task_config": task,
        "tool_config": tool_path or str(PROJECT / f"agent_system/environments/configs/{tool_name}_tool.yaml"),
        "output_dir": env["RUN_DIR"],
        "validation_dir": env["VALIDATION_DATA_DIR"],
        "api_url": env.get("EVAL_API_URL"),
        "api_key_env": env.get("EVAL_API_KEY_ENV", "EVAL_API_KEY"),
        "served_model": env.get("EVAL_SERVED_MODEL", "evaluation-model"),
        "context_length": int(task.get("context_length", env.get("MAX_MODEL_LEN", prompt_length + max_tokens))),
        "prompt_length": prompt_length,
        "max_tokens": max_tokens,
        "max_steps": int(task.get("max_assistant_turns", task.get("max_steps", env.get("MAX_ASSISTANT_TURNS", 50)))),
        "history_length": int(task.get("history_length", env.get("STEP_HISTORY_LENGTH", 2))),
        "concurrency": int(task.get("max_concurrency", env.get("EVAL_CONCURRENCY", env.get("VAL_BATCH_SIZE", 1)))),
        "repeats": int(env.get("VAL_N", 1)),
        "seed": int(task.get("seed", env.get("EVAL_SEED", 42))),
        "limit": int(env.get("VAL_MAX_SAMPLES", -1)),
        "temperature": float(task.get("temperature", env.get("VAL_TEMPERATURE", 0.0))),
        "top_p": float(env.get("VAL_TOP_P", 1.0)),
        "thinking": task.get("thinking", "off"),
        "tensor_parallel_size": int(env.get("EVAL_TENSOR_PARALLEL_SIZE", env.get("ROLLOUT_TP_SIZE", 1))),
        "gpu_memory_utilization": float(env.get("GPU_MEM_UTIL", 0.7)),
        "action_capacity": int(env.get("DYAD_ACTION_CAPACITY", env.get("DYAD_CODEGYM_ACTION_CAPACITY", 256))),
        "encoder_device": env.get("EVAL_ENCODER_DEVICE", "cpu"),
        "ready_timeout": int(env.get("EVAL_READY_TIMEOUT", 3600)),
        "timeout": float(task.get("timeout", env.get("EVAL_REQUEST_TIMEOUT", 600))),
    }
    for item in overrides:
        field, separator, raw = item.lstrip("+").partition("=")
        if field == "data.apply_chat_template_kwargs.enable_thinking":
            if raw.lower() not in {"false", "true"}:
                raise ValueError("enable_thinking must be true or false")
            config["thinking"] = "on" if raw.lower() == "true" else "off"
        elif field in OVERRIDES and separator:
            name, cast = OVERRIDES[field]
            config[name] = cast(raw)
        else:
            raise ValueError(f"Unsupported standalone evaluation override: {item}; use inference/evaluation controls")
    validate(config)
    return config


def validate(config: dict[str, Any]) -> None:
    if config.get("api_url"):
        url = urlsplit(config["api_url"])
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path.rstrip("/") != "/v1"
        ):
            raise ValueError("api_url must be an HTTP(S) /v1 endpoint without embedded credentials")
    import math

    for key in ("temperature", "top_p", "gpu_memory_utilization", "timeout"):
        if not math.isfinite(config[key]):
            raise ValueError(f"{key} must be finite")
    if (
        not 0 <= config["temperature"] <= 2
        or not 0 < config["top_p"] <= 1
        or not 0 < config["gpu_memory_utilization"] < 1
        or config["timeout"] <= 0
    ):
        raise ValueError("Invalid sampling parameters, memory utilization or timeout")
    for key in (
        "context_length",
        "prompt_length",
        "max_tokens",
        "max_steps",
        "concurrency",
        "repeats",
        "tensor_parallel_size",
        "action_capacity",
        "ready_timeout",
    ):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["history_length"] < 0 or config["limit"] == 0 or config["limit"] < -1:
        raise ValueError("Invalid history_length or sample limit")
    if config["prompt_length"] + config["max_tokens"] > config["context_length"]:
        raise ValueError("Prompt plus output budget exceeds inference context_length")
    if config["benchmark"] in {"t2bench", "swebench_verified"} and config["repeats"] != 1:
        raise ValueError("Official evaluation trials come from task_config.num_trials; repeats must be 1")
    if not config["dataset"]:
        raise ValueError("Evaluation requires an explicit test dataset")


def command(config: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "agent_system.evaluation.runner",
        "--config-json",
        json.dumps(config, ensure_ascii=False, separators=(",", ":")),
    ]
