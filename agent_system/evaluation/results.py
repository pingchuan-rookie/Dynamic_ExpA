"""Read standalone inference results through the existing benchmark score validators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def verify_command(run: Path, config: dict) -> None:
    command = json.loads((run / "command.json").read_text())
    if len(command) != 5 or command[1:4] != ["-m", "agent_system.evaluation.runner", "--config-json"]:
        raise ValueError("Expected the standalone evaluation command")
    requested = json.loads(command[-1])
    for key, value in requested.items():
        if key == "api_url" and value is None:
            continue  # A job-owned service receives its ephemeral port after preparation.
        if config.get(key) != value:
            raise ValueError(f"Saved inference evaluation changed after launch: {key}")


def load_result(run: Path, config: dict) -> dict[str, Any]:
    from agent_system.policies.dyad.inference.source import file_digest, source_metadata
    from experiments.shared.main_experiments import environment_results as legacy

    verify_command(run, config)
    benchmark = config["benchmark"]
    allowed_cap = {"alfworld": 134, "webshop": 100}.get(benchmark, -1)
    if config.get("debug") or config["limit"] not in (-1, allowed_cap) or not config.get("code_sha256"):
        raise ValueError("Debug, limited, or unversioned evaluation cannot populate a main table")
    for path, expected in config["dataset_sha256"].items():
        if file_digest(path) != expected:
            raise ValueError("Evaluation dataset changed since the run")
    endpoint = config.get("endpoint_identity", {})
    identity = endpoint.get("identity", {})
    expected_root = Path(config.get("checkpoint") or config.get("projector_init") or config["model_path"]).resolve()
    if Path(endpoint.get("root", "")).resolve() != expected_root or identity.get("restoration_complete") is not True:
        raise ValueError("Evaluation endpoint source/restoration evidence differs")
    checkpoint = config.get("checkpoint")
    method = config["algorithm"] if checkpoint else "zero-shot"
    evidence = [run / "run.status", run / "command.json", run / "resolved_config.json", run / "evaluation_config.json"]
    normalized = {
        **config.get("preparation_evidence", {}),
        **config.get("dataset_identity", {}),
        "backend": "inference_api",
        "inference_configuration": config,
        "benchmark": benchmark,
        "planned_episodes": config["planned_episodes"],
        "model": config["model"],
        "action_interface": config["action_interface"],
    }
    if config["action_interface"] == "dyad":
        if not checkpoint or config.get("projector_init"):
            raise ValueError("Main-table Dyad evaluation requires a complete Agentic RL checkpoint")
        from types import SimpleNamespace

        source = source_metadata(
            SimpleNamespace(checkpoint=checkpoint, projector_init=None, model_config=config.get("model_config"))
        )
        if identity.get("source_identity_sha256") != source["identity_sha256"]:
            raise ValueError("Dyad endpoint source hash differs from the selected checkpoint")
        from agent_system.policies.dyad.inference.checkpoint import verify_native_artifact

        manifest = verify_native_artifact(run / "native_restore", source=source)
        if (
            identity.get("projector_sha256") != manifest["projector_sha256"]
            or identity.get("policy_sha256") != manifest["policy_sha256"]
        ):
            raise ValueError("Endpoint weights differ from the audited restored model")
        normalized["native_source"] = source
        evidence += [run / "native_restore/manifest.json", Path(source["model_config"])]
    else:
        policy_path = Path(identity.get("policy_path", ""))
        hashes = identity.get("weight_files_sha256", {})
        if not hashes or any(file_digest(policy_path / name) != expected for name, expected in hashes.items()):
            raise ValueError("Text endpoint lacks matching loaded weight hashes")
    if checkpoint:
        from agent_system.inference.policy_identity import training_identity
        from agent_system.policies.dyad.checkpoint_compat import read_saved_model_config

        saved = read_saved_model_config(config.get("model_config") or Path(checkpoint).parent / "model_config.json")
        expected = training_identity(saved)["training_method"]
        if method == "dyad":
            method = expected
        if method != expected or saved["model"]["MODEL_NAME"] != config["model"]:
            raise ValueError("Evaluation method/model differs from checkpoint identity")
    # Old score validators name decoding controls after their former Hydra fields.
    # These aliases describe identical measured settings; they are never executed.
    decoding = {
        "data.val_files": config["dataset"],
        "data.val_max_samples": config["limit"],
        "data.max_prompt_length": config["prompt_length"],
        "data.max_response_length": config["max_tokens"],
        "data.apply_chat_template_kwargs.enable_thinking": config["thinking"] == "on",
        "data.filter_overlong_prompts": False,
        "data.truncation": "error",
        "data.seed": config["seed"],
        "actor_rollout_ref.rollout.val_kwargs.n": config["repeats"],
        "actor_rollout_ref.rollout.val_kwargs.temperature": config["temperature"],
        "actor_rollout_ref.rollout.val_kwargs.top_p": config["top_p"],
        "actor_rollout_ref.rollout.val_kwargs.do_sample": config["temperature"] > 0,
        "actor_rollout_ref.rollout.max_model_len": config["context_length"],
        "actor_rollout_ref.rollout.multi_turn.max_assistant_turns": config["max_steps"],
        "algorithm.step_rollout.history_length": config["history_length"],
        "algorithm.step_rollout.max_steps": config["max_steps"],
        "algorithm.step_rollout.profile": f"{benchmark}_native_v2",
        "trainer.nnodes": 1,
    }
    if benchmark in {"alfworld", "webshop"}:
        scores, protocol, paths = legacy._standard(run, benchmark, normalized, decoding)
    elif benchmark == "t2bench":
        normalized["tau"] = config["task_config"]
        scores, protocol, paths = legacy._tau(run, normalized, decoding)
    elif benchmark == "swebench_verified":
        normalized["swebench"] = config["task_config"]
        scores, protocol, paths = legacy._swebench(run, normalized, decoding)
    else:
        raise ValueError("Unsupported main-table benchmark")
    if not scores:
        raise ValueError("Empty evaluation cannot produce a score")
    import math

    return {
        "benchmark": benchmark,
        "model": config["model"],
        "checkpoint": checkpoint,
        "method": method,
        "model_path": str(expected_root),
        "score_percent": 100 * math.fsum(scores) / len(scores),
        "num_episodes": len(scores),
        "protocol": protocol,
        "evidence": [str(path) for path in [*evidence, *paths]],
    }
