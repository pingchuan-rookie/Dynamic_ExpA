"""Run evaluation decisions over HTTP, without verl trainers or policy workers."""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_system.environments.step_session import make_step_session
from agent_system.evaluation.config import validate


def load_dataset(config: dict) -> Any:
    from omegaconf import OmegaConf

    benchmark = config["benchmark"]
    if benchmark == "t2bench":
        from agent_system.environments.backends.tau.dataset import TauEvaluationDataset

        return TauEvaluationDataset(config["dataset"], config=OmegaConf.create({"tau": config["task_config"]}))
    if benchmark == "swebench_verified":
        from agent_system.environments.backends.swebench.dataset import SwebenchEvaluationDataset

        return SwebenchEvaluationDataset(config["dataset"], config={"swebench": config["task_config"]})
    import pyarrow.parquet as pq

    paths = config["dataset"] if isinstance(config["dataset"], list) else [config["dataset"]]
    rows = [row for path in paths for row in pq.read_table(path).to_pylist()]
    limit = config["limit"]
    return rows if limit < 0 else rows[:limit]


def load_tool(config: dict):
    import yaml

    document = yaml.safe_load(Path(config["tool_config"]).read_text())
    if len(document.get("tools", [])) != 1:
        raise ValueError("Environment evaluation requires exactly one environment tool")
    spec = document["tools"][0]
    module, _, name = spec["class_name"].rpartition(".")
    settings = copy.deepcopy(spec["config"])
    settings["pool_size"] = config["concurrency"]
    return getattr(importlib.import_module(module), name)(settings)


def request_actions(session: Any, config: dict) -> tuple[list, dict]:
    """Keep the training prompt/tools and task-local candidate definitions identical."""
    args = {
        "prompt_limit": config["prompt_length"],
        "strip_thinking_prefill": True,
        "template_tools": session.action_tools is not None,
    }
    tools = session.action_tools or []
    if session.environment == "t2bench":
        tools = session.context["action_tools"]
        args["protocol"] = "tau_json"
    elif session.environment == "codegym" and config["action_interface"] == "dyad":
        from agent_system.policies.dyad.actions.codegym_tasks import task_schema

        args["schema"] = task_schema(session.initial_messages, session.reset_spec["env_str"])
    elif session.environment in {"alfworld", "webshop", "gsm8k"} and config["action_interface"] == "dyad":
        schema = config.get("schema_name")
        if not schema:
            raise ValueError("Dyad environment evaluation requires its resolved action schema")
        args["schema_name"] = schema
    return tools, args


async def evaluate_episode(row: dict, config: dict, tool: Any, tokenizer: Any, client: Any, trial: int = 0) -> dict:
    settings = copy.deepcopy(
        (row.get("tools_kwargs") or (row.get("extra_info") or {}).get("tools_kwargs", {})).get(tool.name, {})
    )
    if config["benchmark"] == "gsm8k":
        settings["task_description"] = (row.get("extra_info") or {}).get("question")
    session = make_step_session(tool, uuid4().hex, settings, row.get("raw_prompt", row.get("prompt", [])))
    decisions = []
    reason = "env_done"
    try:
        await session.reset()
        tools, args = request_actions(session, config)
        seed = session.context.get("generation_seed", config["seed"] + trial)
        maximum = min(config["max_tokens"], getattr(session, "max_tokens", None) or config["max_tokens"])
        for index in range(config["max_steps"]):
            if session.done:
                break
            messages = session.messages(config["history_length"])
            payload = {
                "model": config["served_model"],
                "messages": messages,
                "tools": tools,
                "args": args,
                "max_tokens": maximum,
                "temperature": config["temperature"],
                "top_p": config["top_p"],
                "seed": seed,
                "chat_template_kwargs": {"enable_thinking": config["thinking"] == "on"},
            }
            response = await client.post(config["api_url"].rstrip("/") + "/chat/completions", json=payload)
            if response.status_code == 400 and session.environment == "swebench_verified" and decisions:
                detail = response.json().get("detail", "")
                if isinstance(detail, str) and detail.startswith("Prompt budget exceeded:"):
                    reason = "context_budget_exhausted"
                    break
            response.raise_for_status()
            output = response.json()
            ids = output.get("token_ids")
            if ids is None:
                message = output["choices"][0]["message"]
                if message.get("tool_calls") and not output.get("raw_text"):
                    raise ValueError(
                        "Native environment evaluation needs raw_text/token_ids; use the shared inference service"
                    )
                # Standard vLLM endpoints omit token IDs; decoding the returned native text
                # preserves environment parsing, and usage still comes from the endpoint.
                ids = tokenizer.encode(
                    output.get("raw_text") or output["choices"][0]["message"].get("content") or "",
                    add_special_tokens=False,
                )
            if output["usage"]["completion_tokens"] > maximum:
                raise ValueError("Endpoint exceeded the per-decision generation budget")
            evidence = output.get("dyad") or {}
            selections = None
            if config["action_interface"] == "dyad":
                if evidence.get("action_interface") != "dyad" or not isinstance(evidence.get("action_content"), dict):
                    raise ValueError("Dyad evaluation requires the real sampled action trace")
                if session.environment in {"dive", "swebench_verified", "t2bench"}:
                    selections = evidence["selections"]
            before = session.anchor
            raw_text = tokenizer.decode(ids, skip_special_tokens=True)
            transition = await session.execute(raw_text, ids, tokenizer, selections)
            if not math.isfinite(transition.reward):
                raise ValueError("Environment returned a nonfinite reward")
            reason = "env_done" if session.done else "max_steps"
            decisions.append(
                {
                    "step_index": index,
                    "messages": messages,
                    "prompt_ids": output.get("prompt_token_ids"),
                    "response_ids": ids,
                    "response": raw_text,
                    "anchor": before,
                    "action": transition.action,
                    "action_valid": transition.valid,
                    "executed": transition.executed,
                    "reward": transition.reward,
                    "usage": output["usage"],
                    "dyad": evidence,
                    "finish_reason": output["choices"][0]["finish_reason"],
                }
            )
        final_reward, fields = await session.finalize(reason)
        if "episode_result" in fields:
            audits = []
            for index, decision in enumerate(decisions):
                trace = decision["dyad"].get("action_content")
                raw = decision["action"].get("raw_text", decision["response"])
                audits.append(
                    {
                        "assistant_turn": index + 1,
                        "schema_hash": session.context.get("schema_hash"),
                        "submitted": True,
                        "submitted_raw_text": raw,
                        "raw_text": raw,
                        "action_content": trace,
                        "token_ids": decision["response_ids"],
                        "raw_token_ids": trace.get("raw_token_ids") if trace else None,
                        "policy_trace": decision["dyad"].get("policy_trace"),
                        "selections": decision["action"].get("selected_actions"),
                    }
                )
            fields["episode_result"]["action_audit"] = audits
        evidence = session.evidence()
        reward = math.fsum([final_reward, *(d["reward"] for d in decisions)])
        result = {
            "benchmark": config["benchmark"],
            "uid": str(row.get("uid", row.get("index", ""))),
            "trial": trial,
            "seed": seed,
            "termination_reason": reason,
            "decisions": decisions,
            "environment_evidence": evidence,
            "metric_valid": evidence.get("metric_valid", True),
            "episode_reward": reward,
            **fields,
        }
        if result["metric_valid"] is not True:
            result["episode_reward"] = None
        return result
    finally:
        await asyncio.shield(session.close())


async def run(config: dict) -> dict:
    import httpx
    import ray
    from transformers import AutoTokenizer

    dataset = load_dataset(config)
    if not len(dataset):
        raise ValueError("Evaluation task selection is empty")
    tokenizer_path = config["model_path"]
    if config.get("checkpoint") and config["action_interface"] == "text":
        tokenizer_path = str(Path(config["checkpoint"]) / "actor/huggingface")
    config["tokenizer_path"] = tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tool = load_tool(config)
    owned_ray = not ray.is_initialized()
    if owned_ray:
        ray.init(
            address="local",
            num_gpus=0,
            include_dashboard=False,
            num_cpus=int(os.environ.get("RAY_NUM_CPUS") or os.environ.get("EFFECTIVE_CPUS") or os.cpu_count() or 1),
        )
    results = []
    config["planned_episodes"] = len(dataset) * config["repeats"]
    destination = Path(config["validation_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(config["concurrency"])
    headers = {"Authorization": "Bearer " + os.environ.get(config["api_key_env"], "EMPTY")}
    try:
        async with httpx.AsyncClient(timeout=config["timeout"], trust_env=False, headers=headers) as client:
            response = await client.get(config["api_url"].rstrip("/") + "/models")
            response.raise_for_status()
            models = [m for m in response.json()["data"] if m["id"] == config["served_model"]]
            if len(models) != 1:
                raise ValueError("Endpoint does not advertise the selected model")
            identity = models[0].get("identity", {})
            if config["action_interface"] == "dyad" and (
                identity.get("action_interface") != "dyad" or identity.get("restoration_complete") is not True
            ):
                raise ValueError("Endpoint has not restored a complete Dyad model")
            expected_root = str(
                Path(config.get("checkpoint") or config.get("projector_init") or config["model_path"]).resolve()
            )
            if Path(models[0].get("root", "")).resolve() != Path(expected_root):
                raise ValueError("Endpoint weights differ from the explicitly selected evaluation source")
            if config["action_interface"] == "dyad":
                from types import SimpleNamespace

                from agent_system.policies.dyad.inference.source import source_metadata

                expected = source_metadata(
                    SimpleNamespace(
                        checkpoint=config.get("checkpoint"),
                        projector_init=config.get("projector_init"),
                        model_config=config.get("model_config"),
                    )
                )
                if identity.get("source_identity_sha256") != expected["identity_sha256"]:
                    raise ValueError("Endpoint checkpoint/projector identity mismatch")
            config["endpoint_identity"] = models[0]
            write_json(Path(config["output_dir"]) / "resolved_config.json", config)

            async def one(index, trial):
                async with semaphore:
                    result = await evaluate_episode(dataset[index], config, tool, tokenizer, client, trial)
                    write_json(destination / f"episode_{index}_{trial}.json", result)
                    results.append(result)
                    print(f"[evaluation] completed {len(results)}/{len(dataset) * config['repeats']}", flush=True)

            tasks = [
                asyncio.create_task(one(index, trial))
                for index in range(len(dataset))
                for trial in range(config["repeats"])
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        try:
            await tool.pool.shutdown()
        finally:
            if owned_ray:
                ray.shutdown()
    if config["benchmark"] in {"t2bench", "swebench_verified"}:
        module = "tau" if config["benchmark"] == "t2bench" else "swebench"
        summarize = importlib.import_module(f"agent_system.environments.backends.{module}.metrics").summarize_episodes
        summary = summarize(dataset.planned_episodes, [r["episode_result"] for r in results])
    else:
        valid = [r for r in results if r["metric_valid"]]
        complete = len(valid) == len(dataset) * config["repeats"]
        scores = [
            r["environment_evidence"].get("task_score")
            if config["benchmark"] == "webshop"
            else float(r["environment_evidence"].get("won", False))
            if config["benchmark"] == "alfworld"
            else r["episode_reward"]
            for r in valid
        ]
        summary = {
            "complete": complete,
            "planned": len(dataset) * config["repeats"],
            "completed": len(results),
            "score": math.fsum(scores) / len(scores) if complete else None,
            "benchmark": config["benchmark"],
        }
    write_json(destination / "summary.json", summary)
    if config["benchmark"] in {"alfworld", "webshop"}:
        with (destination / "episodes.jsonl").open("w") as stream:
            for result in results:
                evidence = result["environment_evidence"]
                stream.write(
                    json.dumps(
                        {
                            **evidence,
                            "won": float(evidence.get("won", False)),
                            "score": evidence.get("task_score"),
                            "trial": result["trial"],
                        }
                    )
                    + "\n"
                )
    elif config["benchmark"] in {"t2bench", "swebench_verified"}:
        (destination / "official").mkdir(exist_ok=True)
        write_json(destination / "official/summary.json", summary)
    if not summary["complete"]:
        raise RuntimeError("Evaluation contains unscored or missing episodes; see retained results")
    return summary


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def service_command(config: dict, port: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "agent_system.inference.server",
        "--model",
        config["served_model"],
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(config["tensor_parallel_size"]),
        "--max-model-len",
        str(config["context_length"]),
        "--max-num-seqs",
        str(config["concurrency"]),
        "--gpu-memory-utilization",
        str(config["gpu_memory_utilization"]),
        "--action-capacity",
        str(config["action_capacity"]),
        "--encoder-device",
        config["encoder_device"],
        "--restore-dir",
        str(Path(config["output_dir"]) / "native_restore"),
    ]
    if config.get("checkpoint"):
        command += ["--checkpoint", config["checkpoint"]]
    elif config.get("projector_init"):
        command += ["--projector-init", config["projector_init"]]
    else:
        command += ["--model-path", config["model_path"]]
    if config.get("model_config"):
        command += ["--model-config", config["model_config"]]
    return command


def execute(config: dict) -> int:
    import httpx

    validate(config)
    directory = Path(config["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "evaluation_config.json").exists():
        raise ValueError("Evaluation output already exists; choose a new run directory")
    from agent_system.policies.dyad.inference.source import file_digest

    project = Path(__file__).resolve().parents[2]
    roots = (project / "agent_system", project / "experiments/shared/train_eval")
    config["code_sha256"] = {
        str(path.relative_to(project)): file_digest(path)
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".sh", ".yaml"}
    }
    data_paths = config["dataset"] if isinstance(config["dataset"], list) else [config["dataset"]]
    config["dataset_sha256"] = {str(Path(path).resolve()): file_digest(path) for path in data_paths}
    if config.get("projector_init") and not config.get("model_config"):
        model_config = directory / "serving_model_config.json"
        settings = config["model_settings"]
        write_json(
            model_config,
            {
                "version": 1,
                "benchmark": config["benchmark"],
                "algo": "dyad",
                "action_interface": "dyad",
                "adv_estimator": settings.get("DYAD_ADV_ESTIMATOR", "grpo"),
                "model": settings,
            },
        )
        config["model_config"] = str(model_config)
    write_json(directory / "evaluation_config.json", config)
    from agent_system.evaluation.config import command as evaluation_command

    write_json(directory / "command.json", evaluation_command(config))
    process, log = None, None
    rc = 1
    try:
        if not config["api_url"]:
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            config["api_url"] = f"http://127.0.0.1:{port}/v1"
            command = service_command(config, port)
            write_json(directory / "service_command.json", command)
            log = (directory / "serve.log").open("w")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline, report = time.monotonic() + config["ready_timeout"], time.monotonic()
            with httpx.Client(timeout=5, trust_env=False) as client:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"Inference service exited ({process.returncode}); inspect {directory / 'serve.log'}"
                        )
                    try:
                        response = client.get(config["api_url"] + "/models")
                        response.raise_for_status()
                        break
                    except httpx.HTTPError as exc:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"Inference service readiness timeout; inspect {directory / 'serve.log'}"
                            ) from exc
                        if time.monotonic() >= report:
                            print(f"[evaluation] waiting for inference service; {directory / 'serve.log'}", flush=True)
                            report = time.monotonic() + 30
                        time.sleep(0.5)
        summary = asyncio.run(run(config))
        print(json.dumps(summary, ensure_ascii=False))
        rc = 0
        return rc
    finally:
        (directory / "run.status").write_text(f"exit_code={rc}\n")
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                process.wait()
        if log is not None:
            log.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--config-json")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text() if args.config else args.config_json)

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    previous = {number: signal.signal(number, interrupted) for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        return execute(config)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
