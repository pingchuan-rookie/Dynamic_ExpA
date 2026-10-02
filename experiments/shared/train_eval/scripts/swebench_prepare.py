"""Offline SWE-bench Verified evaluation through the shared native-tool collector."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
BENCHMARK = "swebench_verified"
CONFIG = HERE.parent / "config/swebench_verified.yaml"


def parser(env=None):
    env = os.environ if env is None else env
    defaults = yaml.safe_load(CONFIG.read_text())["defaults"]
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("algorithm", nargs="?", default="grpo_react",
                   choices=["baseline", "grpo_react", "gigpo", "dyad", "dyad-grpo", "dyad-gigpo"])
    p.add_argument("--model", default=env.get("MODEL_NAME", defaults["model"]))
    weights = p.add_mutually_exclusive_group()
    weights.add_argument("--model-path")
    weights.add_argument("--checkpoint")
    weights.add_argument("--projector-init")
    p.add_argument("--model-config")
    p.add_argument("--api-url", default=env.get("EVAL_API_URL"))
    p.add_argument("--served-model", default=env.get("EVAL_SERVED_MODEL", "evaluation-model"))
    p.add_argument("--assets", type=Path, default=Path(env.get("SWEBENCH_ASSETS", PROJECT / "data/swebench_verified/assets")))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--mini-swe-config", type=Path,
                   help="Enable the complete mini-swe-agent tool using a local nonsecret model/budget YAML")
    p.add_argument("--debug", action="store_true")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--task-ids", nargs="+")
    selection.add_argument("--limit", type=int)
    for name in ("seed", "num_trials", "max_concurrency", "context_length", "max_prompt_length", "max_tokens",
                 "max_steps", "history_length", "command_timeout", "grading_timeout", "episode_timeout", "pids_limit"):
        p.add_argument("--" + name.replace("_", "-"), type=int, default=defaults[name])
    for name in ("temperature", "environment_cpus"):
        p.add_argument("--" + name.replace("_", "-"), type=float, default=defaults[name])
    p.add_argument("--memory-limit", default=defaults["memory_limit"])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Validate local source, task selection and weights, without Ray")
    mode.add_argument("--preflight", action="store_true", help="Also inspect local Docker assets, without policy generation")
    mode.add_argument("--dry-run", action="store_true", help="Show final shared evaluation command without model loading")
    return p, defaults


def validate(args):
    for name in ("num_trials", "max_concurrency", "context_length", "max_prompt_length", "max_tokens", "max_steps",
                 "command_timeout", "grading_timeout", "episode_timeout", "pids_limit"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.history_length < 0 or not 0 <= args.seed < 2**31:
        raise ValueError("Invalid history length or seed")
    if args.max_prompt_length + args.max_tokens > args.context_length:
        raise ValueError("Prompt plus per-decision generation exceeds context length")
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
        raise ValueError("temperature must be finite and in [0, 2]")
    if not math.isfinite(args.environment_cpus) or args.environment_cpus <= 0:
        raise ValueError("environment_cpus must be finite and positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.task_ids and len(args.task_ids) != len(set(args.task_ids)):
        raise ValueError("Duplicate task IDs")
    if args.debug and (args.num_trials != 1 or (args.limit or 0) > 3 or len(args.task_ids or []) > 3):
        raise ValueError("Debug permits at most three tasks and one trial")
    if args.algorithm == "dyad" and not (args.checkpoint or args.projector_init):
        raise ValueError("Dyad requires an exact complete AgenticRL checkpoint or Alignment projector")
    if args.algorithm != "dyad" and args.projector_init:
        raise ValueError("Only Dyad accepts a projector")
    if args.projector_init and not args.model_config:
        raise ValueError("Alignment Dyad requires an explicit saved model configuration")
    if args.output_dir is not None and not args.output_dir.is_absolute():
        raise ValueError("output-dir must be absolute")


def local_weights(path):
    """Check every indexed shard locally, never resolve a model from the hub."""
    root = Path(path).expanduser().resolve()
    if not (root / "config.json").is_file():
        raise ValueError("Local policy model must contain config.json")
    indexes = list(root.glob("*.index.json"))
    weight_paths = []
    for index in indexes:
        value = json.loads(index.read_text())
        if "weight_map" not in value:
            continue
        for name in set(value["weight_map"].values()):
            if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Invalid model shard path")
            weight_paths.append(root / name)
    if not weight_paths:
        weight_paths = list(root.glob("*.safetensors")) + list(root.glob("pytorch_model*.bin"))
    if not weight_paths or any(not path.is_file() or path.stat().st_size == 0 for path in weight_paths):
        raise ValueError("Local model weights are missing or incomplete")
    if not any((root / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise ValueError("Local tokenizer files are missing")
    return str(root)


def build_command(env, config, defaults):
    from agent_system.evaluation.config import from_environment, command
    return command(from_environment(env, BENCHMARK, dataset=str(Path(env['RUN_DIR']) / 'public_tasks.json'),
                                    task_config=config))


def main(argv=None):
    p, defaults = parser()
    args = p.parse_args(argv)
    try:
        return prepare(args, defaults, dict(os.environ))
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[swebench-prepare] {exc}", file=sys.stderr)
        return 2


def prepare(args, defaults, env):
    from prepare import apply_algorithm, baseline_checkpoint_algorithm, effective_cpus, output_layout, resolve_model
    from tau_prepare import configure_weights
    if args.algorithm == "baseline":
        args.algorithm = baseline_checkpoint_algorithm(args.checkpoint, args.model_config)
    if args.algorithm != "dyad":
        args.algorithm = apply_algorithm(env, args.algorithm).algo
    validate(args)
    mini_config = None
    if args.mini_swe_config is not None:
        from agent_system.environments.env_package.swebench.mini_agent import load_config
        mini_config = load_config(args.mini_swe_config)
    env.update(PYTHON_BIN=sys.executable, RUN_IS_EVAL="1", RUN_ENV=BENCHMARK,
               RUN_IS_DEBUG="1" if args.debug else "0", EFFECTIVE_CPUS=str(effective_cpus()),
               PYTHONDONTWRITEBYTECODE="1", POST_ANALYSIS="0")
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT), str(HERE), env.get("PYTHONPATH", "")])
    for key in ("n_gpus_per_node", "rollout_tp_size", "gpu_mem_util"):
        env.setdefault(key.upper(), str(defaults[key]))
    source = configure_weights(args, env, defaults)
    if args.checkpoint:
        env['EVAL_CHECKPOINT'] = args.checkpoint
    if args.api_url:
        env['EVAL_API_URL'] = args.api_url
    env['EVAL_SERVED_MODEL'] = args.served_model
    if env.get("RESUME_MODE") == "resume_path":
        from agent_system.policies.dyad.inference.checkpoint import native_checkpoint_files
        ranks = len(native_checkpoint_files(env["RESUME_CKPT"]))
        if ranks != int(env["N_GPUS_PER_NODE"]):
            raise ValueError(f"Text checkpoint requires its original {ranks} policy ranks; set N_GPUS_PER_NODE={ranks}")
    selection = apply_algorithm(env, args.algorithm)
    env["RUN_ALGO"] = selection.public_alias
    resolve_model(env)
    local_weights(env["MODEL_PATH"])
    if args.output_dir:
        env["RUN_DIR"] = str(args.output_dir)
    output_layout(env, BENCHMARK, args.algorithm)
    env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled", DO_NOT_TRACK="1")
    # Resolve the tokenizer/model identity without loading weights or permitting hub fallback.
    from transformers import AutoConfig, AutoTokenizer
    from agent_system.parsers.native_tools import native_protocol
    model_config = AutoConfig.from_pretrained(env["MODEL_PATH"], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(env["MODEL_PATH"], local_files_only=True)
    native_protocol(tokenizer)
    maximum = getattr(model_config, "max_position_embeddings", None)
    if maximum is None and getattr(model_config, "text_config", None) is not None:
        maximum = getattr(model_config.text_config, "max_position_embeddings", None)
    if maximum is not None and args.context_length > maximum:
        raise ValueError("Requested context exceeds the saved model context capacity")
    config = {key: value for key, value in vars(args).items()
              if key not in {"model_path", "checkpoint", "projector_init", "model_config", "output_dir", "check", "preflight", "dry_run", "mini_swe_config"}}
    config.update(benchmark=BENCHMARK, assets=str(args.assets.expanduser().resolve()),
                  limit=args.limit if args.limit is not None else (3 if args.debug else -1),
                  output_dir=str(Path(env["RUN_DIR"]) / "episodes"))
    if mini_config is not None:
        config["mini_swe_agent"] = mini_config
    if args.algorithm == "dyad":
        env.update(DYAD_DYNAMIC_ACTIONS="1", DYAD_ACTION_CAPACITY=str(defaults["action_capacity"]),
                   DYAD_ACTION_CONTEXT_DIR=str(Path(env["RUN_DIR"]) / "encoder_contexts"))
        env.setdefault("DYAD_ENCODER_MAX_LENGTH", str(defaults["encoder_max_length"]))
        env.pop("DYAD_CODEGYM_ALL", None)
    from agent_system.environments.env_package.swebench.offline import OFFLINE_ENV, verify_network_isolation
    env.update(OFFLINE_ENV)
    from agent_system.environments.backends.swebench.worker import environment_interpreter
    executable = environment_interpreter(env)
    env.update(SWEBENCH_PYTHON_BIN=executable, SWE_PYTHON_BIN=executable)
    formal = not (args.check or args.dry_run or args.preflight)
    offline_evidence = verify_network_isolation() if formal else {"driver_network": "not_checked"}
    offline_evidence.update(required=formal, environment_python=executable, policy_python=sys.executable)
    config["require_network_isolation"] = formal
    from agent_system.environments.backends.swebench.worker import verify_ray_runtime
    config["expected_ray_runtime"] = verify_ray_runtime()
    # Only the isolated environment interpreter reads evaluator-only task files.
    preflight = environment_preflight(config, env, docker=args.preflight or not (args.check or args.dry_run))
    if preflight.get("evaluation_selection") is not None:
        config["evaluation_selection"] = preflight["evaluation_selection"]
        config["task_ids"] = preflight["evaluation_selection"]["selected_task_ids"]
    from agent_system.environments.backends.swebench.dataset import SwebenchEvaluationDataset
    dataset = SwebenchEvaluationDataset(preflight, config={"swebench": config})
    from agent_system.environments.env_package.swebench.tools import action_tools, build_initial_messages, scaffold_identity
    tools = action_tools(mini_config is not None)
    scaffold_version, schema_hash = scaffold_identity(mini_config is not None)
    if args.algorithm == "dyad":
        from agent_system.policies.dyad.actions.native_tools import compile_tools
        from agent_system.utils.hf_config import text_vocab_size
        compile_tools(tokenizer, text_vocab_size(model_config), tools, defaults["action_capacity"])
    from agent_system.rollout.prompt import strip_thinking_prefill
    from agent_system.environments.prompts.protocol import task_prompt_protocol
    lengths = [len(strip_thinking_prefill(tokenizer.apply_chat_template(
                    build_initial_messages(task, mini_config is not None), tools=tools, add_generation_prompt=True,
                    enable_thinking=False, tokenize=True, return_dict=False), tokenizer))
               for task in preflight["tasks"]]
    if max(lengths) > args.max_prompt_length:
        raise ValueError("A selected initial task prompt exceeds max-prompt-length; no task may be filtered")
    config.update(scaffold_version=scaffold_version, schema_hash=schema_hash,
                  task_prompt_protocol=task_prompt_protocol(BENCHMARK),
                  initial_prompt_max_tokens=max(lengths))
    command = build_command(env, config, defaults)
    from prepare import effective_cpus
    values = json.loads(command[-1])
    required = config["max_concurrency"] * config["environment_cpus"]
    resources = {"runtime": "inference_api", "ray_cpus": effective_cpus(),
                 "required_cpu_lower_bound": required, "reservations": {"SWE environment actors": required}}
    if resources["ray_cpus"] < required:
        raise ValueError(f"Evaluation requires at least {math.ceil(required)} environment CPUs")
    if args.check or args.preflight:
        print(json.dumps({"benchmark": BENCHMARK, "planned": len(dataset), "preflight": preflight,
                          "resources": resources, "offline_evidence": offline_evidence,
                          "source_identity": source["identity"] if source else None,
                          "task_ids": sorted({row["instance_id"] for row in dataset.planned_episodes})}, ensure_ascii=False))
        return 0
    if args.dry_run:
        print(shlex.join(command))
        return 0
    import torch
    required_gpus = 0 if args.api_url else int(env["ROLLOUT_TP_SIZE"])
    if torch.cuda.device_count() < required_gpus:
        raise ValueError(f"Inference requires {required_gpus} visible GPUs")
    directory = Path(env["RUN_DIR"])
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "public_tasks.json").write_text(json.dumps(preflight, ensure_ascii=False, allow_nan=False) + "\n")
    tool_config = {"tools": [{"class_name": "agent_system.environments.backends.swebench.tool.SwebenchLocalEnvTool",
                             "config": {"type": "native", "pool_size": config["max_concurrency"],
                                        "num_cpus_per_worker": config["environment_cpus"],
                                        "python_executable": executable,
                                        "session_config": config,
                                        # One native decision can contain multiple sequential calls.
                                        "step_timeout_s": config["episode_timeout"] + 60,
                                        "finalize_timeout_s": config["grading_timeout"] + 120}}]}
    (directory / "environment_tool.yaml").write_text(yaml.safe_dump(tool_config, sort_keys=False))
    configuration = {"benchmark": BENCHMARK, "algorithm": selection.public_alias,
                     "action_interface": selection.action_interface, "adv_estimator": selection.adv_estimator,
                     "model": args.model, "planned_episodes": len(dataset), "swebench": config,
                     "preflight": preflight, "resources": resources, "hydra": values,
                     "offline_evidence": offline_evidence,
                     "evaluation_checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
                     "source_benchmark": env.get("EVAL_SOURCE_BENCHMARK"), "target_benchmark": BENCHMARK}
    if source:
        configuration.update(native_source=source, source_benchmark=source["identity"]["training_benchmark"],
                             target_actions_rebuilt=True)
    plan = {"mode": "evaluation", "benchmark": BENCHMARK, "algo": args.algorithm,
            "env": env, "command": command, "configuration": configuration}
    with tempfile.TemporaryDirectory(prefix="swebench-evaluation-") as temporary:
        path = Path(temporary) / "plan.json"
        path.write_text(json.dumps(plan))
        from run import run_process
        return run_process([sys.executable, str(HERE / "run.py"), str(path)], env)


def environment_preflight(config, env, *, docker):
    from agent_system.environments.backends.swebench.worker import environment_interpreter
    executable = environment_interpreter(env)
    completed = subprocess.run([executable, "-m", "agent_system.environments.backends.swebench.session",
                                "--preflight" if docker else "--check"],
                               input=json.dumps(config), text=True, capture_output=True, env=env, timeout=300)
    if completed.returncode:
        raise ValueError(f"SWE-bench local preflight failed: {completed.stderr.strip()}")
    return json.loads(completed.stdout)
