#!/usr/bin/env python3
# Copyright 2025 ExpA_sys
# Licensed under the Apache License, Version 2.0 (the "License").
"""Agentic RL startup checks: projector compatibility, Ray resources and environment interaction.

The train/evaluation launchers invoke this file automatically. Heavy dependencies are imported
only by the selected check; the std baseline never imports dyad modules. Temporary Ray clusters
are owned by run.py, including failure and signal cleanup.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
import subprocess
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

SELF = Path(__file__).resolve()
PROJECT_DIR = SELF.parents[4]
PROJECT = PROJECT_DIR
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(SELF.parent))

from agent_system.utils.artifact_paths import artifact_root, run_site
from agent_system.utils.thinking import is_qwen35_model
from agent_system.environments.prompts.protocol import task_prompt_protocol
from verl_extensions.agent_steps.protocol import PARTITION_PROTOCOL

CONFIG_DIR = PROJECT_DIR / "agent_system/environments/configs"

# A small pool on purpose: what is under test is whether the chain works, not how much concurrency it
# sustains. Concurrency is visible in BaseEnvPool.start()'s materialize batches and in the per-step
# waited_ms during real training.
PROBE_POOL_SIZE = 4

LOG = "[check-env]"


@dataclass
class EnvCase:
    """One env's probe: how to build the tool, what task to create, which actions to send."""

    name: str
    tool_config: str
    tool_class: str
    tool_name: str
    # Returns (create_kwargs, actions). A callable because codegym has to scan the env dir first.
    # create_kwargs has the same shape as the dataset's tools_kwargs.<tool>.create_kwargs, so this
    # goes through the real `_build_reset_spec` path instead of around it.
    make_task: Callable[[], tuple[dict, list[str]]]
    # Trainers override pool size through this env var; reusing the same name avoids a second spelling.
    pool_size_env: Optional[str] = None
    #: --depth full only: the data this env cannot run without. Returns the paths it verified, so
    #: the log says what was actually looked at rather than just "OK".
    check_data: Optional[Callable[[], list[str]]] = None
    #: --depth full only: assert on the probe's own result. Carried over from the per-env scripts --
    #: calc's was the one with teeth, and the light probe recorded the reward without asserting it.
    assert_full: Optional[Callable[[dict], None]] = None


# --------------------------------------------------------------------------------------------
# --depth full: data layout
#
# These ran in the per-env shell scripts. They are here so that "which data does this env need"
# has one answer per env instead of one answer per script -- codegym's script had none at all,
# which is why a codegym job with a missing env directory failed inside the rollout.
# --------------------------------------------------------------------------------------------
def _require(paths: list[Path], *, kind: str) -> list[str]:
    missing = [str(p) for p in paths if not (p.is_dir() if kind == "dir" else p.is_file())]
    if missing:
        raise RuntimeError(f"missing {kind}(s): {missing}")
    return [str(p) for p in paths]


def _calc_data() -> list[str]:
    """GRPO, GiGPO and both Dyad losses consume the same algorithm-neutral parquet."""
    path = Path(os.environ.get("TRAIN_DATA") or PROJECT_DIR / "data/gsm8k/dataset/train.parquet")
    return _require([path], kind="file")


def _alfworld_data() -> list[str]:
    data_dir = Path(os.environ.get("ALFWORLD_DATA") or (Path.home() / ".cache/alfworld"))
    checked = _require([data_dir / "json_2.1.1/train", data_dir / "json_2.1.1/valid_seen"], kind="dir")
    checked += _require([data_dir / "logic/alfred.pddl", data_dir / "logic/alfred.twl2"], kind="file")

    # The game list is a separate failure from "the directory exists": the mapping files select
    # which games are used, and an empty or dangling selection fails inside AlfredTWEnv.
    from agent_system.environments.backends.alfworld.pool import load_alfworld_games

    games = load_alfworld_games(
        str(data_dir),
        str(CONFIG_DIR / "alfworld_mappings_train.json"),
        str(CONFIG_DIR / "alfworld_mappings_test.json"),
    )
    if not games or not os.path.exists(games[0]):
        raise RuntimeError(f"ALFWorld game list is unusable: count={len(games)}")
    checked.append(f"{len(games)} games, first={games[0]}")
    return checked


def _codegym_data() -> list[str]:
    from agent_system.environments.backends.codegym.pool import _default_envs_dir

    envs_dir = Path(os.environ.get("CODEGYM_ENVS_DIR") or _default_envs_dir()).expanduser().resolve()
    if not envs_dir.is_dir():
        raise RuntimeError(f"CodeGym env directory does not exist: {envs_dir}")
    files = sorted(p for p in envs_dir.iterdir() if p.name.endswith("Env.py"))
    if not files:
        raise RuntimeError(f"no *Env.py under {envs_dir}")
    from agent_system.environments.backends.codegym.worker import source_compatibility

    report = source_compatibility().validate_environment_sources(envs_dir)
    return [f"{report['checked']} *Env.py under {envs_dir}; source compilation passed",
            f"source compatibility repairs: {json.dumps(report['repairs'], sort_keys=True)}"]


def _webshop_task() -> tuple[dict, list[str]]:
    import pyarrow.parquet as pq

    evaluation = os.environ.get('RUN_IS_EVAL') == '1'
    selected = 'TEST_DATA' if evaluation else 'TRAIN_DATA'
    split = 'test' if evaluation else 'train'
    path = Path(os.environ.get(selected) or PROJECT_DIR / f'data/webshop/dataset/{split}.parquet')
    row = next(pq.ParquetFile(path).iter_batches(batch_size=1)).to_pylist()[0]
    task = row['extra_info']['tools_kwargs']['webshop_action']['create_kwargs']
    return task, ['search[shoes]']


async def _run_webshop_case(case, depth):
    """Probe the real shared-catalog lease path, not BaseEnvPool's one-actor-per-task API."""
    tool = _load_tool(case)
    task, actions = case.make_task()
    opened = []
    try:
        await tool.pool.start()
        slots = tool.pool.pool_size * tool.pool.config['sessions_per_backend']
        for session_id in ('webshop-probe-a', 'webshop-probe-b')[:min(2, slots)]:
            _, response = await tool.create(instance_id=session_id, create_kwargs=task)
            opened.append(session_id)
            if not response.text:
                raise RuntimeError('WebShop reset returned empty text')
        other_initial = tool._instance_dict[opened[1]]['observation'] if len(opened) == 2 else None
        steps = []
        initial = tool._instance_dict[opened[0]]['observation']
        for action in actions:
            response, reward, metrics = await tool.execute(opened[0], {'raw_action': action})
            if (not response.text or not 0 <= float(reward) <= 1 or
                    tool._instance_dict[opened[0]]['observation'] == initial):
                raise RuntimeError('WebShop real search returned invalid observation/reward')
            steps.append({'action': action, 'text_len': len(response.text), 'reward': reward,
                          'done': bool(metrics.get('done'))})
        if other_initial is not None:
            # The official invalid command is a no-op that returns the current page.
            # Query the backend, not only the tool's cached observation, for isolation.
            untouched = await tool.pool.step_session(opened[1], 'invalid')
            if untouched['observation'] != other_initial:
                raise RuntimeError('WebShop same-goal sessions contaminated one another')
        for session_id in list(opened):
            await tool.release(session_id)
            opened.remove(session_id)
        if tool.pool._sessions or tool.pool._free.qsize() != (
                tool.pool.pool_size * tool.pool.config['sessions_per_backend']):
            raise RuntimeError('WebShop probe leaked a session lease')
        return {'env': 'webshop', 'tool_name': case.tool_name, 'depth': depth, 'ok': True,
                'num_actors': tool.pool.pool_size, 'steps': steps, 'import_light': 'not_checked',
                'stable_worker_idx': 'shared-catalog leases', 'returned_to_free_queue': True,
                'same_goal_isolation': other_initial is not None}
    finally:
        try:
            for session_id in opened:
                await tool.release(session_id)
        finally:
            await tool.pool.shutdown()


def _calc_assert_full(out: dict) -> None:
    """calc has to actually score its own answer.

    The light probe records `reward` and asserts only that the text was non-empty, which passes
    even when the reward function is wired to a constant. `answer 72` against `ground_truth="72"`
    must be 1.0 and must end the episode -- that is the whole contract the trainer depends on.
    """
    steps = out.get("steps") or []
    calc_step = next((s for s in steps if s["action"].startswith("48/2=")), None)
    if calc_step is None or "24" not in str(calc_step.get("text", "")):
        raise RuntimeError(f"calc: `48/2=` did not return 24: {calc_step!r}")
    last = steps[-1]
    if float(last.get("reward") or 0.0) != 1.0 or not last.get("done"):
        raise RuntimeError(f"calc: `answer 72` scored {last.get('reward')!r} done={last.get('done')!r}, "
                           "expected reward=1.0 done=True")


def _codegym_task() -> tuple[dict, list[str]]:
    from agent_system.environments.backends.codegym.pool import _default_envs_dir

    envs_dir = Path(os.environ.get("CODEGYM_ENVS_DIR") or _default_envs_dir()).expanduser().resolve()
    files = sorted(p for p in envs_dir.iterdir() if p.name.endswith("Env.py"))
    if not files:
        raise RuntimeError(f"no *Env.py under the CodeGym env dir: {envs_dir}")
    # Pin this one: its constructor arguments are known. Taking files[0] would drift with the
    # directory's contents, so the check would start failing without any code change.
    preferred = next((p for p in files if "LongestSubstringEnv" in p.name), None)
    selected = preferred or files[0]
    payload = '{"s":"abcabcbb"}' if preferred else "{}"
    env_str = f"codegym_v1@{selected.stem}@{payload}"
    os.environ.setdefault("CODEGYM_ENVS_DIR", str(envs_dir))
    actions = [json.dumps({"name": "Observe", "parameters": {}})]
    return {"create_payload": {"env_str": env_str}}, actions


CASES: list[EnvCase] = [
    EnvCase(
        name="calc",
        tool_config=str(CONFIG_DIR / "calc_tool.yaml"),
        tool_class="CalcLocalEnvTool",
        tool_name="calculator",
        make_task=lambda: ({"ground_truth": "72"}, ["48/2=", "answer 72"]),
        pool_size_env="CALC_ENV_POOL_SIZE",
        check_data=_calc_data,
        assert_full=_calc_assert_full,
    ),
    EnvCase(
        name="alfworld",
        tool_config=str(CONFIG_DIR / "alfworld_tool.yaml"),
        tool_class="AlfworldLocalEnvTool",
        tool_name="alfworld_action",
        make_task=lambda: ({"reset_payload": {"game": 0, "world_type": "Text"}}, ["look", "inventory"]),
        pool_size_env="ALFWORLD_ENV_POOL_SIZE",
        check_data=_alfworld_data,
    ),
    EnvCase(
        name="codegym",
        tool_config=str(CONFIG_DIR / "codegym_tool.yaml"),
        tool_class="CodeGymLocalEnvTool",
        tool_name="codegym_call",
        make_task=_codegym_task,
        pool_size_env="CODEGYM_ENV_POOL_SIZE",
        check_data=_codegym_data,
    ),
    EnvCase(
        name='webshop',
        tool_config=str(CONFIG_DIR / 'webshop_tool.yaml'),
        tool_class='WebShopLocalEnvTool',
        tool_name='webshop_action',
        make_task=_webshop_task,
    ),
]


def _load_tool(case: EnvCase):
    from verl.tools.tool_registry import initialize_tools_from_config

    tools = initialize_tools_from_config(case.tool_config)
    classes = [type(t).__name__ for t in tools]
    tool = next((t for t in tools if type(t).__name__ == case.tool_class), None)
    if tool is None:
        raise RuntimeError(f"{case.name}: expected {case.tool_class}, loaded {classes}")
    got = getattr(tool, "name", None)
    if got != case.tool_name:
        # A name mismatch makes the agent loop's _call_tool look it up, miss, and skip *silently*
        # (port_delta section 10.5.1 item 1: that is how gsm8k's reward became a constant 0).
        raise RuntimeError(f"{case.name}: tool name {got!r} != configured {case.tool_name!r}")
    return tool


async def _check_import_light(pool) -> list[dict]:
    """Ask every worker whether torch / verl made it into its process."""
    states = await asyncio.gather(*(pool._call(w, "debug_import_state") for w in pool._workers))
    heavy = [s for s in states if s.get("torch_imported") or s.get("verl_imported")]
    if heavy:
        raise RuntimeError(f"worker is no longer import-light (this causes a worker storm): {heavy}")
    return list(states)


async def _run_case(case: EnvCase, pool_size: int, depth: str) -> dict:
    if case.name == 'webshop':
        return await _run_webshop_case(case, depth)
    if case.pool_size_env:
        os.environ[case.pool_size_env] = str(pool_size)
    out: dict[str, Any] = {"env": case.name, "tool_class": case.tool_class, "tool_name": case.tool_name,
                           "depth": depth}

    # Data first: it is the cheapest check and the one whose failure is least interesting to
    # diagnose from inside Ray. A missing parquet should not cost a pool startup.
    if depth == "full" and case.check_data is not None:
        out["data"] = case.check_data()
        print(f"  data OK: {out['data']}", flush=True)

    create_kwargs, actions = case.make_task()
    tool = _load_tool(case)
    pool = tool.pool
    instance_id = f"{case.name}-probe"
    opened = False
    try:
        await pool.start()
        out["pool_size"] = pool.pool_size
        out["num_actors"] = len(pool._workers)

        states = await _check_import_light(pool)
        out["import_light"] = True
        out["actor_modules"] = [s.get("num_modules") for s in states]

        # create/execute/release on the tool -- the exact path the agent loop takes, rather than
        # bypassing `_build_reset_spec` and feeding worker.reset's arguments directly.
        _, create_resp = await tool.create(instance_id=instance_id, create_kwargs=create_kwargs)
        opened = True
        out["create_text_len"] = len(str(create_resp.text or ""))
        bound = pool._session_worker.get(instance_id, -1)

        steps = []
        for action in actions:
            resp, reward, metrics = await tool.execute(instance_id, {"raw_action": action})
            text = str(resp.text or "")
            steps.append(
                {
                    "action": action[:60],
                    # The text itself, truncated -- `assert_full` needs to look at it (calc has to
                    # see 24), and a length alone cannot distinguish a right answer from a wrong one.
                    "text": text[:200],
                    "text_len": len(text),
                    "reward": reward,
                    "done": bool(metrics.get("done")),
                    "worker_idx": pool._session_worker.get(instance_id, -1),
                }
            )
            if metrics.get("done"):
                break
        out["steps"] = steps

        # One instance must stay bound to one policy LLM backbone for its whole life. That is the observable
        # evidence for "the env is stateful and each trajectory owns an policy LLM backbone" -- switching policy LMs
        # mid-trajectory means the state was lost.
        idxs = {bound} | {s["worker_idx"] for s in steps}
        if len(idxs) != 1 or -1 in idxs:
            raise RuntimeError(f"{case.name}: worker_idx is not stable for one instance: {idxs}")
        out["stable_worker_idx"] = idxs.pop()

        if not steps or all(s["text_len"] == 0 for s in steps):
            raise RuntimeError(f"{case.name}: every step returned empty tool text")

        await tool.release(instance_id)
        opened = False
        # After release the policy LLM backbone must be back in the free queue, or the pool drains after a handful
        # of trajectories.
        if pool._free.qsize() != pool.pool_size:
            raise RuntimeError(
                f"{case.name}: {pool._free.qsize()} free policy LMs after release, expected {pool.pool_size}"
            )
        out["returned_to_free_queue"] = True

        # Env-specific end-state, only at --depth full. Last, because it reads `out["steps"]`.
        if depth == "full" and case.assert_full is not None:
            case.assert_full(out)
            out["assert_full"] = True
        out["ok"] = True
    finally:
        if opened:
            await tool.release(instance_id)
        await pool.shutdown()
    return out


def check_environment(args) -> int:
    import ray

    pool_size = max(1, args.pool_size)
    cases = [c for c in CASES if args.env is None or c.name == args.env]
    address = os.environ.get("RAY_ADDRESS")
    if address:
        ray.init(address=address)
    else:
        # Standalone probes own a private Ray directory; never share another run's temporary state.
        ray.init(
            num_cpus=min(16, os.cpu_count() or 8),
            include_dashboard=False,
            # Leave room for Ray's session/socket suffix under Linux's 107-byte limit.
            _temp_dir=os.environ.get("ENV_CHECK_RAY_DIR", f"/tmp/dyad-env-{os.getpid()}"),
        )
    print(f"{LOG} depth={args.depth} pool_size={pool_size} Ray resources={ray.cluster_resources()}",
          flush=True)

    results, failed = [], []
    for case in cases:
        print(f"\n{LOG} ==== {case.name} ({args.depth}) ====", flush=True)
        try:
            res = asyncio.run(_run_case(case, pool_size, args.depth))
            results.append(res)
            print(f"  PASS {json.dumps(res, ensure_ascii=False)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append(case.name)
            print(f"  FAIL {case.name}: {exc!r}", flush=True)
            traceback.print_exc()
    ray.shutdown()

    print(f"\n{LOG} ==== summary (depth={args.depth}) ====")
    for res in results:
        print(
            f"  {res['env']:<9} policy LMs={res['num_actors']:<3} import-light={res['import_light']} "
            f"worker_idx={res['stable_worker_idx']} steps={len(res['steps'])} "
            f"tool={res['tool_name']} data={'yes' if res.get('data') else '-'} "
            f"end-state={'yes' if res.get('assert_full') else '-'}"
        )
    if failed:
        print(f"{LOG} FAIL: {', '.join(failed)}")
        return 1
    print(f"{LOG} PASS: {len(results)}/{len(cases)} envs at depth={args.depth}")
    return 0



GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"

# Required checkpoint metadata, shared with load_projector_init.
REQUIRED = ("state_dict", "projector", "scale", "policy_model", "encoder_model",
            "policy_hidden", "encoder_hidden")


def check_projector(args) -> int:
    import torch
    from agent_system.policies.dyad.models.action_head_factory import _model_identity, _resolve_projector_init

    problems: list[str] = []

    # Reject ambiguous checkpoint paths; equal run names can refer to different experiments.
    try:
        path = _resolve_projector_init(args.projector_init)
    except Exception as exc:                                     # noqa: BLE001
        print(f"[projector-init] {RED}FAIL{RESET} 路径解析不了")
        print(f"  {exc}")
        return 1
    print(f"[projector-init] 解析 -> {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)

    missing = [k for k in REQUIRED if k not in payload]
    if missing:
        problems.append(
            f"缺字段 {missing}。2026-08-31 之前写的 checkpoint 只带 "
            f"projector/encoder_hidden/policy_hidden，答不出它匹不匹配这次 run —— "
            f"「答不出」与「匹配」是两个不同的答案，只有后者是继续的理由。"
            "要用它得拿现在的 "
            "agent_system/policies/dyad/training/action_encoder_alignment/actenc_alignment_train.py 重训。")
        for line in problems:
            print(f"[projector-init] {RED}FAIL{RESET} {line}")
        return 1

    # Use the same field validators as load_projector_init.
    want_policy = _model_identity(args.model_name)
    got_policy = _model_identity(payload["policy_model"])
    # An empty encoder_model uses the policy model specification with independent weights.
    got_encoder = _model_identity(payload["encoder_model"] or payload["policy_model"])

    checks = [
        ("projector", args.projector, str(payload["projector"]),
         "mean 没有参数、attention 有四个"),
        ("scale", args.scale, str(payload["scale"]),
         "unit 与 uniform 只差一个常数因子，没有任何指标会显示这个不一致"),
        ("policy_model", want_policy, got_policy,
         "w_a 是对着那个模型的 x_t 拟合的，跨模型做内积没有意义"),
        ("encoder_model", want_policy, got_encoder,
         "动作行来自那个模型的 hidden state；两个 LM 永远是同一个模型（S1.6）"),
    ]
    for field, want, got, why in checks:
        if want != got:
            problems.append(f"{field}: ckpt 写的是 {got!r}，这次 run 是 {want!r} —— {why}")

    ph, eh = int(payload["policy_hidden"]), int(payload["encoder_hidden"])
    if ph != eh:
        problems.append(
            f"policy_hidden={ph} != encoder_hidden={eh}。两个 LM 是同一个模型时它们必须相等，"
            f"对不上说明是 bug 不是配置错。")

    for field, want, got, _ in checks:
        mark = GREEN + "ok" + RESET if want == got else RED + "!!" + RESET
        print(f"  {mark} {field:<14} ckpt={got!r} run={want!r}")
    print(f"  {GREEN}ok{RESET} hidden         policy={ph} encoder={eh}")

    # Report source training quality to distinguish weak initialization from an incompatible checkpoint.
    summary = path.parent / "summary.json"
    if summary.is_file():
        import json
        s = json.loads(summary.read_text(encoding="utf-8"))
        val = (s.get("val") or {}).get("top1_accuracy")
        unseen = (s.get("unseen") or {}).get("top1_accuracy")
        steps = s.get("selected_step")
        print(f"  -- 这份 ckpt: step={steps} val_top1={val} unseen_top1={unseen}")

    if problems:
        print(f"[projector-init] {RED}FAIL{RESET}")
        for line in problems:
            print(f"  {line}")
        return 1
    print(f"[projector-init] {GREEN}PASS{RESET} {path.parent.name} 与 {args.model_name} 匹配")
    return 0



def check_ray_resources() -> int:
    import os
    import socket
    import time

    import ray

    expected_cpus = int(os.environ["EXPECTED_CPUS"])
    expected_gpus = int(os.environ["EXPECTED_GPUS"])
    timeout = int(os.environ["CHECK_TIMEOUT_SECONDS"])

    ray.init(
        address=os.environ["RAY_ADDRESS"],
        runtime_env={"env_vars": {"DYNAMIC_DYAD_RAY_PREFLIGHT": "1"}},
    )
    resources = ray.cluster_resources()
    actual_cpus = int(resources.get("CPU", 0))
    actual_gpus = int(resources.get("GPU", 0))
    print(f"[check-ray] cluster_resources={resources}", flush=True)
    if actual_cpus != expected_cpus:
        raise RuntimeError(f"Ray CPU数不符：actual={actual_cpus}, expected={expected_cpus}")
    if actual_gpus != expected_gpus:
        raise RuntimeError(f"Ray GPU数不符：actual={actual_gpus}, expected={expected_gpus}")


    @ray.remote(num_cpus=1)
    class CpuProbe:
        def inspect(self):
            return {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "preflight_env": os.environ.get("DYNAMIC_DYAD_RAY_PREFLIGHT"),
            }


    @ray.remote(num_cpus=0, num_gpus=1)
    class GpuProbe:
        def inspect(self):
            import torch

            return {
                "pid": os.getpid(),
                "ray_gpu_ids": ray.get_gpu_ids(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_cuda_devices": torch.cuda.device_count(),
            }


    cpu_probes = []
    gpu_probes = []
    try:
        cpu_probes = [CpuProbe.remote() for _ in range(expected_cpus)]
        pending = [probe.inspect.remote() for probe in cpu_probes]
        cpu_results = []
        deadline = time.monotonic() + timeout
        # Report slow worker startup by elapsed time rather than logging every registration.
        progress_interval = float(os.environ.get("RAY_WORKER_PROGRESS_INTERVAL_SECONDS", "10"))
        started_at = time.monotonic()
        next_report = started_at + max(progress_interval, 15.0)
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"CPU worker注册超时：ready={len(cpu_results)}/{expected_cpus}")
            ready, pending = ray.wait(pending, num_returns=1, timeout=min(10.0, remaining))
            if ready:
                cpu_results.extend(ray.get(ready))
            now = time.monotonic()
            if now >= next_report:
                print(
                    f"[check-ray] CPU worker注册中: ready={len(cpu_results)}/{expected_cpus} "
                    f"elapsed={now - started_at:.0f}s",
                    flush=True,
                )
                next_report = now + progress_interval

        unique_pids = {item["pid"] for item in cpu_results}
        if len(unique_pids) != expected_cpus:
            raise RuntimeError(
                f"CPU worker未全部独立注册：unique_pids={len(unique_pids)}, expected={expected_cpus}"
            )
        if any(item["preflight_env"] != "1" for item in cpu_results):
            raise RuntimeError("runtime_env未正确传入CPU worker")
        print(f"[check-ray] CPU worker注册 OK: {len(unique_pids)}/{expected_cpus}", flush=True)

        gpu_probes = [GpuProbe.remote() for _ in range(expected_gpus)]
        gpu_results = ray.get([probe.inspect.remote() for probe in gpu_probes], timeout=timeout)
        visible_devices = {item["cuda_visible_devices"] for item in gpu_results}
        if len(visible_devices) != expected_gpus:
            raise RuntimeError(f"GPU actor分配不唯一：{gpu_results}")
        if any(item["torch_cuda_devices"] != 1 for item in gpu_results):
            raise RuntimeError(f"GPU actor可见设备数异常：{gpu_results}")
        print(f"[check-ray] GPU actor调度 OK: {gpu_results}", flush=True)
    finally:
        for probe in gpu_probes:
            ray.kill(probe)
        for probe in cpu_probes:
            ray.kill(probe)
        ray.shutdown()

    print("[check-ray] PASS", flush=True)
    return 0



def effective_cpus() -> int:
    from run import effective_cpus as detect
    return detect()


def positive_env(name: str, default: int) -> int:
    value = int(os.environ.get(name) or default)
    if value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def run_probe(command: str, cpus: int, extra: dict[str, str], args: list[str]) -> int:
    from run import RaySession, run_process
    env = {**os.environ, **extra, 'RAY_NUM_CPUS': str(cpus),
           'PYTHON_BIN': sys.executable,
           'PYTHONPATH': str(PROJECT_DIR) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    with RaySession(env) as address:
        env['RAY_ADDRESS'] = address
        return run_process([sys.executable, str(SELF), command, *args], env,
                           timeout=int(extra.get('CHECK_TIMEOUT_SECONDS', '600')))


def verify_wandb_login() -> None:
    if os.environ.get('WANDB_MODE', 'offline') != 'online':
        return
    import wandb
    key = os.environ.get('WANDB_API_KEY')
    if not key:
        raise ValueError('WANDB_MODE=online requires WANDB_API_KEY')
    try:
        ok = wandb.login(key=key, relogin=True, timeout=30)
    except TypeError:
        ok = wandb.login(key=key, relogin=True)
    if not ok:
        raise RuntimeError('WandB login failed')


def run_preflight(args) -> int:
    if args.profile == 'calc':
        os.environ['MATH_DATA_MODE'] = args.variant
    depth = args.depth
    if depth == 'auto':
        depth = 'light' if os.environ.get('SKIP_TRAINER_PREFLIGHT') == '1' else 'full'
    if depth == 'light' and (args.profile == 'std' or os.environ.get('SKIP_ENV_LIGHT_CHECK') == '1'):
        return 0
    if depth == 'full':
        cpus = positive_env('RAY_PREFLIGHT_CPUS', effective_cpus())
        gpu_value = os.environ.get('N_GPUS_PER_NODE') or os.environ.get('RAY_PREFLIGHT_GPUS')
        if gpu_value is None:
            import torch
            gpu_value = str(torch.cuda.device_count())
        gpus = int(gpu_value)
        if gpus < 1:
            raise ValueError('Ray resource checks require a positive GPU count')
        timeout = positive_env('RAY_PREFLIGHT_TIMEOUT_SECONDS', cpus * 10 + 120)
        rc = run_probe('ray-probe', cpus, {'EXPECTED_CPUS': str(cpus),
                       'EXPECTED_GPUS': str(gpus), 'CHECK_TIMEOUT_SECONDS': str(timeout)}, [])
        if rc:
            return rc
    if args.profile != 'std':
        cpus = (positive_env('RAY_PREFLIGHT_CPUS', effective_cpus()) if depth == 'full'
                else positive_env('ENV_CHECK_RAY_CPUS', 16))
        pool = positive_env('ENV_CHECK_POOL_SIZE', 8 if depth == 'full' else 4)
        rc = run_probe('env-probe', cpus, {},
                       ['--env', args.profile, '--depth', depth, '--pool-size', str(pool)])
        if rc:
            return rc
        if depth == 'full':
            verify_wandb_login()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='Check resources and the selected environment', allow_abbrev=False)
    run.add_argument('profile', choices=('std', 'calc', 'alfworld', 'codegym', 'webshop'))
    run.add_argument('variant', nargs='?', choices=('react', 'dyad'))
    run.add_argument('--depth', choices=('auto', 'light', 'full'), default='auto')
    projector = commands.add_parser('projector', help='Validate initialization weights before loading a model', allow_abbrev=False)
    projector.add_argument('--projector-init', required=True)
    projector.add_argument('--model-name', required=True)
    projector.add_argument('--projector', default='attention')
    projector.add_argument('--scale', default='unit')
    env = commands.add_parser('env-probe', help='Internal environment probe on a temporary Ray cluster', allow_abbrev=False)
    env.add_argument('--env', choices=tuple(c.name for c in CASES), required=True)
    env.add_argument('--depth', choices=('light', 'full'), required=True)
    env.add_argument('--pool-size', type=int, required=True)
    commands.add_parser('ray-probe', help='Internal resource probe on a temporary Ray cluster', allow_abbrev=False)
    return parser


def checks_main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == 'run' and args.profile == 'calc' and args.variant is None:
        parser.error('calc requires variant react or dyad')
    if args.command == 'run' and args.profile != 'calc' and args.variant is not None:
        parser.error('variant is only accepted for calc')
    if args.command == 'env-probe' and args.pool_size < 1:
        parser.error('--pool-size must be positive')
    try:
        # Existing probe diagnostics remain on stderr; stdout carries a concise result.
        with contextlib.redirect_stdout(sys.stderr):
            if args.command == 'run':
                rc = run_preflight(args)
            elif args.command == 'projector':
                rc = check_projector(args)
            elif args.command == 'env-probe':
                rc = check_environment(args)
            else:
                rc = check_ray_resources()
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print('error: ' + json.dumps(str(exc), ensure_ascii=False))
        return 1
    print('status: ' + ('passed' if rc == 0 else 'failed'))
    return rc



# Shared launch preparation. Imports above stay light for baseline launches and --check.
import shlex
import tempfile
from datetime import datetime

TRAIN_EVAL = SELF.parent.parent
MODEL_KEYS = frozenset({
    'MODEL_NAME', 'MODEL_PATH', 'DYAD_ACTION_YAML', 'ACTION_YAML', 'TEMPLATE_STYLE',
    'DYAD_SURFACE_FORM', 'DYAD_VALUES', 'ENV_NAME', 'CODE_ID',
    'DYAD_CODEGYM_ALL', 'DYAD_CODEGYM_ACTION_CAPACITY',
    'DYAD_DYNAMIC_ACTIONS', 'DYAD_ACTION_CAPACITY',
    'DYAD_ENCODER_ENABLED', 'DYAD_ENCODER_BACKBONE', 'DYAD_ENCODER_PROJECTOR',
    'DYAD_ENCODER_REPRESENTATION', 'DYAD_ENCODER_DESCRIPTION', 'DYAD_ENCODER_TRAINING',
    'DYAD_TRAINING_SCHEDULE', 'DYAD_ENCODER_MODEL_PATH', 'DYAD_ENCODER_MAX_LENGTH',
    'DYAD_ENCODER_SCALE', 'DYAD_ENCODER_GATE_INIT', 'DYAD_PROJECTOR_LR',
    'DYAD_ADV_ESTIMATOR',
})


@dataclass(frozen=True)
class AlgorithmSelection:
    """Independent runtime and estimator axes behind the public recipe names."""

    action_interface: str
    adv_estimator: str
    algo: str
    public_alias: str


ALGORITHM_SELECTIONS = {
    'grpo_react': AlgorithmSelection('react', 'grpo', 'grpo_react', 'grpo_react'),
    'gigpo': AlgorithmSelection('react', 'gigpo', 'gigpo', 'gigpo'),
    'dyad-grpo': AlgorithmSelection('dyad', 'grpo', 'dyad', 'dyad-grpo'),
    'dyad-gigpo': AlgorithmSelection('dyad', 'gigpo', 'dyad', 'dyad-gigpo'),
}


def resolve_algorithm(algo, env=None):
    """Resolve public aliases; only legacy ``dyad`` reads DYAD_ADV_ESTIMATOR."""
    if algo == 'dyad':
        estimator = (env or {}).get('DYAD_ADV_ESTIMATOR', 'grpo')
        if estimator not in {'grpo', 'gigpo'}:
            raise ValueError('Dyad advantage estimator must be grpo or gigpo')
        algo = 'dyad-' + estimator
    if algo not in ALGORITHM_SELECTIONS:
        raise ValueError('Unsupported algorithm: ' + str(algo))
    return ALGORITHM_SELECTIONS[algo]


def apply_algorithm(env, algo):
    selection = resolve_algorithm(algo, env)
    env['RUN_ACTION_INTERFACE'] = selection.action_interface
    env['RUN_ADV_ESTIMATOR'] = selection.adv_estimator
    if selection.action_interface == 'dyad':
        env['DYAD_ADV_ESTIMATOR'] = selection.adv_estimator
    return selection


def checkpoint_algorithm(saved):
    """Read legacy metadata, checking optional explicit axes against its identity."""
    from agent_system.policies.dyad.checkpoint_compat import normalize_saved_model_config
    saved = normalize_saved_model_config(saved)
    selection = resolve_algorithm(saved.get('algo'), saved['model'])
    axes = {'action_interface', 'adv_estimator'}
    present = axes.intersection(saved)
    if present and present != axes:
        raise ValueError('Checkpoint must save action_interface and adv_estimator together')
    if present and (saved['action_interface'], saved['adv_estimator']) != (
            selection.action_interface, selection.adv_estimator):
        raise ValueError('Checkpoint action_interface/adv_estimator conflict with legacy algorithm identity')
    return selection


def default(env, key, value):
    if not env.get(key):
        env[key] = str(value)


TARGET_ACTION_KEYS = frozenset({'DYAD_ACTION_YAML', 'ACTION_YAML', 'TEMPLATE_STYLE',
                                'DYAD_SURFACE_FORM', 'DYAD_VALUES',
                                'DYAD_CODEGYM_ALL', 'DYAD_CODEGYM_ACTION_CAPACITY',
                                'DYAD_DYNAMIC_ACTIONS', 'DYAD_ACTION_CAPACITY', 'ENV_NAME', 'CODE_ID'})


def configure_cross_environment_restore(env, checkpoint, config, saved):
    """Keep source identity intact; only derived target actions may be rebuilt."""
    model = saved['model']
    if (model.get('DYAD_ENCODER_ENABLED') != '1' or
            model.get('DYAD_ENCODER_BACKBONE') != 'encoder_lm' or
            model.get('DYAD_ENCODER_TRAINING') not in {'projector_only', 'projector_and_encoder_lm'}):
        raise ValueError('Cross-environment restoration requires the saved independent encoder')
    base = Path(model['MODEL_PATH']).expanduser().resolve()
    encoder = Path(model.get('DYAD_ENCODER_MODEL_PATH') or base).expanduser().resolve()
    if encoder != base:
        raise ValueError('Cross-environment restoration requires the original policy base as frozen encoder')
    for key in ('MODEL_PATH', 'DYAD_ENCODER_MODEL_PATH'):
        if env.get(key) and Path(env[key]).expanduser().resolve() != base:
            raise ValueError(f'{key} conflicts with native source model configuration')
    if env.get('MODEL_NAME') and canonical_model(env['MODEL_NAME']) != canonical_model(model['MODEL_NAME']):
        raise ValueError('MODEL_NAME conflicts with native source model configuration')
    if any(env.get(key) for key in TARGET_ACTION_KEYS | {'ACTION_YAML', 'DYAD_DYNAMIC_ACTIONS'}):
        raise ValueError('Cross-environment evaluation owns target actions; remove action schema/capacity overrides')
    from agent_system.policies.dyad.inference.checkpoint import native_checkpoint_files
    native_checkpoint_files(checkpoint)
    from agent_system.policies.dyad.inference.source import source_metadata
    source = source_metadata(argparse.Namespace(checkpoint=str(checkpoint), projector_init=None,
                                               model_config=str(config)))
    env.update(DYAD_CROSS_ENV_SOURCE=json.dumps(source), RESUME_MODE='disable',
               DYAD_ENCODER_MODEL_PATH=str(base), DYAD_ENCODER_REMOTE='1')
    env.pop('RESUME_CKPT', None)
    default(env, 'DYAD_ENCODER_NUM_GPUS', '1')


def checkpoint_model_config(checkpoint, explicit=None):
    checkpoint = Path(checkpoint).expanduser()
    candidates = [Path(explicit).expanduser()] if explicit else [
        checkpoint / 'model_config.json', checkpoint.parent / 'model_config.json']
    config = next((path for path in candidates if path.is_file()), None)
    if config is None:
        return None, None
    from agent_system.policies.dyad.checkpoint_compat import read_saved_model_config
    saved = read_saved_model_config(config)
    if not isinstance(saved, dict) or saved.get('version') != 1 or not isinstance(saved.get('model'), dict):
        raise ValueError('model_config.json requires version=1 and a model object')
    return config, saved


def baseline_checkpoint_algorithm(checkpoint, explicit=None):
    if not checkpoint:
        raise ValueError('baseline evaluation requires an explicit --checkpoint')
    import re
    source = Path(checkpoint).expanduser()
    if not re.fullmatch(r'global_step_\d+', source.name) or not (source / 'actor').is_dir():
        raise ValueError('checkpoint must be an existing global_step_N directory containing actor/')
    _, saved = checkpoint_model_config(source, explicit)
    if saved is None:
        raise ValueError('baseline checkpoint requires its saved model_config.json')
    selection = checkpoint_algorithm(saved)
    if selection.action_interface != 'react':
        raise ValueError('baseline evaluation requires a GRPO or GiGPO text-policy checkpoint, not Dyad')
    return selection.algo


def load_model_config(env, benchmark, algo):
    ckpt = Path(env.get('EVAL_CHECKPOINT', '')).expanduser()
    import re
    if not re.fullmatch(r'global_step_\d+', ckpt.name) or not (ckpt / 'actor').is_dir():
        raise ValueError('checkpoint must be an existing global_step_N directory containing actor/')
    env['RESUME_CKPT'] = str(ckpt.resolve())
    env['RESUME_MODE'] = 'resume_path'
    explicit = env.get('EVAL_MODEL_CONFIG')
    if algo == 'baseline':
        algo = baseline_checkpoint_algorithm(ckpt, explicit)
    config, data = checkpoint_model_config(ckpt, explicit)
    if config:
        saved_selection = checkpoint_algorithm(data)
        cross_environment = data.get('benchmark') != benchmark
        if (algo != 'auto' and data.get('algo') != algo) or (cross_environment and
                (data.get('benchmark') not in {'alfworld', 'codegym', 'dive', 'webshop'} or
                 benchmark not in {'alfworld', 'codegym', 'dive', 'webshop'})):
            raise ValueError('checkpoint benchmark/algorithm does not match evaluation selection')
        algo = data['algo']
        if algo == 'dyad':
            saved_estimator = saved_selection.adv_estimator
            if env.get('DYAD_ADV_ESTIMATOR', saved_estimator) != saved_estimator:
                raise ValueError('Selected Dyad loss does not match checkpoint model configuration')
            env['DYAD_ADV_ESTIMATOR'] = saved_estimator
        env['EVAL_SOURCE_BENCHMARK'] = str(data['benchmark'])
        env['EVAL_SOURCE_MODEL_NAME'] = data['model'].get('MODEL_NAME', '')
        env['EVAL_SOURCE_MODEL_PATH'] = data['model'].get('MODEL_PATH', '')
        if cross_environment and saved_selection.action_interface == 'dyad':
            configure_cross_environment_restore(env, ckpt, config, data)
        for key, value in data['model'].items():
            if key not in MODEL_KEYS or not isinstance(value, str):
                raise ValueError(f'Unsupported model configuration key/value: {key}')
            if cross_environment and key in TARGET_ACTION_KEYS:
                continue
            if key in {'DYAD_ACTION_YAML', 'ACTION_YAML'}:
                from agent_system.compat import resolve_resource_path

                value = str(resolve_resource_path(value))
                if env.get(key):
                    env[key] = str(resolve_resource_path(env[key]))
            # Model paths may differ by deployment; architecture must match the saved model.
            if key in {'MODEL_PATH', 'MODEL_NAME', 'DYAD_ENCODER_MODEL_PATH'}:
                default(env, key, value)
            else:
                if env.get(key) and env[key] != value:
                    raise ValueError(f'{key} conflicts with checkpoint model configuration')
                env[key] = value
        model_overrides = data.get('model_overrides', [])
        if not isinstance(model_overrides, list) or any(not isinstance(v, str) or not v.lstrip('+').startswith('actor_rollout_ref.model.override_config.') or '=' not in v for v in model_overrides):
            raise ValueError('Invalid saved model overrides')
        env['EVAL_MODEL_OVERRIDES'] = json.dumps(model_overrides)
        env['EVAL_MODEL_CONFIG'] = str(config.resolve())
    elif explicit or algo in {'dyad', 'auto', 'gigpo'}:
        raise ValueError('Checkpoint has no model_config.json; supply --model-config PATH for a legacy checkpoint')
    return apply_algorithm(env, algo).algo


def resolve_alignment_projector(raw, environ=None):
    """Resolve an exact run, never a model prefix, glob, or latest checkpoint."""
    text = str(raw or '').strip()
    if not text or any(char in text for char in '*?[]'):
        raise ValueError('Specify an exact Alignment run directory (including timestamp) or projector.pt; wildcards are not supported')
    path = Path(text).expanduser()
    if path.is_absolute():
        candidates = [path]
    elif len(path.parts) == 1 and path.suffix != '.pt':
        # Bare timestamped run names use the cluster checkpoint namespace.
        candidates = [artifact_root(PROJECT_DIR, environ) / 'ckpt/lucia' / phase / path
                      for phase in ('alignment', 'stage1')]
    elif path.parts[0] in {'local', 'lucia'}:
        candidates = [artifact_root(PROJECT_DIR, environ) / 'ckpt' / path.parts[0] / phase / Path(*path.parts[1:])
                      for phase in ('alignment', 'stage1')]
    else:
        candidates = [PROJECT_DIR / path, Path.cwd() / path]
    hits = set()
    for candidate in candidates:
        if candidate.is_dir():
            candidate /= 'projector.pt'
        if candidate.is_file():
            hits.add(candidate.resolve())
    if len(hits) != 1:
        raise ValueError(f'Alignment projector path must identify exactly one existing file: {text!r}; '
                         'bare run names resolve under ckpt/lucia/alignment; no timestamp fallback is performed')
    return str(hits.pop())


def select_evaluation(env, benchmark, algo):
    mode = env.get('EVAL_MODE', 'base')
    if algo == 'baseline' and (mode != 'checkpoint' or not env.get('EVAL_CHECKPOINT')):
        raise ValueError('baseline evaluation requires an explicit --checkpoint')
    env.pop('DYAD_CROSS_ENV_SOURCE', None)
    env.pop('EVAL_SOURCE_BENCHMARK', None)
    for key in ('EVAL_SOURCE_MODEL_NAME', 'EVAL_SOURCE_MODEL_PATH'):
        env.pop(key, None)
    if mode == 'checkpoint':
        if env.get('EVAL_PROJECTOR_INIT') or env.get('DYAD_ENCODER_PROJECTOR_INIT'):
            raise ValueError('checkpoint evaluation cannot also initialize a standalone projector')
        algo = load_model_config(env, benchmark, algo)
    else:
        if env.get('EVAL_CHECKPOINT') or env.get('EVAL_MODEL_CONFIG'):
            raise ValueError('base/projector evaluation does not accept checkpoint configuration')
        env['RESUME_MODE'] = 'disable'
        env.pop('RESUME_CKPT', None)
        if mode == 'base':
            if algo == 'dyad' or env.get('EVAL_PROJECTOR_INIT') or env.get('DYAD_ENCODER_PROJECTOR_INIT'):
                raise ValueError('Dyad evaluation requires projector weights or a full checkpoint')
        elif mode == 'projector':
            if algo != 'dyad':
                raise ValueError('Projector evaluation requires algo=dyad')
            env['DYAD_ENCODER_PROJECTOR_INIT'] = resolve_alignment_projector(
                env.get('EVAL_PROJECTOR_INIT') or env.get('DYAD_ENCODER_PROJECTOR_INIT'), env)
            # Inference architecture defaults are data, not an executable training recipe.
            for key, value in experiment_recipes()['train']['defaults'].items():
                default(env, key, value)
        else:
            raise ValueError('Unknown evaluation weight mode: ' + mode)
    default(env, 'MODEL_NAME', 'Qwen3.5-2B')
    env.update(VAL_ONLY='True', VAL_BEFORE_TRAIN='True', SAVE_FREQ='-1', RUN_IS_EVAL='1', RUN_EVAL_TAG='bench')
    if algo == 'dyad':
        default(env, 'DYAD_ENCODER_REMOTE', '0')
    return algo


def action_config(env, benchmark):
    surface, values = env.get('DYAD_SURFACE_FORM', 'base'), env.get('DYAD_VALUES', 'open')
    if benchmark not in {'gsm8k', 'alfworld', 'codegym', 'webshop', 'dive'} or surface != 'base' or values not in (
            {'open', 'closed'} if benchmark == 'alfworld' else {'open'}):
        raise ValueError(f'Unsupported action space: {benchmark}/{surface}/{values}')
    env['DYAD_SURFACE_FORM'], env['DYAD_VALUES'] = surface, values
    if benchmark in {'codegym', 'dive'}:
        default(env, 'TEMPLATE_STYLE', 'base')
        return  # Schemas and data are selected per task.
    default(env, 'DYAD_ACTION_YAML', f"{benchmark}/base{'_closed' if values == 'closed' else ''}.yaml")


def shared_step_action_schema(benchmark, values):
    if benchmark not in {'alfworld', 'webshop'}:
        return None
    suffix = '_closed' if values == 'closed' else ''
    return f'{benchmark}/shared_step_v2{suffix}.yaml'


def marker_only_schema_change(source, target):
    """Known shared schemas retain action row order and weights, only markers change."""
    import yaml
    pairs = [(f'{benchmark}/base{suffix}.yaml', f'{benchmark}/shared_step_v2{suffix}.yaml')
             for benchmark, suffix in [('alfworld', ''), ('alfworld', '_closed'), ('webshop', '')]]
    if not any({source, target} == set(pair) for pair in pairs):
        return False
    schemas = []
    for name in (source, target):
        raw = yaml.safe_load((PROJECT_DIR / 'agent_system/policies/dyad/actions/schemas' / name).read_text())
        for key in ('enter', 'exit', 'preserve_exit_prefix', 'preserve_partial_exit_prefix'):
            raw['markers'].pop(key, None)
        schemas.append(raw)
    return schemas[0] == schemas[1] and list(schemas[0]['actions']) == list(schemas[1]['actions'])


def configure_step_action_schema(env, benchmark, algo):
    if algo != 'dyad' or env.get('STEP_ROLLOUT_ENABLED') != 'True':
        return
    target = shared_step_action_schema(benchmark, env.get('DYAD_VALUES', 'open'))
    if not target:
        return
    source = env.get('DYAD_ACTION_YAML')
    if source == target:
        return
    if (env.get('RUN_IS_EVAL') == '1' and env.get('EVAL_MODE') == 'checkpoint'
            and not env.get('DYAD_CROSS_ENV_SOURCE')):
        # Saved evaluation schema is immutable, including its marker protocol.
        # Cross-environment targets are derived, so they follow the shared step prompt.
        return
    if not source or not marker_only_schema_change(source, target):
        raise ConfigError('Shared step Dyad training requires its compatible shared_step_v2 action schema')
    env['DYAD_ACTION_YAML'] = target


SCALE_KEYS = frozenset({
    # batch
    "PPO_MINI_BATCH_SIZE",
    "PPO_EPOCHS", "LR", "KL_LOSS_COEF", "KL_LOSS_TYPE", "ROLLOUT_TP_SIZE", "LR_WARMUP_STEPS", "ROLLOUT_TEMPERATURE", "VAL_BATCH_SIZE", "VAL_MAX_SAMPLES", "TRAIN_MAX_SAMPLES", "TRAIN_BATCH_SIZE", "MICRO_BATCH_SIZE", "ROLLOUT_N",
    # GPUs
    "N_GPUS_PER_NODE", "CUDA_VISIBLE_DEVICES", "GPU_MEM_UTIL",
    # Dedicated encoder GPUs contribute to the total alongside N_GPUS_PER_NODE.
    "DYAD_ENCODER_NUM_GPUS", "DYAD_EXACT_BATCH_PADDING",
    # lengths and turn budgets
    "MAX_PROMPT_LENGTH", "MAX_RESPONSE_LENGTH", "MAX_TOOL_RESPONSE_LENGTH",
    "MAX_ASSISTANT_TURNS", "MAX_TOOL_TURNS",
    "MAX_NUM_SEQS", "MAX_NUM_BATCHED_TOKENS",
    # concurrency and memory strategy
    "AGENT_NUM_WORKERS", "PARAM_OFFLOAD", "OPTIMIZER_OFFLOAD",
    "USE_REMOVE_PADDING", "ATTN_IMPL",
    # env pool. gsm8k / alfworld and codegym's grpo_react all derive it as batch x n inside the
    # engine; only codegym/dyad.sh has no such derivation, so its debug profile states the value
    # here (S4 asserts it really equals batch x n, so it cannot drift into a batch-independent
    # constant).
    "CODEGYM_ENV_POOL_SIZE", "WEBSHOP_BACKEND_REPLICAS", "WEBSHOP_SESSIONS_PER_BACKEND", "WEBSHOP_MAX_STEPS",
    # how long to run, how often to save
    "TOTAL_EPOCHS", "TOTAL_TRAINING_STEPS", "SAVE_FREQ", "TEST_FREQ",
    "GIGPO_GAMMA", "GIGPO_MODE", "GIGPO_STEP_ADVANTAGE_W",
    "ROLLOUT_TOP_P", "VAL_TEMPERATURE", "VAL_TOP_P", "ENTROPY_COEFF", "CLIP_RATIO", "LOSS_AGG_MODE",
    "STEP_ROLLOUT_ENABLED", "STEP_PROFILE", "STEP_HISTORY_LENGTH", "STEP_INVALID_ACTION_PENALTY",
})

# GiGPO-only defaults; shared baseline resources still come from each benchmark.
GIGPO_DEFAULTS = {'GIGPO_GAMMA': '0.95', 'GIGPO_MODE': 'mean_std_norm',
                  'GIGPO_STEP_ADVANTAGE_W': '1.0'}

STEP_PROFILE_KEYS = frozenset({
    'STEP_ROLLOUT_ENABLED', 'STEP_PROFILE', 'STEP_HISTORY_LENGTH', 'STEP_INVALID_ACTION_PENALTY',
    'MAX_PROMPT_LENGTH', 'MAX_RESPONSE_LENGTH', 'MAX_ASSISTANT_TURNS', 'MAX_TOOL_TURNS',
    'TRAIN_BATCH_SIZE', 'PPO_MINI_BATCH_SIZE', 'ROLLOUT_N', 'PPO_EPOCHS',
    'ROLLOUT_TEMPERATURE', 'ROLLOUT_TOP_P', 'VAL_TEMPERATURE', 'VAL_TOP_P',
    'LR', 'KL_LOSS_COEF', 'ENTROPY_COEFF', 'CLIP_RATIO', 'LOSS_AGG_MODE',
    'GIGPO_GAMMA', 'GIGPO_MODE', 'GIGPO_STEP_ADVANTAGE_W',
})


# Persist only protocol/budget fields, never arbitrary inherited environment values.
STEP_PARAMETER_FIELDS = {
    'GIGPO_GAMMA': 'algorithm.gamma', 'GIGPO_MODE': 'algorithm.gigpo.mode',
    'GIGPO_STEP_ADVANTAGE_W': 'algorithm.gigpo.step_advantage_w',
    'MAX_PROMPT_LENGTH': 'data.max_prompt_length', 'MAX_RESPONSE_LENGTH': 'data.max_response_length',
    'MAX_ASSISTANT_TURNS': 'actor_rollout_ref.rollout.multi_turn.max_assistant_turns',
    'PPO_MINI_BATCH_SIZE': 'actor_rollout_ref.actor.ppo_mini_batch_size',
    'ROLLOUT_TEMPERATURE': 'actor_rollout_ref.rollout.temperature',
    'ROLLOUT_TOP_P': 'actor_rollout_ref.rollout.top_p',
    'VAL_TEMPERATURE': 'actor_rollout_ref.rollout.val_kwargs.temperature',
    'VAL_TOP_P': 'actor_rollout_ref.rollout.val_kwargs.top_p',
    'LR': 'actor_rollout_ref.actor.optim.lr', 'KL_LOSS_COEF': 'actor_rollout_ref.actor.kl_loss_coef',
    'ENTROPY_COEFF': 'actor_rollout_ref.actor.entropy_coeff',
    'CLIP_RATIO': 'actor_rollout_ref.actor.clip_ratio',
    'LOSS_AGG_MODE': 'actor_rollout_ref.actor.loss_agg_mode',
}


THINKING_FIELD = 'data.apply_chat_template_kwargs.enable_thinking'
QWEN35_THINKING = {'enable_thinking': False, 'policy': 'qwen35_no_thinking_v1'}


def qwen35_run(env):
    """Use model identity and local config, including relocated snapshot paths."""
    return is_qwen35_model(*(env.get(key) for key in (
        'MODEL_NAME', 'MODEL_PATH', 'DYAD_MODEL_SOURCE_PATH', 'EVAL_SOURCE_MODEL_NAME', 'EVAL_SOURCE_MODEL_PATH')))


def thinking_metadata(env):
    if not qwen35_run(env):
        return None
    return dict(QWEN35_THINKING)


def require_no_thinking(env, overrides):
    """Reject CLI bypasses before --check returns or any resources are started."""
    model_paths = []
    for arg in overrides:
        if arg.lstrip('+').startswith('actor_rollout_ref.model.path='):
            # Tau emits JSON-quoted strings while shell builders emit raw paths.
            # Compare Hydra's actual scalar, not its serialized spelling; never
            # strip quotes blindly or resolve user-controlled interpolations.
            from hydra.core.override_parser.overrides_parser import OverridesParser
            value = OverridesParser.create().parse_override(arg).value()
            if not isinstance(value, str):
                raise ConfigError('actor_rollout_ref.model.path must be a string path')
            model_paths.append(value)
    if not qwen35_run(env) and not is_qwen35_model(*model_paths):
        return
    for path in model_paths:
        if path != env.get('MODEL_PATH'):
            raise ConfigError('Use MODEL_PATH to select the Qwen3.5 model so thinking metadata remains consistent')
    for arg in overrides:
        stripped = arg.lstrip('+~')
        key, _, value = stripped.partition('=')
        prefix = arg[:len(arg) - len(stripped)]
        if env.get('RUN_IS_EVAL') != '1':
            resume_fields = {'trainer.resume_mode': env.get('RESUME_MODE', 'disable'),
                             'trainer.resume_from_path': env.get('RESUME_CKPT', '')}
            if key in resume_fields and (prefix not in ('', '+', '++') or value != resume_fields[key]):
                raise ConfigError('Qwen3.5 resume uses RESUME_MODE/RESUME_CKPT to validate the saved thinking profile')
        if (THINKING_FIELD.startswith(key + '.') or
                (key == THINKING_FIELD and (prefix not in ('', '+', '++') or value.lower() != 'false'))):
            raise ConfigError('Qwen3.5 requires data.apply_chat_template_kwargs.enable_thinking=False; '
                              'remove override ' + arg)


def configure_weight_import(env, benchmark, algo, overrides):
    """Only explicit model-only loading may cross training protocol versions."""
    env.pop('STEP_WEIGHT_IMPORT', None)
    if env.get('RUN_IS_EVAL') == '1':
        return
    field = 'actor_rollout_ref.actor.checkpoint.load_contents'
    selected = [arg for arg in overrides if arg.lstrip('+~').partition('=')[0] == field]
    if not selected:
        return
    from hydra.core.override_parser.overrides_parser import OverridesParser
    parsed = [OverridesParser.create().parse_override(arg) for arg in selected]
    if any(item.is_delete() or item.is_sweep_override() for item in parsed):
        raise ConfigError('Checkpoint load_contents must be one explicit list')
    values = [item.value() for item in parsed]
    if any(value != values[0] for value in values):
        raise ConfigError('Conflicting checkpoint load_contents overrides')
    if values[-1] != ['model']:
        return
    if env.get('DYAD_ENCODER_PROJECTOR_INIT'):
        raise ConfigError('Weight-only actor import cannot also initialize a standalone projector')
    if env.get('RESUME_MODE') != 'resume_path' or not env.get('RESUME_CKPT'):
        raise ConfigError('Weight-only import requires explicit RESUME_MODE=resume_path and RESUME_CKPT')
    source = Path(env['RESUME_CKPT']).expanduser().resolve()
    for key in ('RUN_DIR', 'CKPT_DIR'):
        if env.get(key):
            target = Path(env[key]).expanduser().resolve()
            if target == source or target in source.parents or source in target.parents or (target / 'model_config.json').exists():
                raise ConfigError('Weight-only import must start a new run with separate output/checkpoint directories')
    import re
    if not re.fullmatch(r'global_step_\d+', source.name) or not (source / 'actor').is_dir():
        raise ConfigError('Weight-only import requires an existing global_step_N checkpoint containing actor/')
    config_path, saved = checkpoint_model_config(source)
    if not saved:
        raise ConfigError('Weight-only import requires saved model_config.json for actor/action schema validation')
    selection, original = apply_algorithm(env, algo), checkpoint_algorithm(saved)
    if selection.action_interface != original.action_interface or saved.get('benchmark') != benchmark:
        raise ConfigError('Weight-only import requires matching action interface and environment schema')
    model = saved['model']
    if not model.get('MODEL_NAME') or canonical_model(env['MODEL_NAME']) != canonical_model(model['MODEL_NAME']):
        raise ConfigError('Weight-only import requires the same model architecture identity')
    if selection.action_interface == 'dyad':
        required = {'DYAD_ACTION_YAML', 'DYAD_ENCODER_ENABLED', 'DYAD_ENCODER_BACKBONE',
                    'DYAD_ENCODER_PROJECTOR', 'DYAD_ENCODER_REPRESENTATION', 'DYAD_ENCODER_DESCRIPTION',
                    'DYAD_SURFACE_FORM', 'DYAD_VALUES'}
        if required - model.keys():
            raise ConfigError('Weight-only Dyad import requires complete saved actor/action schema')
        for key in ('DYAD_DYNAMIC_ACTIONS', 'DYAD_ACTION_CAPACITY', 'DYAD_CODEGYM_ACTION_CAPACITY',
                    'DYAD_CODEGYM_ALL'):
            if env.get(key) and key not in model:
                raise ConfigError('Weight-only actor/action schema missing source field: ' + key)
    excluded = {'MODEL_NAME', 'MODEL_PATH', 'DYAD_ADV_ESTIMATOR', 'DYAD_PROJECTOR_LR',
                'DYAD_ENCODER_TRAINING', 'DYAD_TRAINING_SCHEDULE'}
    for key, value in model.items():
        if key not in MODEL_KEYS or not isinstance(value, str):
            raise ConfigError('Invalid saved model schema field: ' + key)
        if key in excluded:
            continue
        if env.get(key) and env[key] != value:
            if key == 'DYAD_ACTION_YAML' and marker_only_schema_change(value, env[key]):
                continue
            raise ConfigError('Weight-only actor/action schema mismatch: ' + key)
        env[key] = value
    saved_overrides = saved.get('model_overrides', [])
    if not isinstance(saved_overrides, list) or any(not isinstance(arg, str) or not arg.lstrip('+').startswith(
            'actor_rollout_ref.model.override_config.') for arg in saved_overrides):
        raise ConfigError('Invalid saved model architecture overrides')
    from hashlib import sha256
    env['STEP_WEIGHT_IMPORT'] = json.dumps({
        'kind': 'actor_weights_only', 'checkpoint': str(source),
        'source_model_config': str(config_path.resolve()),
        'source_model_config_sha256': sha256(config_path.read_bytes()).hexdigest(),
        'source_benchmark': saved['benchmark'], 'source_action_interface': original.action_interface,
        'source_estimator': original.adv_estimator, 'source_model': model,
        'source_training_protocol': saved.get('training_protocol'),
        'source_rollout_profile': saved.get('rollout_profile'), 'source_thinking': saved.get('thinking'),
        'source_model_overrides': saved_overrides,
        'restored_optimizer': False, 'restored_global_step': False, 'restored_dataloader': False,
    })
    for arg in overrides:
        key, _, value = arg.lstrip('+~').partition('=')
        if key == 'trainer.del_local_ckpt_after_load' and value.lower() != 'false':
            raise ConfigError('Weight-only import must preserve the source checkpoint')
        if key.startswith('actor_rollout_ref.model.override_config') and arg not in saved_overrides:
            raise ConfigError('Weight-only import cannot change saved actor architecture overrides')
        if key == 'actor_rollout_ref.model.path' or key in {'actor_rollout_ref.model', 'actor_rollout_ref',
                'actor_rollout_ref.actor', 'actor_rollout_ref.actor.checkpoint'}:
            raise ConfigError('Weight-only import requires MODEL_PATH and an explicit leaf load_contents override')
    overrides[:0] = saved_overrides
    overrides.append('trainer.del_local_ckpt_after_load=False')


def validate_thinking_resume(env):
    if env.get('STEP_WEIGHT_IMPORT') or env.get('RUN_IS_EVAL') == '1' or env.get('RESUME_MODE', 'disable') == 'disable':
        return
    if env.get('RESUME_MODE') != 'resume_path':
        if qwen35_run(env):
            raise ConfigError('Qwen3.5 training resume requires explicit RESUME_MODE=resume_path and RESUME_CKPT')
        return
    _, saved = checkpoint_model_config(env.get('RESUME_CKPT', ''))
    if qwen35_run(env) or (saved and qwen35_run(saved['model'])):
        if not saved or saved.get('thinking') != QWEN35_THINKING:
            raise ConfigError('Cannot resume a legacy Qwen3.5 thinking profile for no-thinking training; start a new run')


def thinking_protocol_metadata(env, benchmark, algo, command):
    result = {}
    thinking = thinking_metadata(env)
    if thinking is not None:
        result['thinking'] = thinking
    values = dict(arg.lstrip('+').split('=', 1) for arg in command[3:] if '=' in arg)
    if values.get('algorithm.step_rollout.evaluation_only', '').lower() == 'true':
        result['evaluation_protocol'] = {
            'task_prompt_protocol': task_prompt_protocol(benchmark),
            'profile': values['algorithm.step_rollout.profile'],
            'history_length': int(values['algorithm.step_rollout.history_length']),
            'action_interface': values['algorithm.step_rollout.action_interface'],
            'thinking': thinking,
        }
    elif values.get('algorithm.step_rollout.enabled', '').lower() == 'true':
        fields = dict(STEP_PARAMETER_FIELDS)
        fields.update({
            'STEP_ROLLOUT_ENABLED': 'algorithm.step_rollout.enabled',
            'STEP_PROFILE': 'algorithm.step_rollout.profile',
            'STEP_HISTORY_LENGTH': 'algorithm.step_rollout.history_length',
            'STEP_INVALID_ACTION_PENALTY': 'algorithm.step_rollout.invalid_action_penalty',
            'TRAIN_BATCH_SIZE': 'data.train_batch_size', 'ROLLOUT_N': 'actor_rollout_ref.rollout.n',
            'PPO_EPOCHS': 'actor_rollout_ref.actor.ppo_epochs',
            'MAX_TOOL_TURNS': 'actor_rollout_ref.rollout.multi_turn.max_user_turns',
        })
        parameters = {key: values[field] for key, field in fields.items()
                      if key in STEP_PROFILE_KEYS and field in values}
        result['training_protocol'] = {
            'version': 2, 'estimator': values['algorithm.adv_estimator'],
            'partition_protocol': PARTITION_PROTOCOL,
            'task_prompt_protocol': task_prompt_protocol(benchmark),
            'action_interface': 'dyad' if values['algorithm.step_rollout.action_interface'] == 'dyad' else 'react',
            'profile': values['algorithm.step_rollout.profile'], 'thinking': thinking,
            'parameters': parameters,
            'step_rollout': {key.removeprefix('algorithm.step_rollout.'): value
                             for key, value in values.items() if key.startswith('algorithm.step_rollout.')},
            'loss_reduction': 'reference_microbatch', 'resampling': 'reference_copy',
        }
        if env.get('STEP_WEIGHT_IMPORT'):
            result['weight_import'] = json.loads(env['STEP_WEIGHT_IMPORT'])
            result['weight_import']['effective_training_protocol'] = result['training_protocol']
            result['weight_import']['effective_action_schema'] = env.get('DYAD_ACTION_YAML')
        if env.get('STEP_SOURCE_PROTOCOL'):
            result['evaluation_protocol'] = {
                'source_training_protocol': json.loads(env['STEP_SOURCE_PROTOCOL']),
                'effective_training_protocol': result['training_protocol'], 'protocol_changed': True,
            }
    return result


def configure_step_profile(env, benchmark, algo, overrides):
    """Require v2; old checkpoints need explicit model-only import into a new run."""
    if benchmark not in {'alfworld', 'webshop', 'dive', 'codegym'}:
        return
    checkpoint = env.get('RESUME_CKPT')
    saved = None
    if checkpoint and env.get('RESUME_MODE') == 'resume_path':
        _, saved = checkpoint_model_config(checkpoint, env.get('EVAL_MODEL_CONFIG'))
    protocol = (saved or {}).get('training_protocol')
    evaluation = env.get('RUN_IS_EVAL') == '1'
    if evaluation and checkpoint and (not protocol or protocol.get('version') != 2):
        raise ConfigError('Old interaction protocols were removed; checkpoint evaluation requires step v2. '
                          'Use explicit model-only import to start a new step v2 run.')
    weight_import = bool(env.get('STEP_WEIGHT_IMPORT'))
    if weight_import:
        protocol = None
    if not evaluation and not weight_import and env.get('RESUME_MODE', 'disable') != 'disable':
        if env.get('RESUME_MODE') != 'resume_path':
            raise ConfigError('Step training requires explicit RESUME_MODE=resume_path and RESUME_CKPT')
        if not protocol or protocol.get('version') != 2:
            raise ConfigError('Cannot full-state resume a legacy checkpoint under the shared step protocol; '
                              'start a new weight-only run')
        if protocol.get('partition_protocol') != PARTITION_PROTOCOL:
            raise ConfigError('Cannot full-state resume a checkpoint with missing or different partition_protocol; '
                              'use an explicit model-only import to start a new run')
        if protocol.get('task_prompt_protocol') != task_prompt_protocol(benchmark):
            raise ConfigError('Cannot full-state resume a checkpoint with missing or different task_prompt_protocol; '
                              'use an explicit model-only import to start a new run')
        selection = apply_algorithm(env, algo)
        if (protocol.get('estimator') != selection.adv_estimator
                or protocol.get('action_interface') != selection.action_interface
                or (saved or {}).get('benchmark') != benchmark):
            raise ConfigError('Checkpoint step protocol identity does not match training selection')
    env['STEP_ROLLOUT_ENABLED'] = 'True'
    if protocol and evaluation:
        if saved.get('benchmark') != benchmark:
            env['STEP_SOURCE_PROTOCOL'] = json.dumps(protocol)
            protocol = None
        elif protocol.get('task_prompt_protocol') != task_prompt_protocol(benchmark):
            # Preserve saved budgets while recording the new task semantics explicitly.
            env['STEP_SOURCE_PROTOCOL'] = json.dumps(protocol)
    if protocol:
        if protocol.get('version') != 2 or protocol.get('loss_reduction') != 'reference_microbatch' or protocol.get('resampling') != 'reference_copy':
            raise ConfigError('Invalid or unsupported checkpoint shared step protocol')
        parameters = protocol.get('parameters', {})
        if not isinstance(parameters, dict):
            raise ConfigError('Invalid checkpoint step profile parameters')
        for key, value in parameters.items():
            if key not in STEP_PROFILE_KEYS or not isinstance(value, str):
                raise ConfigError('Invalid checkpoint step profile parameter: ' + key)
            if env.get(key) and env[key] != value:
                raise ConfigError(key + ' conflicts with checkpoint step protocol')
            env[key] = value
        env['STEP_SAVED_PROTOCOL'] = json.dumps(protocol)
    for key in ('GIGPO_ROLLOUT_LAYOUT', 'GIGPO_HISTORY_LENGTH', 'GIGPO_INVALID_ACTION_PENALTY'):
        if env.get(key):
            raise ConfigError(key + ' was removed; use the shared step v2 profile')
    for arg in overrides:
        key = arg.lstrip('+~').partition('=')[0]
        if key.startswith('algorithm.gigpo.rollout_layout'):
            raise ConfigError('Use the shared step protocol, not algorithm.gigpo.rollout_layout')


# Configuration parsing lives with preparation; config/ contains data only.
class ConfigError(ValueError):
    pass


def load_config(path):
    import yaml
    class UniqueLoader(yaml.SafeLoader):
        pass
    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ConfigError(f'{path}: duplicate key {key!r}')
            result[key] = loader.construct_object(value_node, deep=deep)
        return result
    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        data = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f'Cannot load {path}: {exc}') from exc
    if not isinstance(data, dict):
        raise ConfigError(f'{path}: expected a mapping')
    return data


def config_scalar(value):
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if not isinstance(value, (str, int, float)):
        raise ConfigError(f'Expected a scalar parameter, got {value!r}')
    return str(value)


PARAMETER_ALIASES = {'learning_rate': 'LR', 'trainer_gpus': 'N_GPUS_PER_NODE',
                     'encoder_gpus': 'DYAD_ENCODER_NUM_GPUS'}


def config_parameters(block):
    if not isinstance(block, dict):
        raise ConfigError('Parameter block must be a mapping')
    result = {}
    for key, value in block.items():
        name = PARAMETER_ALIASES.get(key, key.upper())
        if name not in SCALE_KEYS:
            raise ConfigError(f'Unknown parameter {key!r}; add a supported parameter in prepare.py')
        if name in result:
            raise ConfigError(f'Duplicate parameter aliases for {name}')
        result[name] = config_scalar(value)
    return result


def experiment_recipes():
    doc = load_config(PROJECT_DIR / 'experiments/dyad_training/agentic_rl/train_eval/config/experiments.yaml')
    if set(doc) != {'main'}:
        raise ConfigError('experiments.yaml only accepts main; put each ablation in ablation/<name>.yaml')
    def convert(block):
        result = {}
        for key, value in block.items():
            name = 'DYAD_' + key.upper()
            if name not in MODEL_KEYS | {'DYAD_ENCODER_REMOTE'}:
                raise ConfigError(f'Unknown experiment parameter {key}')
            result[name] = ('1' if value else '0') if isinstance(value, bool) and name in {'DYAD_ENCODER_ENABLED', 'DYAD_ENCODER_REMOTE'} else config_scalar(value)
        return result
    result = {'train': {'defaults': convert(doc['main']['defaults']), 'envs': doc['main']['benchmarks'],
                        'parameters': config_parameters(doc['main'].get('parameters', {}))}}
    for path in sorted((PROJECT_DIR / 'experiments/dyad_training/agentic_rl/train_eval/ablation').glob('*.yaml')):
        name, entry = path.stem, load_config(path)
        if name == 'train' or set(entry) - {'title', 'description', 'overrides', 'benchmarks', 'parameters'}:
            raise ConfigError(f'{path}: invalid ablation name or fields')
        if not isinstance(entry.get('benchmarks'), list) or not entry['benchmarks'] or set(entry['benchmarks']) - set(doc['main']['benchmarks']):
            raise ConfigError(f'{path}: benchmarks must be a nonempty subset of main.benchmarks')
        result[name] = {'overrides': convert(entry.get('overrides', {})), 'envs': entry['benchmarks'],
                        'parameters': config_parameters(entry.get('parameters', {}))}
    return result


def canonical_model(model):
    return model.rsplit('/', 1)[-1].lower().replace('-instruction', '-instruct')


def hardware():
    return {'a6000': {'match': ['RTX A6000']}, 'h100': {'match': ['H100']},
            'h200': {'match': ['H200']}, 'debug': {'match': []}}


def detect_profile(gpu_name):
    for name, entry in hardware().items():
        if any(s.lower() in gpu_name.lower() for s in entry['match']):
            return name
    return None


def benchmark_config(benchmark, algo):
    doc = load_config(TRAIN_EVAL / f'config/{benchmark}.yaml')
    if algo not in doc['algorithms']:
        raise ConfigError(f'{benchmark} has no algorithm {algo}')
    unknown = set(doc) - {'defaults', 'algorithms', 'models', 'debug', 'step_profile'}
    if unknown:
        raise ConfigError(f'{benchmark}: unknown sections {sorted(unknown)}')
    def sections(block, allowed, where):
        if not isinstance(block, dict):
            raise ConfigError(f'{where}: expected a mapping')
        extra = set(block) - allowed
        if extra:
            raise ConfigError(f'{where}: unknown sections {sorted(extra)}')
    config_parameters(doc.get('defaults', {}))
    config_parameters(doc.get('step_profile', {}))
    for values in doc['algorithms'].values():
        config_parameters(values)
    for model, block in doc['models'].items():
        sections(block, {'defaults', 'hardware', 'note'}, model)
        config_parameters(block.get('defaults', {}))
        for hw, entry in block.get('hardware', {}).items():
            sections(entry, {'defaults', 'algorithms', 'experiments', 'note'}, model + '/' + hw)
            config_parameters(entry.get('defaults', {}))
            for group in ('algorithms', 'experiments'):
                for parameters in entry.get(group, {}).values():
                    config_parameters(parameters)
            if set(entry.get('algorithms', {})) - set(doc['algorithms']):
                raise ConfigError(f'{model}/{hw}: unknown algorithm override')
    dbg = doc.get('debug', {})
    sections(dbg, {'defaults', 'algorithms', 'experiments', 'step_profile'}, 'debug')
    config_parameters(dbg.get('defaults', {}))
    config_parameters(dbg.get('step_profile', {}))
    for group in ('algorithms', 'experiments'):
        for parameters in dbg.get(group, {}).values():
            config_parameters(parameters)
    return doc


def profiles_of(benchmark, algo, model='Qwen2.5-0.5B-Instruct'):
    doc = benchmark_config(benchmark, algo)
    name = canonical_model(model)
    if name not in doc['models']:
        raise ConfigError(f'Unknown model {model}; add models.{name} to config/{benchmark}.yaml')
    return dict(doc['models'][name].get('hardware', {}), debug=doc.get('debug', {}))


def choose_profile(benchmark, algo, *, explicit, is_debug, gpu_name, model='Qwen2.5-0.5B-Instruct'):
    profile = explicit or detect_profile(gpu_name)
    # Legacy SCALE_PROFILE=debug is an explicit A6000 smoke selection, never auto-selected.
    if not profile:
        raise ConfigError('Cannot identify GPU; use --hardware a6000|h100|h200')
    available = profiles_of(benchmark, algo, model)
    if profile not in available or (profile == 'debug' and 'a6000' not in available):
        raise ConfigError(f'No {model}/{profile} settings in config/{benchmark}.yaml; available: {list(available)}')
    return profile


def resolve(benchmark, algo, profile, model='Qwen2.5-0.5B-Instruct', *, debug=False, experiment='train'):
    doc = benchmark_config(benchmark, algo)
    name = canonical_model(model)
    if name not in doc['models']:
        raise ConfigError(f'Unknown model {model}; add models.{name} to config/{benchmark}.yaml')
    model_block = doc['models'][name]
    selected = 'a6000' if profile == 'debug' else profile
    if selected not in model_block.get('hardware', {}):
        raise ConfigError(f'No {model}/{selected} settings in config/{benchmark}.yaml')
    hw = model_block['hardware'][selected]
    values = dict(GIGPO_DEFAULTS) if algo == 'gigpo' else {}
    def merge(block):
        values.update(config_parameters(block))
    def merge_algorithm(block):
        # Keep every baseline config layer aligned, with explicit GiGPO overrides last.
        if algo == 'gigpo':
            merge(block.get('grpo_react', {}))
        merge(block.get(algo, {}))
    merge(doc.get('defaults', {})); merge_algorithm(doc['algorithms'])
    merge(model_block.get('defaults', {})); merge(hw.get('defaults', {}))
    merge_algorithm(hw.get('algorithms', {}))
    if algo == 'dyad':
        recipes = experiment_recipes()
        if experiment not in recipes:
            raise ConfigError(f'Unknown experiment {experiment}')
        values.update(recipes['train'].get('parameters', {}))
        if experiment != 'train':
            values.update(recipes[experiment].get('parameters', {}))
    elif experiment != 'train':
        raise ConfigError('Ablations require dyad')
    merge(hw.get('experiments', {}).get(experiment, {}))
    if debug or profile == 'debug':
        dbg = doc.get('debug', {})
        merge(dbg.get('defaults', {})); merge_algorithm(dbg.get('algorithms', {}))
        # Debug GPU topology remains the selected hardware topology unless explicitly stated here.
        merge(dbg.get('experiments', {}).get(experiment, {}))
    shared_step = bool(doc.get('step_profile'))
    if shared_step:
        # Environment semantics and optimizer budget are shared by every policy/estimator.
        merge(doc['step_profile'])
        if debug or profile == 'debug':
            merge(doc.get('debug', {}).get('step_profile', {}))
    budget = training_budget(benchmark, debug=debug or profile == 'debug',
                             step_profile=shared_step)
    for key, expected in budget.items():
        if values.get(key) != expected:
            raise ConfigError(f'{benchmark}: {key} must be {expected} in every model/hardware/experiment; '
                              'adjust MICRO_BATCH_SIZE instead')
    return values


def training_budget(benchmark, *, debug=False, step_profile=False):
    document = load_config(TRAIN_EVAL / f'config/{benchmark}.yaml')
    defaults = dict(document['defaults'])
    if step_profile:
        defaults.update(document.get('step_profile', {}))
    if debug:
        defaults.update(document.get('debug', {}).get('defaults', {}))
        if step_profile:
            defaults.update(document.get('debug', {}).get('step_profile', {}))
    return {key.upper(): str(defaults[key]) for key in ('train_batch_size', 'ppo_mini_batch_size', 'rollout_n', 'ppo_epochs')}


def validate_training_budget(env, benchmark, overrides=()):
    """Reject environment/Hydra changes to an env's shared sampling/update budget."""
    debug = env.get('RUN_IS_DEBUG') == '1' or env.get('SCALE_PROFILE') == 'debug'
    shared_step = env.get('STEP_ROLLOUT_ENABLED') == 'True'
    step_profile = shared_step
    budget = training_budget(benchmark, debug=debug, step_profile=step_profile)
    if step_profile and debug:
        # Debug runs may explicitly choose a positive step mini-batch size.
        mini = env.get('PPO_MINI_BATCH_SIZE', budget['PPO_MINI_BATCH_SIZE'])
        for arg in overrides:
            if arg.lstrip('+').startswith('actor_rollout_ref.actor.ppo_mini_batch_size='):
                mini = arg.partition('=')[2]
        if not str(mini).isdigit() or int(mini) < 1:
            raise ConfigError('GiGPO debug step mini-batch must be a positive integer')
        budget['PPO_MINI_BATCH_SIZE'] = str(mini)
        env['PPO_MINI_BATCH_SIZE'] = str(mini)
    for key, expected in budget.items():
        if str(env.get(key, expected)) != expected:
            raise ConfigError(f'{benchmark}: {key} must be {expected}; adjust MICRO_BATCH_SIZE instead')
    protected = {
        'data.train_batch_size': budget['TRAIN_BATCH_SIZE'],
        'data.gen_batch_size': budget['TRAIN_BATCH_SIZE'],
        'actor_rollout_ref.actor.ppo_mini_batch_size': budget['PPO_MINI_BATCH_SIZE'],
        'actor_rollout_ref.rollout.n': budget['ROLLOUT_N'],
        'actor_rollout_ref.actor.ppo_epochs': budget['PPO_EPOCHS'],
    }
    for arg in overrides:
        key, _, value = arg.lstrip('+~').partition('=')
        if key in protected and (arg.startswith('~') or value != protected[key]):
            raise ConfigError(f'{benchmark}: {key} must be {protected[key]}; remove override {arg}')
    if step_profile and env.get('TOTAL_TRAINING_STEPS'):
        iterations = {'trainer.total_epochs': env['TOTAL_EPOCHS'],
                      'trainer.total_training_steps': env['TOTAL_TRAINING_STEPS']}
        for arg in overrides:
            key, _, value = arg.lstrip('+~').partition('=')
            if key in iterations:
                iterations[key] = value if not arg.startswith('~') else ''
        if any(not str(value).isdigit() or int(value) < 1 for value in iterations.values()):
            raise ConfigError('GiGPO step profile requires positive epoch and iteration limits')
        if int(iterations['trainer.total_epochs']) < int(iterations['trainer.total_training_steps']):
            raise ConfigError('GiGPO step profile requires TOTAL_EPOCHS >= TOTAL_TRAINING_STEPS '
                              'so even a single-batch epoch can reach its rollout iteration budget')


def require_initial_training_validation(env, overrides):
    """Validate loaded weights before updates, independently of periodic validation."""
    if env.get('VAL_BEFORE_TRAIN', 'True').lower() != 'true':
        raise ConfigError('Agentic RL training requires VAL_BEFORE_TRAIN=True; initial validation cannot be disabled')
    for arg in overrides:
        stripped = arg.lstrip('+~')
        key, _, value = stripped.partition('=')
        prefix = arg[:len(arg) - len(stripped)]
        # Replacing/deleting the parent can bypass a leaf-only override check.
        if key == 'trainer' or (key == 'trainer.val_before_train' and (
                prefix not in ('', '+', '++') or value.lower() != 'true')):
            raise ConfigError('Agentic RL training requires trainer.val_before_train=True; remove override ' + arg)
    env['VAL_BEFORE_TRAIN'] = 'True'


def validate_gigpo_selection(env, algo, overrides):
    """Reject identity changes and unsupported GiGPO contracts before any execution."""
    selection = apply_algorithm(env, algo)
    algo = selection.algo
    expected_advantage = selection.adv_estimator
    shared_step = env.get('STEP_ROLLOUT_ENABLED') == 'True'
    if shared_step:
        dynamic = {'algorithm.filter_groups.enable': 'false',
                   'algorithm.filter_groups.max_num_gen_batches': '0',
                   'algorithm.filter_groups.metric': 'null',
                   'algorithm.filter_groups.max_inflight_gen_batches': '1',
                   'trainer.v1.sampler.sync_refill_failed_groups': 'false',
                   'trainer.v1.sampler.custom_sampler.path': 'null',
                   'trainer.v1.sampler.custom_sampler.name': 'null'}
        for arg in overrides:
            key, _, value = arg.lstrip('+~').partition('=')
            if key in dynamic:
                dynamic[key] = value.lower()
        if dynamic['algorithm.filter_groups.enable'] == 'true':
            attempts = dynamic['algorithm.filter_groups.max_num_gen_batches']
            if not attempts.isdigit() or int(attempts) < 1:
                raise ConfigError('Shared dynamic sampling requires positive max_num_gen_batches')
            expected = {'algorithm.filter_groups.max_inflight_gen_batches': '1',
                        'trainer.v1.sampler.sync_refill_failed_groups': 'false',
                        'trainer.v1.sampler.custom_sampler.path': 'null',
                        'trainer.v1.sampler.custom_sampler.name': 'null'}
            if dynamic['algorithm.filter_groups.metric'] not in {'null', 'episode_return'} or any(
                    dynamic[key] != value for key, value in expected.items()):
                raise ConfigError('Shared dynamic sampling requires episode_return and built-in whole-batch retries')
        env['STEP_ACTION_INTERFACE'] = 'dyad' if selection.action_interface == 'dyad' else 'text'
        fixed_step = {
            'algorithm.step_rollout.enabled': 'true',
            'algorithm.step_rollout.protocol_version': '2',
            'algorithm.rollout_correction.bypass_mode': 'false',
            'algorithm.rollout_correction.rollout_is': 'null',
            'algorithm.rollout_correction.rollout_rs': 'null',
            'actor_rollout_ref.actor.use_rollout_log_probs': 'false',
            'actor_rollout_ref.actor.shuffle': 'false',
            'reward.reward_model.enable': 'false',
            'algorithm.step_rollout.profile': env['STEP_PROFILE'],
            'algorithm.step_rollout.action_interface': env['STEP_ACTION_INTERFACE'],
            'algorithm.step_rollout.history_length': env['STEP_HISTORY_LENGTH'],
            'algorithm.step_rollout.max_steps': env['MAX_ASSISTANT_TURNS'],
            'algorithm.step_rollout.invalid_action_penalty': env['STEP_INVALID_ACTION_PENALTY'],
            'algorithm.step_rollout.resampling': 'reference_copy',
            'algorithm.step_rollout.loss_reduction': 'reference_microbatch',
            'actor_rollout_ref.actor.ppo_mini_batch_size_unit': 'step',
            'actor_rollout_ref.rollout.agent.default_agent_loop': 'environment_step_agent',
            'actor_rollout_ref.rollout.agent.agent_loop_config_path': str(
                PROJECT_DIR / 'agent_system/rollout/environment_step_agent_loop.yaml').lower(),
            'trainer.use_v1': 'true', 'trainer.v1.trainer_mode': 'sync',
            'trainer.v1.sync.parameter_sync_step': '1',
            'trainer.resume_mode': env.get('RESUME_MODE', 'disable'),
            'trainer.resume_from_path': env.get('RESUME_CKPT', ''),
            'data.tool_config_path': 'null', 'data.function_tool_path': 'null',
            'data.filter_overlong_prompts': 'false',
        }
        for arg in overrides:
            key, _, value = arg.lstrip('+~').partition('=')
            if (any(field.startswith(key + '.') for field in fixed_step)
                    or (key in fixed_step and (arg.startswith('~') or value.lower() != fixed_step[key]))):
                raise ConfigError('Shared step protocol owns this parameter; remove override ' + arg)
    for arg in overrides:
        key, _, value = arg.lstrip('+~').partition('=')
        if key == 'algorithm':
            raise ConfigError(f'{selection.public_alias} owns its loss identity; do not replace the algorithm configuration')
        profile_contract = {
            'actor_rollout_ref.actor.ppo_mini_batch_size_unit': 'step' if shared_step else 'trajectory',
        }
        if key in profile_contract and (arg.startswith('~') or value != profile_contract[key]):
            raise ConfigError('GiGPO rollout layout and mini-batch unit belong to the selected profile')
        if key == 'algorithm.adv_estimator':
            if arg.startswith('~') or value != expected_advantage:
                raise ConfigError('Select dyad-grpo, dyad-gigpo or gigpo through the public algorithm argument, not an advantage override')
    if algo == 'dyad':
        env['DYAD_ADV_ESTIMATOR'] = expected_advantage
        identity = selection.public_alias
        if env.get('RUN_ALGO') and env['RUN_ALGO'] != identity:
            raise ConfigError(f'Dyad requires RUN_ALGO={identity} to preserve loss identity')
        env['RUN_ALGO'] = identity
    if expected_advantage != 'gigpo':
        return
    for key, value in GIGPO_DEFAULTS.items():
        default(env, key, value)
    if algo == 'gigpo':
        if env.get('DYAD_ENCODER_ENABLED', '0') not in ('0', 'False', 'false'):
            raise ConfigError('The GiGPO text baseline cannot enable the Dyad encoder; select dyad-gigpo')
        if env.get('RUN_ALGO') and env['RUN_ALGO'] != 'gigpo':
            raise ConfigError('GiGPO requires RUN_ALGO=gigpo to preserve run identity')
    fixed = {
        'algorithm.adv_estimator': 'gigpo',
        'algorithm.use_kl_in_reward': 'false',
        'algorithm.filter_groups.enable': 'false',
        'algorithm.gigpo.enable_similarity': 'false',
        'actor_rollout_ref.actor.strategy': 'dyad' if algo == 'dyad' else 'fsdp',
        'actor_rollout_ref.actor.policy_loss.loss_mode': 'vanilla',
        'actor_rollout_ref.rollout.name': 'dyadvllm' if algo == 'dyad' else 'vllm',
        'actor_rollout_ref.rollout.agent.default_agent_loop': 'environment_step_agent',
        'actor_rollout_ref.rollout.agent.agent_loop_config_path': str(
            PROJECT_DIR / 'agent_system/rollout/environment_step_agent_loop.yaml').lower(),
        'actor_rollout_ref.rollout.multi_turn.enable': 'true',
        'actor_rollout_ref.rollout.multi_turn.format': 'dyad' if algo == 'dyad' else ('dive' if env.get('RUN_ENV') == 'dive' else 'react'),
        'data.continuous_token.enable': 'false',
    }
    if shared_step:
        fixed.update(fixed_step)
        fixed.pop('algorithm.gigpo.enable_similarity', None)
        fixed.pop('algorithm.filter_groups.enable', None)
    for arg in overrides:
        key, _, value = arg.lstrip('+~').partition('=')
        parent = any(field.startswith(key + '.') for field in fixed)
        if parent or (key in fixed and (arg.startswith('~') or value.lower() != fixed[key])):
            raise ConfigError('GiGPO owns this algorithm/rollout contract; remove override ' + arg)
    values = {'algorithm.gamma': env['GIGPO_GAMMA'], 'algorithm.gigpo.mode': env['GIGPO_MODE'],
              'algorithm.gigpo.step_advantage_w': env['GIGPO_STEP_ADVANTAGE_W']}
    for arg in overrides:
        key, _, value = arg.lstrip('+~').partition('=')
        if key in values:
            if arg.startswith('~'):
                raise ConfigError('GiGPO requires parameter ' + key)
            values[key] = value
    import math
    try:
        gamma = float(values['algorithm.gamma'])
        weight = float(values['algorithm.gigpo.step_advantage_w'])
    except ValueError as exc:
        raise ConfigError('GiGPO gamma and step advantage weight must be finite numbers') from exc
    if not math.isfinite(gamma) or not 0 <= gamma <= 1 or not math.isfinite(weight) or weight < 0:
        raise ConfigError('GiGPO requires gamma in [0,1] and finite nonnegative step advantage weight')
    if values['algorithm.gigpo.mode'] not in {'mean_norm', 'mean_std_norm'}:
        raise ConfigError('GiGPO mode must be mean_norm or mean_std_norm')


def scale_config(env, benchmark, algo):
    gpu_name = ''
    import shutil
    if shutil.which('nvidia-smi'):
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
        gpu_name = next(iter(result.stdout.splitlines()), '')
    profile = choose_profile(benchmark, algo, explicit=env.get('HARDWARE_PROFILE') or env.get('SCALE_PROFILE'),
                             is_debug=env.get('RUN_IS_DEBUG') == '1', gpu_name=gpu_name, model=env['MODEL_NAME'])
    values = resolve(benchmark, algo, profile, env['MODEL_NAME'], debug=env.get('RUN_IS_DEBUG') == '1',
                     experiment=env.get('TRAINING_EXPERIMENT', 'train') if env.get('RUN_IS_EVAL') != '1' else 'train')
    for key, value in values.items():
        default(env, key, value)
    env['SCALE_PROFILE'] = profile
    if env.get('RUN_IS_EVAL') != '1':
        validate_training_budget(env, benchmark)
    env['EFFECTIVE_CPUS'] = str(effective_cpus())


@contextlib.contextmanager
def environment(values):
    before = os.environ.copy()
    os.environ.clear()
    os.environ.update(values)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)


def debug_config(env, benchmark, algo):
    if env.get('RUN_IS_DEBUG') != '1':
        if env.get('RUN_IS_EVAL') != '1':
            default(env, 'GRPO_PRINT_STEP_ACTION', '0')
            default(env, 'GRPO_PRINT_ENV_STEP', '0')
        return
    for key, value in {
        'SKIP_TRAINER_PREFLIGHT': '1', 'DYAD_DIAG_ENABLED': '1', 'DYAD_DIAG_LEVEL': 'full',
        'DYAD_DIAG_MAX_VALUES': '64', 'GRPO_PRINT_STEP_ACTION': '1', 'GRPO_PRINT_ENV_STEP': '1',
        'GRPO_FULL_DUMP_LIMIT': '64', 'GRPO_TOOL_TIMEOUT_MS': '0',
    }.items():
        default(env, key, value)
    if env.get('RUN_IS_EVAL') == '1':
        default(env, 'VAL_MAX_SAMPLES', '8')
        default(env, 'VAL_BATCH_SIZE', '4')
        # ALFWorld/CodeGym pools reserve the actual dispatched shard, including validation.



def resolve_model(env):
    if not env.get('MODEL_PATH'):
        cache = Path(env.get('HF_HUB_DIR') or env.get('HF_HUB_CACHE')
                     or Path(env.get('HF_HOME', str(Path.home() / '.cache/huggingface'))) / 'hub').expanduser()
        paths = sorted(p for p in cache.glob('models--*--*/snapshots/*/config.json')
                       if canonical_model(p.parents[2].name.split('--')[-1]) == canonical_model(env['MODEL_NAME']))
        if not paths:
            raise ValueError('Cannot resolve model; set MODEL_PATH to a local model directory')
        env['MODEL_PATH'] = str(paths[0].parent)
    if not (Path(env['MODEL_PATH']) / 'config.json').is_file():
        raise ValueError('MODEL_PATH must contain config.json')
    env['MODEL_PATH'] = str(Path(env['MODEL_PATH']).resolve())
    default(env, 'MODEL_TAG', env['MODEL_NAME'].split('/')[-1].lower())
    default(env, 'DYAD_MODEL_SOURCE_PATH', env['MODEL_PATH'])


def output_layout(env, benchmark, algo):
    # Retired console-copy setting: tolerate old launch YAMLs and entrypoints.
    env.pop('DYAD_LOG_FILE', None)
    env['PYTHONUNBUFFERED'] = '1'
    site = run_site(env)
    storage = artifact_root(PROJECT_DIR, env)
    env.update(ARTIFACT_ROOT=str(storage), RUN_SITE=site)
    out, ckpt = storage / 'outputs', storage / 'ckpt'
    default(env, 'TIME_SUFFIX', datetime.now().strftime('%Y%m%d_%H%M%S'))
    default(env, 'RUN_ENV', benchmark)
    selection = apply_algorithm(env, algo)
    default(env, 'RUN_ALGO', selection.public_alias)
    tag = env['TIME_SUFFIX']
    slug = env.get('RUN_SLUG') or (env.get('ENV_NAME') if benchmark == 'codegym' and algo == 'dyad' else '')
    if slug:
        tag += '_' + slug
    if env.get('RUN_IS_DEBUG') == '1':
        tag += '_debug'
    if env.get('RUN_IS_EVAL') == '1':
        tag += '_bench'
    tag += '_' + env.get('RUN_MODEL_TAG', env.get('MODEL_TAG', env['MODEL_NAME'].lower())).replace('-', '_')
    default(env, 'RUN_ID', tag)
    experiment = 'bench' if env.get('RUN_IS_EVAL') == '1' else env.get('TRAINING_EXPERIMENT', 'train')
    prefix = env['RUN_ALGO'] + '-' if algo == 'dyad' else ''
    default(env, 'EXP_NAME', prefix + experiment + '-' + tag)
    default(env, 'OUTPUT_ROOT', out / site / 'agentic_rl')
    default(env, 'CKPT_ROOT', ckpt / site / 'agentic_rl')
    relative = Path(env['RUN_ALGO']) / benchmark / env['RUN_ID']
    default(env, 'RUN_DIR', Path(env['OUTPUT_ROOT']) / relative)
    default(env, 'CKPT_DIR', Path(env['CKPT_ROOT']) / relative)
    for key in ('DYAD_DIAG_DIR', 'HYDRA_RUN_DIR'):
        default(env, key, env['RUN_DIR'])
    default(env, 'VALIDATION_DATA_DIR', Path(env['RUN_DIR']) / 'val_generations')
    default(env, 'WANDB_DIR', env['RUN_DIR'])
    for key, subdir in (('WANDB_CACHE_DIR', 'cache'), ('WANDB_CONFIG_DIR', 'config'),
                        ('WANDB_DATA_DIR', 'data')):
        default(env, key, Path(env['RUN_DIR']) / 'wandb' / subdir)
    want = 'offline' if env.get('RUN_IS_DEBUG') == '1' else 'online'
    if env.get('WANDB_MODE') and env['WANDB_MODE'] != want:
        print(f'[prepare] WANDB_MODE={env["WANDB_MODE"]} -> {want}')
    env['WANDB_MODE'] = want
    validate_output_paths(env)


def validate_output_paths(env):
    """Reject path escapes, including symlinks, before any persistent write."""
    storage = artifact_root(PROJECT_DIR, env)
    run = Path(env['RUN_DIR']).expanduser().resolve()
    for key in ('OUTPUT_ROOT', 'RUN_DIR', 'CKPT_ROOT', 'CKPT_DIR', 'DYAD_DIAG_DIR',
                'HYDRA_RUN_DIR', 'VALIDATION_DATA_DIR', 'WANDB_DIR', 'WANDB_CACHE_DIR',
                'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR'):
        if not env.get(key):
            continue
        if key in {'CKPT_ROOT', 'CKPT_DIR'}:
            root = storage / 'ckpt'
        elif key in {'OUTPUT_ROOT', 'RUN_DIR'}:
            root = storage / 'outputs'
        else:
            root = run
        raw = Path(env[key]).expanduser().absolute()
        if any(path.is_symlink() for path in (raw, *raw.parents)):
            raise ValueError(f'{key} must not use a symlinked output path: {raw}')
        value = raw.resolve()
        if not value.is_relative_to(root) or (key in {'RUN_DIR', 'CKPT_DIR'} and value == root):
            raise ValueError(f'{key} must be under {root}; got {value}')
        env[key] = str(value)


def configure_rollout_output(env, mode, overrides):
    """DYAD-OUTPUT: keep training trajectory dumps inside this run, including resume."""
    field = 'trainer.rollout_data_dir'
    selected = str(Path(env['RUN_DIR']) / 'rollout') if mode == 'training' else None
    retained = []
    for argument in overrides:
        key = argument.lstrip('+~').partition('=')[0]
        if key != field:
            if field.startswith(key + '.') or key.startswith(field + '.'):
                raise ConfigError('Set trainer.rollout_data_dir directly; output parent overrides are not allowed')
            retained.append(argument)
            continue
        from hydra.core.override_parser.overrides_parser import OverridesParser

        try:
            parsed = OverridesParser.create().parse_override(argument)
            value = parsed.value()
        except Exception as exc:
            raise ConfigError('trainer.rollout_data_dir must be one literal absolute path') from exc
        if parsed.is_delete() or parsed.is_sweep_override():
            raise ConfigError('trainer.rollout_data_dir cannot be deleted or swept')
        if value is None and mode == 'evaluation':
            selected = None
            continue
        if not isinstance(value, str) or not value or '${' in value:
            raise ConfigError('trainer.rollout_data_dir must be one literal absolute path')
        raw = Path(value).expanduser()
        if not raw.is_absolute():
            raise ConfigError('trainer.rollout_data_dir must be an absolute path inside RUN_DIR')
        if any(path.is_symlink() for path in (raw, *raw.parents)):
            raise ConfigError('trainer.rollout_data_dir must not use a symlinked output path')
        run, destination = Path(env['RUN_DIR']).resolve(), raw.resolve()
        if destination == run or not destination.is_relative_to(run):
            raise ConfigError('trainer.rollout_data_dir must be strictly inside RUN_DIR')
        # Validate every occurrence, so an earlier escape is never silently hidden.
        selected = str(destination)
    if selected is not None:
        # The default needs the same symlink guard as an explicit destination.
        raw = Path(selected)
        if any(path.is_symlink() for path in (raw, *raw.parents)):
            raise ConfigError('trainer.rollout_data_dir must not use a symlinked output path')
        retained.append(field + '=' + json.dumps(selected))
    elif mode == 'evaluation' and any(arg.lstrip('+~').partition('=')[0] == field for arg in overrides):
        retained.append(field + '=null')
    overrides[:] = retained


def codegym_full_data(env, algo, check_only=False):
    """Use global benchmark sources; sample limits work as for other benchmarks."""
    for key in ('ENV_NAME', 'CODE_ID', 'DATA_DIR', 'MAX_ROWS', 'DYAD_VAL_ACTION_YAML', 'DYAD_VAL_TOOL_CONFIG'):
        if env.get(key):
            raise ValueError(f'CodeGym always uses all environments; remove {key}')
    directory = PROJECT_DIR / 'data/codegym/dataset'
    if env.get('CODEGYM_LEGACY_DATA_DIR'):
        raise ValueError('CODEGYM_LEGACY_DATA_DIR was removed; use the shared step v2 dataset')
    for key, split in [('TRAIN_DATA', 'train'), ('TEST_DATA', 'test')]:
        path = directory / (split + '.parquet')
        if env.get(key) and Path(env[key]).resolve() != path.resolve():
            raise ValueError(f'Full CodeGym uses {key}={path}; subset overrides are not supported')
        env[key] = str(path)
    if env.get('VAL_DATA') and Path(env['VAL_DATA']).resolve() != Path(env['TEST_DATA']).resolve():
        raise ValueError('CodeGym validation uses the full global test split, not a single-env VAL_DATA')
    env['VAL_DATA'] = env['TEST_DATA']
    if algo == 'dyad':
        for key in ('DYAD_ACTION_YAML', 'ACTION_YAML'):
            if env.get(key):
                raise ValueError(f'Full CodeGym builds schemas per task; remove fixed {key}')
        if (env.get('EVAL_MODE') == 'checkpoint' and not env.get('DYAD_CROSS_ENV_SOURCE')
                and env.get('DYAD_CODEGYM_ALL') != '1'):
            raise ValueError('Legacy single-environment CodeGym checkpoint cannot be silently loaded as an all-environment model')
        if env.get('TEMPLATE_STYLE', 'base') != 'base':
            raise ValueError('Full CodeGym requires TEMPLATE_STYLE=base')
        env['DYAD_CODEGYM_ALL'] = '1'
        # A shared sampling replica serves task definitions; the trainer owns encoder gradients.
        if env.get('DYAD_ENCODER_BACKBONE', 'encoder_llm') not in ('encoder_llm', 'encoder_lm') or env.get('DYAD_ENCODER_TRAINING', 'projector_only') not in ('projector_only', 'none', 'projector_and_encoder_lm'):
            raise ValueError('Full CodeGym requires a separate encoder and a supported encoder training mode')
        env['DYAD_ENCODER_REMOTE'] = '1'
        if int(env.get('DYAD_ENCODER_NUM_GPUS', '0')) < 1:
            raise ValueError('Full CodeGym needs DYAD_ENCODER_NUM_GPUS>=1 for its shared task encoder')
    if check_only:
        return
    import pyarrow.parquet as pq
    coverage = {}
    capacity = 0
    for split in ('train', 'test'):
        path = directory / (split + '.parquet')
        if 'agent_name' in pq.ParquetFile(path).schema_arrow.names:
            raise ValueError(f'{path} contains algorithm routing; regenerate the shared dataset')
        count, names = 0, set()
        columns = ['ability', 'extra_info'] + (['prompt'] if algo == 'dyad' else [])
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=columns):
            for row in batch.to_pylist():
                extra = row.get('extra_info') or {}
                if extra.get('split') != split:
                    raise ValueError(f'{path} contains a row outside the {split} split')
                names.add(row['ability'].split('@', 2)[1])
                if algo == 'dyad':
                    from agent_system.policies.dyad.actions.codegym_tasks import task_action_names
                    capacity = max(capacity, len(task_action_names(row['prompt'])))
                count += 1
        if not count:
            raise ValueError(f'Full CodeGym {split} dataset is empty')
        coverage[split] = {'rows': count, 'environments': len(names)}
    if algo == 'dyad':
        saved = int(env.get('DYAD_CODEGYM_ACTION_CAPACITY', '0'))
        if saved and saved < capacity:
            raise ValueError('Checkpoint action capacity is smaller than the full CodeGym dataset')
        env['DYAD_CODEGYM_ACTION_CAPACITY'] = str(saved or capacity)
    env['CODEGYM_COVERAGE'] = json.dumps(coverage)
    print('[prepare] full CodeGym coverage: ' + env['CODEGYM_COVERAGE'])


def configure_dive(env, algo, overrides, check_only=False):
    """Own task protocol and service identity without persisting credentials."""
    from urllib.parse import urlsplit
    from agent_system.environments.env_package.dive.model_api.providers.trapi import resolve_trapi_config
    from agent_system.environments.env_package.dive.judge_config import resolve_judge_config, restore_judge_config
    saved_judge = None
    if env.get('RESUME_CKPT') and not env.get('STEP_WEIGHT_IMPORT'):
        _, saved = checkpoint_model_config(env['RESUME_CKPT'], env.get('EVAL_MODEL_CONFIG'))
        if saved and saved.get('benchmark') == 'dive':
            saved_judge = restore_judge_config(saved)
            if saved_judge is None:
                # Old checkpoints did not persist judge identity. Never apply a new ambient default.
                if not all(env.get(key) for key in ('DIVE_JUDGE_PROVIDER', 'DIVE_JUDGE_MODEL', 'DIVE_JUDGE_BASE_URL')):
                    raise ValueError('Old DIVE checkpoint lacks judge identity; explicitly supply '
                                     'DIVE_JUDGE_PROVIDER, DIVE_JUDGE_MODEL and DIVE_JUDGE_BASE_URL')
    judge = resolve_judge_config(saved_judge, env)
    env.update(DIVE_JUDGE_PROVIDER=judge['provider'], DIVE_JUDGE_MODEL=judge['model'],
               DIVE_JUDGE_BASE_URL=judge['base_url'], DIVE_JUDGE_CONFIG=json.dumps(judge))
    if 'api_key_env' in judge:
        env['DIVE_JUDGE_API_KEY_ENV'] = judge['api_key_env']
    browse_provider = env.get('BROWSE_LLM_PROVIDER') or 'openai_compatible'
    if browse_provider not in ('openai_compatible', 'trapi'):
        raise ValueError('Unsupported DIVE browse provider')
    if browse_provider == 'trapi':
        resolved = resolve_trapi_config(model=env.get('BROWSE_LLM_MODEL'),
                                        base_url=env.get('BROWSE_LLM_BASE_URL'), env=env)
        env.update(BROWSE_LLM_MODEL=resolved['model'], BROWSE_LLM_BASE_URL=resolved['base_url'])
    for key in ('DIVE_JUDGE_BASE_URL', 'SANDBOX_FUSION_URL'):
        if not env.get(key):
            continue
        parsed = urlsplit(env[key])
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(f'{key} must be HTTP(S) without embedded credentials')
    for key in ('DYAD_ACTION_YAML', 'ACTION_YAML', 'TOOL_CONFIG_PATH', 'DIVE_SESSION_CONFIG'):
        if env.get(key):
            raise ValueError(f'DIVE owns per-task schemas and transport; remove {key}')
    default(env, 'DIVE_ENV_POOL_SIZE', '4')
    default(env, 'DIVE_ENV_CPUS_PER_WORKER', '0.25')
    import math
    cpus = float(env['DIVE_ENV_CPUS_PER_WORKER'])
    if int(env['DIVE_ENV_POOL_SIZE']) < 1 or not math.isfinite(cpus) or cpus <= 0:
        raise ValueError('DIVE pool size and CPUs per worker must be positive')
    if int(env['MAX_ASSISTANT_TURNS']) < 1:
        raise ValueError('DIVE MAX_ASSISTANT_TURNS must be positive')
    default(env, 'DIVE_PYTHON_BIN', PROJECT_DIR / '.venvs/dive/bin/python')
    env['PYTHONPATH'] = str(PROJECT_DIR) + os.pathsep + env.get('PYTHONPATH', '')
    env['RUN_ENV'] = 'dive'
    if algo == 'dyad':
        if env.get('DYAD_ENCODER_BACKBONE', 'encoder_llm') not in {'encoder_llm', 'encoder_lm'} or env.get('DYAD_ENCODER_TRAINING', 'projector_only') not in {'projector_only', 'none', 'projector_and_encoder_lm'}:
            raise ValueError('DIVE dynamic actions require a separate encoder and a supported encoder training mode')
        if int(env.get('DYAD_ENCODER_NUM_GPUS', '0')) < 1:
            raise ValueError('DIVE needs DYAD_ENCODER_NUM_GPUS>=1 in addition to policy GPUs')
        if env.get('DYAD_ACTION_CAPACITY', '84') != '84' or env.get('DYAD_DYNAMIC_ACTIONS', '1') != '1':
            raise ValueError('DIVE requires DYAD_DYNAMIC_ACTIONS=1 and DYAD_ACTION_CAPACITY=84')
        env.update(DYAD_DYNAMIC_ACTIONS='1', DYAD_ACTION_CAPACITY='84', DYAD_ENCODER_REMOTE='1',
                   DYAD_ACTION_CONTEXT_DIR=str(Path(env['RUN_DIR']) / 'encoder_contexts'))
    protected = ('data.train_files', 'data.val_files', 'data.custom_cls', 'data.dive',
                 'data.filter_overlong_prompts', 'data.truncation',
                 'actor_rollout_ref.rollout.multi_turn.tool_config_path',
                 'actor_rollout_ref.rollout.multi_turn.format',
                 'actor_rollout_ref.rollout.multi_turn.max_assistant_turns')
    for arg in overrides:
        key = arg.lstrip('+~').partition('=')[0]
        if any(key == p or key.startswith(p + '.') or p.startswith(key + '.') for p in protected):
            raise ValueError('DIVE owns dataset/protocol/turn-budget invariants; remove override ' + key)
    session = {'max_steps': int(env['MAX_ASSISTANT_TURNS']),
               'judge_config': judge,
               'sandbox_url': env.get('SANDBOX_FUSION_URL')}
    env['DIVE_SESSION_CONFIG'] = json.dumps(session)
    env['DIVE_TOOL_CONFIG_PATH'] = str(Path(env['RUN_DIR']) / 'dive_tool.yaml')
    if not check_only:
        import yaml
        path = Path(env['DIVE_TOOL_CONFIG_PATH'])
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {'tools': [{'class_name': 'agent_system.environments.backends.dive.tool.DiveLocalEnvTool',
                            'config': {'type': 'native', 'pool_size': int(env['DIVE_ENV_POOL_SIZE']),
                                       'num_cpus_per_worker': float(env['DIVE_ENV_CPUS_PER_WORKER']),
                                       'python_executable': env['DIVE_PYTHON_BIN'],
                                       'forward_env_vars': [judge['api_key_env']] if 'api_key_env' in judge else [],
                                       'session_config': session}}]}
        content = yaml.safe_dump(config)
        if path.exists() and path.read_text() != content:
            raise ValueError('Existing DIVE tool config differs; choose a new RUN_ID')
        path.write_text(content)


def benchmark_data(env, benchmark, algo, check_only=False, overrides=None):
    if benchmark == 'dive':
        import importlib.util
        path = PROJECT_DIR / 'experiments/shared/dataset/dive.py'
        spec = importlib.util.spec_from_file_location('dyad_dive_dataset', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = Path(env.get('DIVE_SOURCE_DIR', PROJECT_DIR / 'data/dive')).expanduser().resolve()
        directory = Path(env.get('DIVE_DATA_DIR', PROJECT_DIR / module.DEFAULT_DATA_DIR)).expanduser().resolve()
        flag = env.get('DIVE_DROP_INVALID', '1')
        if flag not in {'0', '1'}:
            raise ValueError('DIVE_DROP_INVALID must be 0 or 1')
        drop_invalid = flag == '1'
        # An existing audited conversion is already an explicit exclusion decision.
        manifest_path = directory / 'manifest.json'
        if 'DIVE_DROP_INVALID' not in env and manifest_path.is_file():
            saved = json.loads(manifest_path.read_text())
            if saved.get('format') == 'dive-runtime-v1':
                drop_invalid = saved.get('drop_invalid') is True
        for key, split in [('TRAIN_DATA', 'train'), ('TEST_DATA', 'test')]:
            path = directory / (split + '.parquet')
            if env.get(key) and Path(env[key]).resolve() != path:
                raise ValueError('DIVE uses audited shared splits; remove ' + key)
            env[key] = str(path)
        if env.get('VAL_DATA') and Path(env['VAL_DATA']).resolve() != Path(env['TEST_DATA']):
            raise ValueError('DIVE evaluation uses the audited test split; remove VAL_DATA')
        manifest = module.prepare(source, directory, drop_invalid=drop_invalid, check=check_only)
        module.prepare_data(env, PROJECT_DIR, overrides or [], check_only=check_only, manifest=manifest)
        return
    if benchmark == 'webshop':
        import importlib.util
        path = PROJECT_DIR / 'experiments/shared/dataset/webshop.py'
        spec = importlib.util.spec_from_file_location('dyad_webshop_dataset', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with environment(env):
            module.prepare_data(env, PROJECT_DIR, overrides if overrides is not None else [],
                                check_only=check_only)
        return
    if benchmark == 'alfworld' and env.get('RUN_IS_EVAL') == '1':
        import importlib.util
        path = PROJECT_DIR / 'experiments/shared/dataset/alfworld.py'
        spec = importlib.util.spec_from_file_location('dyad_alfworld_dataset', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.prepare_evaluation(env, PROJECT_DIR, overrides if overrides is not None else [],
                                  check_only=check_only)
    data_root = PROJECT_DIR / 'data'
    for key, split in [('TRAIN_DATA', 'train'), ('TEST_DATA', 'test')]:
        default(env, key, data_root / benchmark / 'dataset' / (split + '.parquet'))
    if benchmark == 'codegym':
        codegym_full_data(env, algo, check_only)


def validate_shared_dataset(path, *, agent_loop=None):
    """Check packaged task inputs and keep them out of legacy prompt consumers."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = pq.ParquetFile(path)
    schema = source.schema_arrow
    if 'agent_name' in schema.names:
        raise ValueError(f'{path} contains agent_name; regenerate an algorithm-neutral Agentic RL dataset')
    if 'extra_info' not in schema.names or not pa.types.is_struct(schema.field('extra_info').type):
        return
    extra = schema.field('extra_info').type
    if 'dataset_format' not in [field.name for field in extra]:
        return
    formats = source.read(columns=['extra_info.dataset_format']).column('extra_info').flatten()[0]
    known = set(formats.to_pylist())
    if known != {'source_tasks_v1'}:
        raise ValueError(f'{path} has mixed or unsupported dataset_format values')
    if agent_loop != 'environment_step_agent':
        raise ValueError(
            f'{path} uses source_tasks_v1 task inputs, not a legacy full-trajectory prompt. '
            'Use environment_step_agent with the shared step v2 protocol.'
        )
    from experiments.shared.dataset.utils.source_snapshot import verify_transport

    verify_transport(Path(path))


def write_command(command):
    # The shared shell emitter ends here, so direct internal builders cannot bypass
    # the same guard used by public --check and the trainer's runtime guard.
    require_no_thinking(os.environ, command[3:])
    if qwen35_run(os.environ):
        command = [arg for arg in command if arg.lstrip('+').partition('=')[0] != THINKING_FIELD]
        command.append('++' + THINKING_FIELD + '=False')
    target = Path(os.environ['DYAD_COMMAND_FILE'])
    target.write_text(json.dumps({'command': command, 'env': dict(os.environ)}))
    return 0


def launch(args):
    env = dict(os.environ)
    if env.get('LOCAL_DEBUG_SMOKE', '0') not in {'', '0'}:
        raise ConfigError('LOCAL_DEBUG_SMOKE is no longer supported; use --debug with the configured training budget')
    benchmark, algo = args.benchmark, args.algo
    # Keep checkpoint auto-selection unresolved until its saved identity is loaded.
    # Likewise, bare legacy dyad must not acquire a GRPO default before restore.
    if algo not in {'auto', 'baseline', 'dyad'}:
        algo = apply_algorithm(env, algo).algo
    mode = args.mode
    env['RUN_IS_EVAL'] = '1' if mode == 'evaluation' else '0'
    if benchmark == 'tbench':
        raise ValueError('Unsupported benchmark: tbench has been removed.')
    if mode == 'training' and benchmark not in {'dive', 'codegym', 'alfworld', 'webshop'}:
        raise ValueError(
            f'{benchmark} is evaluation-only; use evaluate.sh. '
            'Training datasets: dive, codegym, alfworld, webshop')
    env['RUN_IS_EVAL'] = '1' if mode == 'evaluation' else '0'
    env['PYTHON_BIN'] = sys.executable
    env['PYTHONPATH'] = str(PROJECT_DIR) + os.pathsep + env.get('PYTHONPATH', '')
    if mode == 'evaluation':
        algo = select_evaluation(env, benchmark, algo)
    if benchmark not in {'gsm8k', 'alfworld', 'codegym', 'webshop', 'dive'} or algo not in {'dyad', 'grpo_react', 'gigpo'}:
        raise ValueError('Unsupported benchmark/algorithm: ' + benchmark + '/' + algo)
    env['RUN_ALGO_BASE'] = algo
    overrides = list(args.overrides)
    if overrides and overrides[0] == '--':
        overrides.pop(0)
    for arg in overrides:
        key = arg.lstrip('+~').partition('=')[0]
        if (key == 'algorithm.step_rollout.local_debug_smoke'
                or key.startswith('algorithm.step_rollout.local_debug_smoke.')):
            raise ConfigError('algorithm.step_rollout.local_debug_smoke is no longer supported')
    if mode == 'training':
        require_initial_training_validation(env, overrides)
    if mode == 'training' and env.get('DYAD_ENCODER_PROJECTOR_INIT'):
        if algo != 'dyad':
            raise ValueError('Alignment projector initialization requires algo=dyad')
        env['DYAD_ENCODER_PROJECTOR_INIT'] = resolve_alignment_projector(env['DYAD_ENCODER_PROJECTOR_INIT'], env)
    if algo == 'dyad':
        action_config(env, benchmark)
    configure_weight_import(env, benchmark, algo, overrides)
    require_no_thinking(env, overrides)
    validate_thinking_resume(env)
    configure_step_profile(env, benchmark, algo, overrides)
    debug_config(env, benchmark, algo)
    scale_config(env, benchmark, algo)
    configure_step_action_schema(env, benchmark, algo)
    validate_gigpo_selection(env, algo, overrides)
    debug_config(env, benchmark, algo)
    if not args.check:
        resolve_model(env)
    require_no_thinking(env, overrides)
    validate_thinking_resume(env)
    output_layout(env, benchmark, algo)
    configure_rollout_output(env, mode, overrides)
    if env.get('STEP_WEIGHT_IMPORT'):
        source = Path(json.loads(env['STEP_WEIGHT_IMPORT'])['checkpoint'])
        for key in ('RUN_DIR', 'CKPT_DIR'):
            target = Path(env[key]).expanduser().resolve()
            if target == source or target in source.parents or source in target.parents or (target / 'model_config.json').exists():
                raise ConfigError('Weight-only import must start a new run with separate output/checkpoint directories')
    if benchmark == 'dive':
        configure_dive(env, algo, overrides, check_only=args.check)
    padding = env.get('DYAD_EXACT_BATCH_PADDING', '0')
    if padding not in ('0', '1'):
        raise ValueError('DYAD_EXACT_BATCH_PADDING must be 0 or 1')
    if padding == '1' and (benchmark, algo) != ('codegym', 'dyad'):
        raise ValueError('DYAD_EXACT_BATCH_PADDING requires CodeGym Dyad')
    if env.get('DYAD_ENCODER_PROJECTOR_INIT'):
        print('[prepare] Alignment projector: ' + env['DYAD_ENCODER_PROJECTOR_INIT'])
    if mode == 'training':
        validate_training_budget(env, benchmark, overrides)
    if mode == 'evaluation' and env.get('EVAL_MODEL_OVERRIDES'):
        saved = json.loads(env['EVAL_MODEL_OVERRIDES'])
        for value in saved:
            key = value.lstrip('+').split('=', 1)[0]
            if any(arg.lstrip('+').split('=', 1)[0] == key and arg.lstrip('+') != value.lstrip('+') for arg in overrides):
                raise ValueError('Model override conflicts with checkpoint: ' + key)
        overrides = saved + overrides
    if benchmark == 'codegym':
        protected = ('data.train_files', 'data.val_files',
                     'data.filter_overlong_prompts', 'data.truncation')
        for arg in overrides:
            if arg.lstrip('+~').split('=', 1)[0] in protected:
                raise ValueError('Full CodeGym owns data selection and sample coverage; remove ' + arg)
        env['DYAD_CODEGYM_CONTEXT_DIR'] = str(Path(env['RUN_DIR']) / 'encoder_contexts')
    benchmark_data(env, benchmark, algo, args.check, overrides)
    output_keys = {'hydra.run.dir', 'trainer.default_local_dir', 'trainer.validation_data_dir'}
    for arg in overrides:
        key = arg.lstrip('+~').split('=', 1)[0]
        if any(key == field or field.startswith(key + '.') for field in output_keys):
            raise ValueError('Set RUN_DIR, CKPT_DIR or VALIDATION_DATA_DIR instead; outputs must stay in artifact roots')
    if mode == 'evaluation':
        forbidden = ('data.val_files', 'trainer.val_only', 'trainer.val_before_train', 'trainer.resume_',
                     'trainer.del_local_ckpt_after_load', 'actor_rollout_ref.actor.checkpoint.load_contents')
        for arg in overrides:
            key = arg.lstrip('+~').split('=', 1)[0]
            if key == 'trainer' or key.startswith(forbidden):
                raise ValueError('Evaluation owns this parameter; use its model/test-data options: ' + arg)
        if mode == 'evaluation' and benchmark == 'codegym':
            env.pop('CODEGYM_BENCHMARK_TEST_DATA', None)
    if mode == 'evaluation':
        from agent_system.evaluation.config import from_environment, command
        explicit_overrides = list(args.overrides)
        if explicit_overrides and explicit_overrides[0] == '--':
            explicit_overrides.pop(0)
        evaluation = from_environment(env, benchmark, explicit_overrides)
        if args.check:
            print(json.dumps({'backend': 'inference_api', 'configuration': evaluation}, ensure_ascii=False))
            return 0
        if args.dry_run:
            print(shlex.join(command(evaluation)))
            return 0
        if evaluation['projector_init'] and not evaluation['model_config']:
            directory = Path(env['RUN_DIR'])
            directory.mkdir(parents=True, exist_ok=True)
            model_config = directory / 'serving_model_config.json'
            saved = {'version': 1, 'benchmark': benchmark, 'algo': 'dyad',
                     'action_interface': 'dyad', 'adv_estimator': env['RUN_ADV_ESTIMATOR'],
                     'model': {key: env[key] for key in MODEL_KEYS if env.get(key)}}
            model_config.write_text(json.dumps(saved, indent=2) + '\n')
            evaluation['model_config'] = str(model_config)
        from run import run_process
        return run_process(command(evaluation), env)
    if args.check:
        agent_loop = ('environment_step_agent' if env.get('STEP_ROLLOUT_ENABLED') == 'True'
                      or (benchmark == 'gsm8k' and mode == 'evaluation') else
                      'tool_agent')
        for key in ('TRAIN_DATA', 'TEST_DATA', 'VAL_DATA'):
            if env.get(key) and Path(env[key]).is_file():
                validate_shared_dataset(env[key], agent_loop=agent_loop)
        print(f'[prepare] configuration passed: mode={mode} benchmark={benchmark} algo={algo} '
              f'action_interface={env["RUN_ACTION_INTERFACE"]} adv_estimator={env["RUN_ADV_ESTIMATOR"]}')
        print(f'[prepare] outputs={env["RUN_DIR"]}\n[prepare] checkpoints={env["CKPT_DIR"]}')
        if thinking_metadata(env):
            print('[prepare] thinking=' + json.dumps(thinking_metadata(env), sort_keys=True))
        return 0
    cross_source = json.loads(env['DYAD_CROSS_ENV_SOURCE']) if env.get('DYAD_CROSS_ENV_SOURCE') else None
    if cross_source:
        restore = Path(env['RUN_DIR']) / 'native_restore'
        env.update(DYAD_NATIVE_RESTORE_DIR=str(restore), DYAD_NATIVE_MODEL_CONFIG=env['EVAL_MODEL_CONFIG'],
                   MODEL_PATH=str(restore / 'policy'))
    if mode == 'training':
        # Training validation uses the same persistent sample location as evaluation.
        overrides.append('trainer.validation_data_dir=' + env['VALIDATION_DATA_DIR'])
    executor = SELF.parent / (mode + '.sh')
    with tempfile.TemporaryDirectory(prefix='dyad-command-') as tmp:
        target = Path(tmp) / 'command.json'
        env['DYAD_COMMAND_FILE'] = str(target)
        subprocess.run(['bash', str(executor), '--build', benchmark, algo, *overrides], env=env, cwd=PROJECT_DIR, check=True)
        plan = json.loads(target.read_text())
        plan.update(mode=mode, benchmark=benchmark, algo=algo)
        plan['configuration'] = {'alfworld_eval_data': json.loads(env.get('ALFWORLD_EVAL_DATA', '{}')),
                                 'codegym_coverage': json.loads(env.get('CODEGYM_COVERAGE', '{}')),
                                 'webshop_data': json.loads(env.get('WEBSHOP_DATA', '{}')),
                                 'dive_data': json.loads(env.get('DIVE_DATA', '{}')),
                                 'benchmark': benchmark, 'algorithm': algo, 'model': env['MODEL_NAME'],
                                 'action_interface': env['RUN_ACTION_INTERFACE'],
                                 'adv_estimator': env['RUN_ADV_ESTIMATOR'],
                                 'hardware': env['SCALE_PROFILE'], 'debug': env.get('RUN_IS_DEBUG') == '1',
                                 'experiment': env.get('TRAINING_EXPERIMENT', 'train'),
                                 'parameters': {k: plan['env'][k] for k in sorted(SCALE_KEYS) if k in plan['env']},
                                 'hydra': dict(arg.lstrip('+').split('=', 1) for arg in plan['command'][3:] if '=' in arg)}
        plan['configuration'].update(thinking_protocol_metadata(plan['env'], benchmark, algo, plan['command']))
        if env.get('DYAD_ENCODER_PROJECTOR_INIT'):
            plan['configuration']['alignment_projector'] = env['DYAD_ENCODER_PROJECTOR_INIT']
        if mode == 'evaluation':
            plan['configuration']['source_benchmark'] = env.get('EVAL_SOURCE_BENCHMARK')
            plan['configuration']['target_benchmark'] = benchmark
            plan['configuration']['evaluation_checkpoint'] = env.get('EVAL_CHECKPOINT')
        if cross_source:
            plan['configuration']['native_source'] = cross_source
            plan['configuration']['target_actions_rebuilt'] = True
        default(plan['env'], 'MODEL_STAGE_TO_LOCAL', '1' if benchmark == 'gsm8k' and algo != 'dyad' and int(env['N_GPUS_PER_NODE']) >= 4 else '0')
        values = dict(arg.lstrip('+').split('=', 1) for arg in plan['command'][3:] if '=' in arg)
        for key in ('data.train_files', 'data.val_files'):
            path = values.get(key)
            if path and Path(path).is_file():
                validate_shared_dataset(path, agent_loop=values.get('actor_rollout_ref.rollout.agent.default_agent_loop'))
        if args.dry_run:
            if 'evaluation_protocol' in plan['configuration']:
                print('[prepare] evaluation_protocol=' + json.dumps(plan['configuration']['evaluation_protocol'],
                                                                   sort_keys=True))
            print(shlex.join(plan['command']))
            return 0
        # Validate fully assembled parameters before Ray or model staging starts.
        if mode == 'evaluation' or benchmark == 'dive':
            from evaluation_resources import check_evaluation_resources
            plan['configuration']['resources'] = check_evaluation_resources(plan['command'][2], values, plan['env'])
        for key in ('data.train_files', 'data.val_files'):
            path = values.get(key)
            if path and not Path(path).is_file():
                raise ValueError(f'{key} does not exist: {path}')
            if path:
                validate_shared_dataset(path, agent_loop=values.get('actor_rollout_ref.rollout.agent.default_agent_loop'))
        if 'actor_rollout_ref.model.path' in values and values['actor_rollout_ref.model.path'] != env['MODEL_PATH']:
            raise ValueError('Use MODEL_PATH to select the model so saved model configuration stays consistent')
        with environment(plan['env']):
            if cross_source:
                from agent_system.policies.dyad.inference.checkpoint import prepare_native
                prepare_native(cross_source, restore, target_benchmark=benchmark)
            if algo == 'dyad' and env.get('DYAD_ENCODER_PROJECTOR_INIT'):
                rc = check_projector(argparse.Namespace(projector_init=env['DYAD_ENCODER_PROJECTOR_INIT'], model_name=env['MODEL_NAME'], projector=env.get('DYAD_ENCODER_PROJECTOR', 'attention'), scale=env.get('DYAD_ENCODER_SCALE', 'unit')))
                if rc:
                    return rc
            if benchmark == 'dive':
                # The audited three-domain subset has no financial or general tools.
                rc = subprocess.run([env['DIVE_PYTHON_BIN'], '-m', 'ops.env_deps.dive.verify',
                                     '--driver-python', env['PYTHON_BIN']], env=plan['env']).returncode
            else:
                profile = 'calc' if benchmark == 'gsm8k' else benchmark
                rc = run_preflight(argparse.Namespace(profile=profile, variant='dyad' if algo == 'dyad' else 'react', depth='auto'))
            if rc:
                return rc
        target.write_text(json.dumps(plan))
        from run import run_process
        return run_process([sys.executable, str(SELF.with_name('run.py')), str(target)], env)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'swebench-evaluation':
        from swebench_prepare import main as swebench_evaluation_main
        return swebench_evaluation_main(argv[1:])
    if argv and argv[0] == 'tau-evaluation':
        from tau_prepare import main as tau_evaluation_main
        return tau_evaluation_main(argv[1:])
    if argv and argv[0] == 'write-command':
        return write_command(argv[1:])
    if not argv or argv[0] != 'launch':
        return checks_main(argv)
    parser = argparse.ArgumentParser(description='Shared preparation for training and benchmark evaluation', allow_abbrev=False)
    parser.add_argument('command', choices=['launch'])
    parser.add_argument('mode', choices=['training', 'evaluation'])
    parser.add_argument('benchmark')
    parser.add_argument('algo')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    # Separate Hydra overrides explicitly; argparse must still parse --check after positionals.
    split = argv.index('--') if '--' in argv else len(argv)
    args = parser.parse_args(argv[:split])
    args.overrides = argv[split + 1:]
    try:
        return launch(args)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'[prepare] {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    from run import signal_scope
    with signal_scope():
        raise SystemExit(main())
