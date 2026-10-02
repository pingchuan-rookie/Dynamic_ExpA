#!/usr/bin/env python3
"""Own a Agentic RL run: Ray, optional model staging, execution, cleanup and debug analysis."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

SELF = Path(__file__).resolve()
PROJECT_DIR = SELF.parents[4]


def effective_cpus():
    physical = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    root = Path(os.environ.get('RAY_CGROUP_ROOT', '/sys/fs/cgroup'))
    try:
        if (root / 'cpu.max').is_file():
            quota, period = (root / 'cpu.max').read_text().split()
            if quota != 'max':
                return max(1, min(physical, math.ceil(int(quota) / int(period))))
        else:
            quota = int((root / 'cpu/cpu.cfs_quota_us').read_text())
            period = int((root / 'cpu/cpu.cfs_period_us').read_text())
            if quota > 0:
                return max(1, min(physical, math.ceil(quota / period)))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return physical


def stop_process(process):
    """Stop only this child and its descendants; never issue global ray stop/pkill."""
    import psutil
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for child in children:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.terminate()
    # These are grandchildren, not waitpid children. Avoid psutil's pidfd wait
    # path, which can return EINVAL on container kernels even for exited PIDs.
    alive = children
    deadline = time.monotonic() + 5
    while alive and time.monotonic() < deadline:
        pending = []
        for child in alive:
            with contextlib.suppress(psutil.NoSuchProcess):
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    pending.append(child)
        alive = pending
        if alive:
            time.sleep(0.1)
    for child in alive:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()


@contextlib.contextmanager
def signal_scope():
    previous = {}
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, interrupted)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def run_process(command, env, timeout=None):
    process = subprocess.Popen(command, env=env, cwd=PROJECT_DIR, start_new_session=True,
                               text=True, errors='replace')
    try:
        return process.wait(timeout=timeout)
    finally:
        stop_process(process)


class RaySession:
    """A bounded-start, job-owned local Ray head shared by probes and the trainer."""
    def __init__(self, env):
        self.env = dict(env)
        self.process = None
        self.directory = None
        self.log = None

    def __enter__(self):
        self.directory = tempfile.TemporaryDirectory(prefix='dyad-ray-')
        folder = Path(self.directory.name)
        ready = folder / 'ready.json'
        self.log = (folder / 'head.log').open('w+')
        env = dict(self.env)
        env.pop('RAY_ADDRESS', None)
        try:
            self.process = subprocess.Popen([env.get('PYTHON_BIN', sys.executable), str(SELF), '--ray-host', str(ready)],
                                            env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + int(env.get('RAY_START_TIMEOUT_SECONDS', '90'))
            while not ready.is_file():
                if self.process.poll() is not None:
                    raise RuntimeError('Ray head failed to start')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Ray head startup timed out')
                time.sleep(0.1)
            address = json.loads(ready.read_text())['address']
            print('[run] Ray ready: ' + address, flush=True)
            return address
        except BaseException:
            self.log.flush()
            self.log.seek(0)
            print(self.log.read()[-12000:], file=sys.stderr)
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        if self.process:
            stop_process(self.process)
        if self.log:
            self.log.flush()
            if self.env.get('RUN_DIR'):
                directory = Path(self.env['RUN_DIR'])
                directory.mkdir(parents=True, exist_ok=True)
                self.log.seek(0)
                with (directory / 'ray_head.log').open('a') as saved:
                    shutil.copyfileobj(self.log, saved)
            self.log.close()
        if self.directory:
            self.directory.cleanup()


def ray_host(ready):
    # ray.init owns this cluster. ray.shutdown terminates only the processes it started.
    import ray
    with signal_scope():
        try:
            info = ray.init(address='local', include_dashboard=False,
                            num_cpus=int(os.environ.get('RAY_NUM_CPUS') or effective_cpus()),
                            _temp_dir=str(ready.parent / 'ray'),
                            _system_config={'worker_register_timeout_seconds': int(os.environ.get('RAY_WORKER_REGISTER_TIMEOUT_SECONDS', '120'))})
            temp = ready.with_suffix('.tmp')
            temp.write_text(json.dumps({'address': info.address_info['gcs_address']}))
            temp.replace(ready)
            while True:
                time.sleep(1)
        finally:
            ray.shutdown()


def debug_analysis(env, rc):
    if env.get('RUN_IS_DEBUG') != '1' or env.get('POST_ANALYSIS', '1') == '0':
        return
    directory = Path(env['RUN_DIR'])
    if not directory.is_dir():
        return
    analysis = directory / 'analysis'
    analysis.mkdir(exist_ok=True)
    algo = env['RUN_ALGO_BASE']
    verifier = {'dyad': 'verify_dyad.py', 'grpo_react': 'verify_grpo_react.py',
                'gigpo': 'verify_grpo_react.py'}[algo]
    scripts = [(verifier, []), ('decode_trajectories.py', ['--out'])]
    source = PROJECT_DIR / 'experiments/shared/analysis/run'
    statuses = [f'train_rc={rc}']
    for script, args in scripts:
        try:
            result = run_process([env['PYTHON_BIN'], str(source / script), str(directory), *args], env)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f'[analysis] {script}: {exc}', file=sys.stderr)
            result = 127
        statuses.append(f'{script} rc={result}')
    (analysis / 'post_analysis.status').write_text('\n'.join(statuses) + '\n')


def write_model_config(plan):
    from prepare import (MODEL_KEYS, THINKING_FIELD, checkpoint_algorithm, qwen35_run,
                         require_no_thinking, resolve_algorithm, thinking_protocol_metadata)
    env = plan['env']
    command = plan.get('command', [])
    require_no_thinking(env, command[3:])
    if qwen35_run(env):
        from hydra.core.override_parser.overrides_parser import OverridesParser
        parser = OverridesParser.create()
        if not any(arg.lstrip('+').partition('=')[0] == THINKING_FIELD and
                   parser.parse_override(arg).value() is False for arg in command[3:]):
            raise ValueError('Qwen3.5 command must explicitly disable thinking before saving effective metadata')
    selection = resolve_algorithm(plan['algo'], env)
    for key, expected in (('RUN_ACTION_INTERFACE', selection.action_interface),
                          ('RUN_ADV_ESTIMATOR', selection.adv_estimator)):
        if env.get(key, expected) != expected:
            raise ValueError(f'{key} conflicts with algorithm identity')
    overrides = {}
    for arg in plan.get('command', [])[3:]:
        if arg.lstrip('+').startswith('actor_rollout_ref.model.override_config.'):
            overrides[arg.lstrip('+').split('=', 1)[0]] = '++' + arg.lstrip('+')
    data = {'version': 1, 'benchmark': plan['benchmark'], 'algo': selection.algo,
            'action_interface': selection.action_interface, 'adv_estimator': selection.adv_estimator,
            'model': {key: env[key] for key in sorted(MODEL_KEYS) if env.get(key)},
            'model_overrides': list(overrides.values())}
    if plan['benchmark'] == 'dive':
        from agent_system.environments.env_package.dive.judge_config import restore_judge_config
        # Older saved command plans carry an env mapping but no explicit provider.
        # Restore from that mapping, never from ambient process defaults.
        judge = restore_judge_config({'env': env})
        if judge is None:
            raise ValueError('Saved DIVE command plan lacks judge identity; rebuild via the public entrypoint')
        data['dive_judge'] = judge
        env.update(DIVE_JUDGE_PROVIDER=judge['provider'], DIVE_JUDGE_CONFIG=json.dumps(judge))
    if selection.action_interface == 'dyad':
        data['model']['DYAD_ADV_ESTIMATOR'] = selection.adv_estimator
    data.update(thinking_protocol_metadata(env, plan['benchmark'], selection.algo, plan.get('command', [])))
    # Save alongside every run and at the checkpoint run root so step_N resolves without a recipe.
    targets = [Path(env['RUN_DIR'])]
    if plan['mode'] == 'training':
        targets.append(Path(env['CKPT_DIR']))
    for directory in targets:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'model_config.json'
        if path.exists():
            from agent_system.policies.dyad.checkpoint_compat import read_saved_model_config
            saved = read_saved_model_config(path)
            saved_selection = checkpoint_algorithm(saved)
            comparable = {**saved, 'action_interface': saved_selection.action_interface,
                          'adv_estimator': saved_selection.adv_estimator}
            if saved_selection.action_interface == 'dyad':
                comparable['model'] = {**saved['model'], 'DYAD_ADV_ESTIMATOR': saved_selection.adv_estimator}
            if comparable != data:
                raise ValueError('Refusing to overwrite different model configuration: ' + str(path))
            # An equivalent legacy file remains byte-for-byte intact on resume.
            continue
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def execute(plan):
    from prepare import validate_output_paths
    from agent_system.utils.artifact_ownership import restore_run_ownership
    env = dict(plan['env'])
    validate_output_paths(env)
    paths = [env['RUN_DIR']]
    if plan['mode'] == 'training':
        paths.append(env['CKPT_DIR'])
    try:
        return _execute({**plan, 'env': env})
    finally:
        try:
            restore_run_ownership(paths, env)
        except Exception as exc:
            print(f'[ownership] cleanup failed: {exc}', file=sys.stderr)


def _execute(plan):
    if plan['mode'] == 'evaluation':
        # Evaluation command owns its inference service and environment lifecycle.
        command = list(plan['command'])
        config = json.loads(command[-1])
        config['preparation_evidence'] = plan.get('configuration', {})
        command[-1] = json.dumps(config)
        plan = {**plan, 'command': command}
        directory = Path(plan['env']['RUN_DIR'])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'command.json').write_text(json.dumps(plan['command'], indent=2) + '\n')
        return run_process(plan['command'], dict(plan['env']))
    env, command = dict(plan['env']), list(plan['command'])
    env.pop('DYAD_LOG_FILE', None)
    env['PYTHONUNBUFFERED'] = '1'
    from prepare import validate_output_paths
    validate_output_paths(env)
    plan = {**plan, 'env': env}
    directory = Path(env['RUN_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    for key in ('WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR'):
        if env.get(key):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
    write_model_config(plan)
    if 'configuration' in plan:
        (directory / 'resolved_config.json').write_text(json.dumps(plan['configuration'], ensure_ascii=False, indent=2) + '\n')
    rc = 1
    with tempfile.TemporaryDirectory(prefix='dyad-model-') as temporary:
        try:
            if env.get('MODEL_STAGE_TO_LOCAL', '0') == '1':
                destination = Path(temporary) / 'model'
                shutil.copytree(env['MODEL_PATH'], destination, symlinks=False)
                env['DYAD_MODEL_SOURCE_PATH'] = env['MODEL_PATH']
                env['MODEL_PATH'] = str(destination)
                command = ['actor_rollout_ref.model.path=' + str(destination) if arg.startswith('actor_rollout_ref.model.path=') else arg for arg in command]
            # Record no inherited environment: credentials must not enter the run manifest.
            (directory / 'command.json').write_text(json.dumps(command, ensure_ascii=False, indent=2) + '\n')
            with RaySession(env) as address:
                env['RAY_ADDRESS'] = address
                shell = SELF.with_name(plan['mode'] + '.sh')
                rc = run_process(['bash', str(shell), '--execute', *command], env)
        except SystemExit as exc:
            rc = int(exc.code)
            raise
        finally:
            (directory / 'run.status').write_text(f'exit_code={rc}\n')
            # This must not mask the execution status, including when analysis itself fails.
            try:
                debug_analysis(env, rc)
            except Exception as exc:
                print(f'[analysis] Failed: {exc}', file=sys.stderr)
    return rc if rc >= 0 else 128 - rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', nargs='?')
    parser.add_argument('--ray-host', type=Path)
    args = parser.parse_args()
    if args.ray_host:
        ray_host(args.ray_host)
        return 0
    if not args.plan:
        parser.error('plan required (normally supplied by prepare.py)')
    with signal_scope():
        return execute(json.loads(Path(args.plan).read_text()))


if __name__ == '__main__':
    raise SystemExit(main())
