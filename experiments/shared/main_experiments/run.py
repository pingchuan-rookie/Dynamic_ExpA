"""List the two main experiment matrices and dispatch one explicitly selected job."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

PROJECT = Path(__file__).resolve().parents[3]
MATRIX = Path(__file__).with_name('matrix.yaml')


@dataclass(frozen=True)
class Job:
    id: str
    suite: str
    kind: str
    model: str
    source: str
    method: str
    seed: int | None
    target: str
    training_id: str | None = None


def load_matrix():
    return yaml.safe_load(MATRIX.read_text())


def jobs(seeds):
    matrix = load_matrix()
    result = []
    for suite, spec in matrix['suites'].items():
        for model in spec['models']:
            for source in spec['sources']:
                for method in matrix['methods']:
                    for seed in seeds:
                        key = f'{suite}/{model}/{source}/{method}/seed-{seed}'
                        result.append(Job(key + '/train', suite, 'train', model, source, method, seed, source))
                        targets = spec['targets'] if suite == 'transfer' else [source]
                        for target in targets:
                            result.append(Job(key + '/eval/' + target, suite, 'environment', model,
                                              source, method, seed, target, key + '/train'))
                        if suite == 'retention':
                            for target in matrix['capabilities']:
                                result.append(Job(key + '/capability/' + target, suite, 'capability', model,
                                                  source, method, seed, target, key + '/train'))
    # Initial evaluations have no training-source or training-seed dependency.
    for suite, spec in matrix['suites'].items():
        targets = spec['targets'] if suite == 'transfer' else matrix['capabilities']
        for model in spec['models']:
            for target in targets:
                result.append(Job(f'{suite}/{model}/initial/{target}', suite,
                                  'environment' if suite == 'transfer' else 'capability',
                                  model, 'initial', 'zero-shot', None, target))
    return result


def read_bindings(path):
    if path is None:
        return {}
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate binding key: {key}')
            result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict) or set(value) - {'checkpoints', 'projectors', 'initial_models'}:
        raise ValueError('Bindings accept checkpoints, projectors and initial_models mappings only')
    for group in value.values():
        if not isinstance(group, dict) or any(not isinstance(v, str) or not v.strip() for v in group.values()):
            raise ValueError('Each binding must map a job/model key to an explicit path')
        if any(not Path(v).expanduser().is_absolute() for v in group.values()):
            raise ValueError('Binding paths must be absolute; latest/glob selection is not supported')
    return value


def checkpoint_for(job, bindings):
    value = bindings.get('checkpoints', {}).get(job.training_id)
    if not value:
        raise ValueError(f'Missing checkpoint binding: {job.training_id}')
    checkpoint = Path(value).expanduser().resolve()
    if not checkpoint.name.startswith('global_step_') or not checkpoint.name[12:].isdigit():
        raise ValueError('Checkpoint must select an exact global_step_N directory')
    if not (checkpoint / 'actor').is_dir():
        raise ValueError(f'Missing actor checkpoint: {checkpoint}')
    candidates = [checkpoint / 'model_config.json', checkpoint.parent / 'model_config.json']
    metadata = next((p for p in candidates if p.is_file()), None)
    if metadata is None:
        raise ValueError('Checkpoint needs its saved model_config.json')
    config = json.loads(metadata.read_text())
    algorithm = load_matrix()['methods'][job.method]
    # Share the pure metadata validator with policy export and capability results.
    spec = importlib.util.spec_from_file_location(
        'main_matrix_policy_identity', PROJECT / 'experiments/capability_eval/policy_identity.py')
    identity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity)
    saved = identity.training_identity(config)['training_method']
    if (config.get('benchmark') != job.source or saved != algorithm
            or config.get('model', {}).get('MODEL_NAME') != job.model):
        raise ValueError('Checkpoint source, method or model does not match the selected matrix job')
    return str(checkpoint)


def command(job, args, bindings):
    matrix = load_matrix()
    if job.target in matrix['blocked_targets']:
        raise ValueError(matrix['blocked_targets'][job.target])
    common = PROJECT / 'experiments/shared/train_eval'
    algorithm = matrix['methods'].get(job.method, 'grpo_react')
    if job.kind == 'train':
        cmd = ['bash', str(common / 'train.sh'), job.source, algorithm, '--model', job.model]
        if job.method.startswith('dyad-'):
            projector = bindings.get('projectors', {}).get(job.model)
            if not projector or not Path(projector).expanduser().exists():
                raise ValueError(f'Missing shared Alignment projector binding for {job.model}')
            cmd += ['--projector-init', projector]
        cmd += [f'data.seed={job.seed}', f'actor_rollout_ref.rollout.seed={job.seed}',
                f'actor_rollout_ref.actor.data_loader_seed={job.seed}']
    elif job.kind == 'environment':
        cmd = ['bash', str(common / 'evaluate.sh'), job.target, algorithm, '--model', job.model]
        if job.method == 'zero-shot':
            if job.target != 't2bench':
                cmd += ['--weights', 'base']
        else:
            cmd += ['--checkpoint', checkpoint_for(job, bindings)]
    else:
        if not args.eval_python:
            raise ValueError('Capability jobs require --eval-python for the isolated evaluator')
        if job.target == 'livecodebench_v6' and not args.sandbox_config:
            raise ValueError('LiveCodeBench requires --sandbox-config for isolated scoring')
        cmd = [sys.executable, str(PROJECT / 'docker_entrypoint/capability_eval/job.py'),
               '--eval-python', str(args.eval_python), '--label', job.id,
               '--benchmarks', job.target, '--tensor-parallel-size', str(args.tensor_parallel_size)]
        if job.method == 'zero-shot':
            model_path = bindings.get('initial_models', {}).get(job.model)
            if not model_path or not Path(model_path).expanduser().is_dir():
                raise ValueError(f'Missing exact initial HF model binding for {job.model}')
            cmd += ['--model-path', model_path]
        else:
            cmd += ['--checkpoint', checkpoint_for(job, bindings)]
        if job.target == 'livecodebench_v6':
            cmd += ['--sandbox-config', str(args.sandbox_config)]
    if job.kind != 'capability' and args.hardware:
        cmd += ['--hardware', args.hardware]
    return cmd


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', nargs='+', type=int, default=[42], help='Explicit training seeds; default plans one seed')
    p.add_argument('--suite', choices=['transfer', 'retention'])
    p.add_argument('--kind', choices=['train', 'environment', 'capability'])
    p.add_argument('--model', choices=['Qwen3.5-4B', 'Qwen3.5-9B', 'Qwen3.5-27B'])
    p.add_argument('--id', help='Exact job ID from the listing')
    p.add_argument('--bindings', type=Path, help='Explicit checkpoint/projector/initial-model path mappings')
    p.add_argument('--hardware')
    p.add_argument('--eval-python', type=Path)
    p.add_argument('--tensor-parallel-size', type=int, default=1)
    p.add_argument('--sandbox-config', type=Path)
    p.add_argument('--json', action='store_true')
    action = p.add_mutually_exclusive_group()
    action.add_argument('--check', action='store_true', help='Run the selected public entrypoint preflight, not a training job')
    action.add_argument('--execute', action='store_true', help='Execute exactly one explicitly selected job')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
            raise ValueError('Training seeds must be distinct nonnegative integers')
        if args.tensor_parallel_size < 1:
            raise ValueError('Tensor parallel size must be positive')
        if (args.execute or args.check) and not args.id:
            raise ValueError('--execute/--check requires --id; bulk execution is deliberately disabled')
        bindings = read_bindings(args.bindings)
        known = jobs(args.seeds)
        valid_training = {j.id for j in known if j.kind == 'train'}
        if set(bindings.get('checkpoints', {})) - valid_training:
            raise ValueError('Checkpoint bindings contain unknown training IDs; include the corresponding --seeds')
        checkpoint_paths = [str(Path(path).expanduser().resolve()) for path in bindings.get('checkpoints', {}).values()]
        if len(set(checkpoint_paths)) != len(checkpoint_paths):
            raise ValueError('One checkpoint cannot represent multiple independent training jobs or seeds')
        selected = [j for j in known if (not args.suite or j.suite == args.suite)
                    and (not args.kind or j.kind == args.kind)
                    and (not args.model or j.model == args.model) and (not args.id or j.id == args.id)]
        if not selected:
            raise ValueError('No matching matrix job')
        records = []
        for job in selected:
            record = asdict(job)
            try:
                record['command'] = command(job, args, bindings)
                record['status'] = 'planned'
            except (ValueError, OSError) as error:
                record.update(status='blocked', reason=str(error), command=None)
            records.append(record)
        if args.execute or args.check:
            record = records[0]
            if record['status'] != 'planned':
                raise ValueError(record['reason'])
            cmd = record['command'] + (['--check'] if args.check else [])
            environment = os.environ.copy()
            # A reused terminal must not turn initial evaluation into checkpoint evaluation.
            for key in ('EVAL_MODE', 'EVAL_CHECKPOINT', 'EVAL_MODEL_CONFIG', 'EVAL_PROJECTOR_INIT',
                        'DYAD_ENCODER_PROJECTOR_INIT', 'EVAL_ALGO', 'MODEL_PATH', 'MODEL_NAME'):
                environment.pop(key, None)
            return subprocess.run(cmd, cwd=PROJECT, env=environment, check=False).returncode
        if args.json:
            print(json.dumps(records, ensure_ascii=False, indent=2))
        else:
            for record in records:
                print(f"{record['id']} [{record['status']}]")
                print('  ' + (shlex.join(record['command']) if record['command'] else record['reason']))
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(f'main experiments: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
