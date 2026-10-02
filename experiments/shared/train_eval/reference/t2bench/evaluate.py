#!/usr/bin/env python3
"""AgenticRL t2bench CLI. No training, model loading, GPU use or official-repo writes."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shlex
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[5]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# Set before importing any upstream package. Offline checks must not fetch price maps.
sys.dont_write_bytecode = True
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'

from config import BENCH, COMMIT, VERSION, git_identity, output_path, parser, validate_args, validate_services, weights_metadata


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main(argv=None):
    try:
        args = parser().parse_args(argv)
        validate_args(args)
        run, site = output_path(args)
        from experiments.shared.train_eval.reference.t2bench.dyad_client import source_metadata
        weights = source_metadata(args) if args.algorithm == 'dyad' else weights_metadata(args.checkpoint or args.model_path)
        benchmark = git_identity(BENCH)
        if benchmark['commit'] != COMMIT:
            raise ValueError(f'Expected official t2bench commit {COMMIT}; found {benchmark["commit"]}')
        if benchmark['runtime_data_dirty']:
            raise ValueError('Official t2bench src/data has changes; separately audit it before evaluation')
        data_override = os.getenv('TAU2_DATA_DIR')
        if data_override and Path(data_override).expanduser().resolve() != (PROJECT / 'data/t2bench/source/data').resolve():
            raise ValueError('TAU2_DATA_DIR must point to the pinned official data/t2bench/source/data')
        os.environ['TAU2_DATA_DIR'] = str((PROJECT / 'data/t2bench/source/data').resolve())
        sys.path.insert(0, str(BENCH / 'src'))
        # Upstream logs full messages and raw SDK errors. Keep CLI output compact and safe.
        from loguru import logger
        logger.remove()
        from runtime import TASK_PROMPT_PROTOCOL, TEXT_PROTOCOL, inventory, probe, solve_episode, summarize
        import tau2
        if not Path(tau2.__file__).resolve().is_relative_to(BENCH.resolve()):
            raise ValueError('tau2 imported from an unexpected installation')
        tasks, task_info = inventory(args)
        judge_required = any(info['judge_required_task_ids'] for info in task_info.values())
        checked = {'status': 'checked_offline', 'benchmark': {**benchmark, 'label': 't2bench includes tau3, not original tau2 paper results'},
                   'algorithm': args.algorithm, 'agent_protocol': 'dyad' if args.algorithm == 'dyad' else 'text_react', 'text_protocol': TEXT_PROTOCOL, 'tasks': task_info,
                   'task_prompt_protocol': TASK_PROMPT_PROTOCOL, 'history_length': args.history_length,
                   'num_trials': args.num_trials, 'planned_episodes': sum(len(t) for t in tasks.values()) * args.num_trials,
                   'completed_episodes': 0, 'judge_required': judge_required, 'output_dir': str(run)}
        if args.check:
            print(json.dumps(checked, ensure_ascii=False, indent=2))
            return 0
        validate_services(args, judge_required)
        args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        command = [str(Path(sys.executable).absolute()), str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
        config = {**args_dict, **checked, 'status': 'configured', 'site': site, 'weights': weights,
                  'command': command, 'shell_command': shlex.join(command),
                  'adapter_version': VERSION, 'adapter_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                  'benchmark_path': str(BENCH), 'evaluation_type': 'ALL', 'user_implementation': 'official UserSimulator',
                  'packages': {name: importlib.metadata.version(name) for name in ('litellm', 'pydantic', 'jsonschema', 'httpx')},
                  'seed_note': 'Stable per-domain/task/trial derived seed, passed to official participants; backend determinism not guaranteed',
                  'weights_note': 'Local files verified separately from server /models root attestation; endpoint-only runs do not attest weight files',
                  'tokenizer': args.tokenizer or (weights['source'] if weights['local_files_verified'] else args.model)}
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / 'resolved_config.json', config)
        summaries = {d: summarize([], [t.id for t in ts], args.num_trials) for d, ts in tasks.items()}
        write_json(run / 'summary.json', summaries)
        roles = ['agent', 'user', *(['judge'] if judge_required else [])]
        identities = {}
        try:
            for role in roles:
                identities[role] = probe(args, role, weights if role == 'agent' else None)
                write_json(run / 'endpoint_preflight.json', identities)
        except Exception as exc:
            from runtime import ServiceError, error_category
            error = {'role': role, 'category': error_category(exc), 'type': type(exc).__name__,
                     'message': str(exc) if isinstance(exc, ServiceError) else 'Endpoint preflight failed',
                     'planned_episodes': checked['planned_episodes'], 'completed_episodes': 0}
            write_json(run / 'preflight_error.json', error)
            print(json.dumps({'status': 'preflight_failed', 'output_dir': str(run), 'error': error}))
            return 1
        if args.preflight:
            print(json.dumps({'status': 'preflight_passed', 'completed_episodes': 0, 'output_dir': str(run)}))
            return 0
        with (run / 'run.log').open('x') as log:
            for domain, selected in tasks.items():
                domain_dir = run / domain
                domain_dir.mkdir()
                results = []
                for trial in range(args.num_trials):
                    for index, task in enumerate(selected):
                        result = solve_episode(args, domain, task, trial)
                        # IDs can contain slashes or long composite telecom names. Never use them as paths.
                        filename = f'task_{index:04d}_{hashlib.sha256(task.id.encode()).hexdigest()[:12]}_trial_{trial}.json'
                        result['trajectory_path'] = str(domain_dir / filename)
                        write_json(domain_dir / filename, result)
                        results.append(result)
                        summaries[domain] = summarize(results, [t.id for t in selected], args.num_trials)
                        write_json(domain_dir / 'summary.json', summaries[domain])
                        write_json(domain_dir / 'results.json', results)
                        write_json(run / 'summary.json', summaries)
                        log.write(json.dumps({k: result[k] for k in ('domain', 'task_id', 'trial', 'status', 'reward', 'trajectory_path')}, ensure_ascii=False) + '\n')
                        log.flush()
        complete = all(s['metrics_complete'] for s in summaries.values())
        print(json.dumps({'status': 'completed' if complete else 'incomplete', 'output_dir': str(run), 'summary': str(run / 'summary.json')}))
        return 0 if complete else 1
    except (ValueError, OSError, ImportError) as exc:
        # Configuration errors have no request headers; do not expose upstream exception reprs.
        print(json.dumps({'status': 'configuration_error', 'error': str(exc), 'help': 'Run evaluate.py --help; use .venvs/t2bench/bin/python'}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
