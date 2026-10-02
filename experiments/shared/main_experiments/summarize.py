"""Render both main tables from explicitly bound, complete evaluation artifacts."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from environment_results import load_environment_result

# The public training launcher also owns a module named run; do not shadow it.
_matrix_spec = importlib.util.spec_from_file_location('main_experiments_matrix', Path(__file__).with_name('run.py'))
_matrix = importlib.util.module_from_spec(_matrix_spec)
sys.modules[_matrix_spec.name] = _matrix
_matrix_spec.loader.exec_module(_matrix)
checkpoint_for, jobs = _matrix.checkpoint_for, _matrix.jobs
load_matrix, read_bindings = _matrix.load_matrix, _matrix.read_bindings


def capability_module():
    path = Path(__file__).resolve().parents[2] / 'capability_eval/summarize.py'
    spec = importlib.util.spec_from_file_location('capability_table_summary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect(seeds, bindings, runs):
    planned = {job.id: job for job in jobs(seeds) if job.kind != 'train'}
    if not isinstance(runs, dict) or set(runs) - planned.keys():
        raise ValueError('Result index must map known evaluation job IDs to explicit run directories')
    if any(not isinstance(path, str) or not Path(path).expanduser().is_absolute() for path in runs.values()):
        raise ValueError('Result paths must be absolute directories')
    checkpoint_paths = [str(Path(path).expanduser().resolve()) for path in bindings.get('checkpoints', {}).values()]
    if len(set(checkpoint_paths)) != len(checkpoint_paths):
        raise ValueError('One checkpoint cannot represent multiple independent training jobs or seeds')
    matrix = load_matrix()
    capability = capability_module()
    results, rejected, protocols = {}, {}, {}
    for key, job in planned.items():
        if job.target in matrix['blocked_targets']:
            rejected[key] = matrix['blocked_targets'][job.target]
            continue
        if key not in runs:
            continue
        try:
            if job.kind == 'environment':
                result = load_environment_result(runs[key])
                expected_checkpoint = None if job.method == 'zero-shot' else checkpoint_for(job, bindings)
                expected_method = matrix['methods'].get(job.method, 'zero-shot')
                if (result['benchmark'] != job.target or result['model'] != job.model
                        or result['method'] != expected_method or result['checkpoint'] != expected_checkpoint):
                    raise ValueError('Environment result identity differs from its matrix slot')
                protocol_key = (job.suite, job.model, job.target)
                previous = protocols.setdefault(protocol_key, result['protocol'])
                if result['protocol'] != previous:
                    raise ValueError('Compared environment results use different evaluation protocols')
                results[key] = result
            elif job.method == 'zero-shot':
                # Paired summarization below verifies the initial model against trained provenance.
                result = capability.load_run(runs[key])
                initial = bindings.get('initial_models', {}).get(job.model)
                if not initial or Path(initial).resolve() != Path(result['model']).resolve():
                    raise ValueError('Initial capability model differs from explicit model binding')
                if result['benchmark'] != job.target or result['config']['model_identity'].get('policy_export'):
                    raise ValueError('Initial capability slot must contain the original model on the selected benchmark')
                results[key] = {'score_percent': 100 * result['score'], 'evidence': result['directory']}
            else:
                initial_key = f'retention/{job.model}/initial/{job.target}'
                if initial_key not in runs:
                    raise ValueError('Missing matching initial capability evaluation')
                result = capability.summarize(runs[initial_key], runs[key], checkpoint_for(job, bindings))
                if result['benchmark'] != job.target or result['training_benchmark'] != job.source:
                    raise ValueError('Capability result benchmark/source differs from its matrix slot')
                initial = bindings.get('initial_models', {}).get(job.model)
                identity_path = result['initial_model_identity']['path']
                if not initial or Path(initial).resolve() != Path(identity_path).resolve():
                    raise ValueError('Paired initial policy differs from explicit model binding')
                results[key] = result
        except (ValueError, OSError, KeyError, TypeError) as error:
            rejected[key] = str(error)
    # A protocol disagreement invalidates the comparison, not just whichever row was visited last.
    mismatched = { (planned[key].suite, planned[key].model, planned[key].target)
                   for key, reason in rejected.items() if 'different evaluation protocols' in reason }
    for key in list(results):
        job = planned[key]
        if (job.suite, job.model, job.target) in mismatched:
            rejected[key] = 'Compared environment results use different evaluation protocols'
            del results[key]
    return results, rejected


def render(seeds, results):
    matrix = load_matrix()
    lines = ['# Main experiment results', '', 'Scores are percentages; forgetting is initial minus trained, in percentage points.', '']
    names = {'zero-shot': 'Zero-shot', 'grpo': 'GRPO', 'gigpo': 'GiGPO',
             'dyad-grpo': 'Dyad + GRPO', 'dyad-gigpo': 'Dyad + GiGPO'}
    def score(key, field='score_percent'):
        result = results.get(key)
        return '-' if result is None else f'{result[field]:.1f}'
    for seed in seeds:
        lines += [f'## Training seed {seed}', '', '### Cross-environment performance', '',
                  '| Model | Method | DIVE: ALF | DIVE: T2 | DIVE: WS | DIVE: OOD | CodeGym: ALF | CodeGym: T2 | CodeGym: WS | CodeGym: OOD |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for model in matrix['suites']['transfer']['models']:
            for method in ('zero-shot', *matrix['methods']):
                cells = []
                for source in matrix['suites']['transfer']['sources']:
                    for target in matrix['suites']['transfer']['targets']:
                        key = (f'transfer/{model}/initial/{target}' if method == 'zero-shot' else
                               f'transfer/{model}/{source}/{method}/seed-{seed}/eval/{target}')
                        cells.append(score(key))
                lines.append('| ' + ' | '.join([model, names[method], *cells]) + ' |')
        lines += ['', '### Task performance and forgetting', '',
                  '| Model | Method | ALF | WS | Knowledge (ALF / WS) | Reasoning (ALF / WS) | Coding (ALF / WS) |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for model in matrix['suites']['retention']['models']:
            for method in matrix['methods']:
                prefix = lambda source: f'retention/{model}/{source}/{method}/seed-{seed}'
                cells = [score(prefix(source) + '/eval/' + source) for source in ('alfworld', 'webshop')]
                for target in matrix['capabilities']:
                    cells.append(' / '.join(score(prefix(source) + '/capability/' + target, 'forgetting')
                                            for source in ('alfworld', 'webshop')))
                lines.append('| ' + ' | '.join([model, names[method], *cells]) + ' |')
        lines.append('')
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--bindings', type=Path)
    p.add_argument('--runs', type=Path, required=True, help='JSON mapping evaluation job IDs to actual output directories')
    p.add_argument('--format', choices=['markdown', 'json'], default='markdown')
    args = p.parse_args(argv)
    try:
        if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
            raise ValueError('Training seeds must be distinct nonnegative integers')
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f'Duplicate result key: {key}')
                result[key] = value
            return result
        runs = json.loads(args.runs.read_text(), object_pairs_hook=unique)
        results, rejected = collect(args.seeds, read_bindings(args.bindings), runs)
        if args.format == 'json':
            print(json.dumps({'results': results, 'rejected': rejected}, indent=2, ensure_ascii=False, allow_nan=False))
        else:
            print(render(args.seeds, results))
            if rejected:
                print('\n## Unavailable or rejected results\n')
                for key, reason in rejected.items():
                    print(f'- `{key}`: {reason}')
        # Planned undefined DIVE-OOD slots are not errors unless supplied as purported measurements.
        return 2 if set(rejected) & set(runs) else 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'main experiment summary: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
