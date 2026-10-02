"""Online capability metrics with an explicit, text-free upload boundary."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time


BENCHMARKS = ('mmlu_pro', 'hmmt26', 'livecodebench_v6')


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def public_config(config):
    benchmarks = config.get('benchmarks') or [config['benchmark']]
    generations = config.get('generation_config_by_benchmark') or {
        name: config['generation_config'] for name in benchmarks}
    identity = config.get('model_identity') or {}
    export = identity.get('policy_export') or {}
    native = identity.get('training_identity') or {}
    result = {key: config[key] for key in ('label', 'code_commit', 'tensor_parallel_size',
                                          'benchmark_parallelism', 'ready_timeout',
                                          'safetensors_load_strategy', 'limit') if key in config}
    result.update(benchmarks=benchmarks, evaluation_kind='smoke' if config.get('limit') is not None else 'full',
                  eval_batch_size=config.get('resolved_eval_batch_size', config.get('eval_batch_size')),
                  generation={name: {key: value for key, value in generation.items()
                                     if key in ('temperature', 'seed', 'max_tokens')}
                              for name, generation in generations.items()},
                  token_scope='final_cached_policy_responses_excludes_unrecorded_retries',
                  model_metadata_sha256=identity.get('metadata_sha256', {}))
    for key in ('training_algorithm', 'manifest_sha256'):
        if key in export:
            result[key] = export[key]
    if native:
        result.update(native)
        result['model_action_interface'] = identity.get('action_interface')
    source = config.get('checkpoint') or config.get('model_path') or identity.get('path')
    if source:
        result['model_source_name'] = Path(source).name
    if config.get('checkpoint'):
        result['checkpoint_step'] = Path(config['checkpoint']).name
    return result


def benchmark_result(directory):
    """Validate full scores strictly and smoke scores against their limited reports."""
    from token_metrics import collect_token_metrics
    directory = Path(directory)
    config = read_json(directory / 'config.json')
    status = read_json(directory / 'status.json')
    if status.get('status') != 'completed' or status.get('scoring') != 'completed':
        raise ValueError('Only completed and scored benchmark outputs can be uploaded')
    benchmark = config['benchmark']
    if benchmark not in BENCHMARKS:
        raise ValueError('Unsupported benchmark')
    kind = config['evaluation_kind']
    if kind == 'full':
        from summarize import load_run
        validated = load_run(directory)
        score, count = validated['score'], validated['count']
    elif kind == 'smoke' and type(config.get('limit')) is int and config['limit'] > 0:
        task = config['protocol']['tasks']['task'] if benchmark != 'livecodebench_v6' else 'capability_livecodebench_v6'
        reports = [read_json(path) for path in (directory / 'evalscope/reports').rglob('*.json')]
        reports = [report for report in reports if report.get('dataset_name') == task]
        if len(reports) != 1:
            raise ValueError('Missing or ambiguous smoke report')
        report = reports[0]
        primary = {'name': 'accuracy', 'aggregation': 'mean', 'dimensions': {}}
        if benchmark == 'livecodebench_v6':
            primary.update(aggregation='pass_at_k', dimensions={'k': 1})
        metrics = [metric for metric in report.get('metrics', []) if metric.get('identity') == primary]
        execution = report.get('execution_summary') or {}
        count = report.get('num')
        if (report.get('model_name') != config.get('model') or report.get('primary_metric_identity') != primary
                or len(metrics) != 1 or type(count) is not int or count < 1
                or execution.get('requested') != count or execution.get('succeeded') != count
                or execution.get('errored') != 0 or execution.get('incomplete') is not False
                or metrics[0].get('num') != count):
            raise ValueError('Incomplete smoke report or metric mismatch')
        score = metrics[0]['score']
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid smoke score')
    else:
        raise ValueError('Invalid evaluation kind or limit')
    if benchmark == 'hmmt26' and kind == 'smoke':
        from hmmt import validate_average
        validate_average(directory, config, count, score)
        count //= config['samples_per_problem']
    tokens = collect_token_metrics(directory)
    return {'benchmark': benchmark, 'evaluation_kind': kind, 'score': score, 'samples': count,
            'max_tokens': config['generation_config']['max_tokens'], **tokens}


class Tracker:
    def __init__(self, config, directory, *, replay=False):
        import wandb
        self.started = time.monotonic()
        self.replay = replay
        settings = wandb.Settings(disable_code=True, disable_git=True, save_code=False,
                                  console='off', x_disable_meta=True, x_save_requirements=False,
                                  x_disable_stats=True)
        self.run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'capability_eval'),
                              entity=os.environ.get('WANDB_ENTITY') or None,
                              name=f'{config["label"]}-{Path(directory).name}',
                              job_type='capability_eval_upload' if replay else 'capability_eval',
                              tags=['capability_eval', 'smoke' if config.get('limit') is not None else 'full'],
                              config=public_config(config), mode='online', dir=str(directory), settings=settings)
        if self.run is None:
            raise RuntimeError('W&B did not create an online run')
        self.url = self.run.url
        print(f'capability_eval W&B: {self.url}', flush=True)

    def result(self, directory):
        result = benchmark_result(directory)
        name = result['benchmark']
        prefix = 'eval/' + name
        score_name = {'livecodebench_v6': 'pass_at_1', 'hmmt26': 'avg_at_4'}.get(name, 'accuracy')
        values = {f'scores/{name}/{score_name}': result['score'], f'{name}/samples': result['samples'],
                  f'{name}/max_tokens': result['max_tokens']}
        output_token_metrics = {'mean', 'correct_mean', 'incorrect_mean'}
        values.update({f'{prefix}/output_tokens/{key}': value for key, value in result['metrics'].items()
                       if value is not None and key in output_token_metrics})
        self.run.summary.update(values)
        self.run.log(values)
        return result

    def finish(self, state):
        if not self.replay:
            self.run.summary['elapsed_seconds'] = time.monotonic() - self.started
        self.run.finish(exit_code=0 if state == 'completed' else 1)


def upload_job(directory, destination):
    """Replay recorded outputs without altering original configs, reports or statuses."""
    directory, destination = Path(directory), Path(destination)
    config = read_json(directory / 'config.json')
    summary = read_json(directory / 'summary.json')
    if summary.get('check_only') or summary.get('status') not in ('completed', 'failed'):
        raise ValueError('Expected a finished evaluation job, not preflight or a running job')
    kind = 'smoke' if config.get('limit') is not None else 'full'
    if (summary.get('evaluation_kind') != kind or summary.get('label') != config.get('label')
            or set(summary['benchmarks']) != set(config['benchmarks'])):
        raise ValueError('Job summary differs from recorded configuration')
    # Validate all eligible evidence before publishing any data.
    for name, outcome in summary['benchmarks'].items():
        if name not in config['benchmarks']:
            raise ValueError('Job benchmark selection mismatch')
        if outcome['status'] == 'completed':
            child = read_json(Path(outcome['output_dir']) / 'config.json')
            if (child.get('model_identity') != config.get('model_identity')
                    or child.get('model') != config.get('model') or child.get('label') != config.get('label')
                    or child.get('benchmark') != name or child.get('evaluation_kind') != kind
                    or child.get('limit') != config.get('limit')
                    or child.get('generation_config') != config.get('generation_config_by_benchmark', {}).get(
                        name, config['generation_config'])):
                raise ValueError('Child evaluation identity or generation protocol differs from job')
            benchmark_result(outcome['output_dir'])
    destination.mkdir(parents=True, exist_ok=False)
    tracker = Tracker(config, destination, replay=True)
    try:
        for name, outcome in summary['benchmarks'].items():
            if outcome['status'] == 'completed':
                tracker.result(outcome['output_dir'])
    except BaseException:
        tracker.finish('failed')
        raise
    tracker.finish(summary['status'])
    receipt = {'url': tracker.url, 'run_id': tracker.run.id, 'source_job': str(directory)}
    (destination / 'upload.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'W&B uploaded: {tracker.url}', flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--upload-dir', type=Path, required=True, help='New directory for W&B files and receipt')
    args = parser.parse_args(argv)
    try:
        upload_job(args.job_dir, args.upload_dir)
    except Exception as error:
        print(f'W&B upload failed ({type(error).__name__}): {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
