"""Read text-free token statistics from pinned EvalScope prediction and review caches."""
from __future__ import annotations

import json
from pathlib import Path


def _records(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict) or type(row.get('index')) is not int:
                    raise ValueError('Invalid EvalScope sample index')
                yield row


def _score(record):
    sample = record.get('sample_score') or {}
    if sample.get('sample_id') not in (None, record['index']):
        raise ValueError('Review sample ID differs from its index')
    score = sample.get('score') or {}
    if score.get('status') != 'success':
        return None
    values = score.get('value') or {}
    name = score.get('main_score_name')
    if name not in ('accuracy', 'acc'):
        return None
    value = values.get(name)
    if type(value) not in (bool, int, float) or value not in (0, 1):
        return None
    return float(value)


def collect_token_metrics(directory):
    """Count final cached responses only; discarded retry attempts are unavailable.

    Missing usage is never converted to zero. Only the three requested means are
    returned; cached scores and token counts are used transiently to compute them.
    """
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text())
    model, benchmark = config['model'], config['benchmark']
    if not isinstance(model, str) or Path(model).name != model or model in ('.', '..'):
        raise ValueError('Invalid cached model name')
    task = 'capability_livecodebench_v6' if benchmark == 'livecodebench_v6' else benchmark
    if benchmark not in ('mmlu_pro', 'hmmt26', 'livecodebench_v6'):
        raise ValueError('Unsupported capability benchmark')
    rows, reviewed = {}, set()
    for kind in ('predictions', 'reviews'):
        root = directory / 'evalscope' / kind / model
        for path in sorted(root.glob(f'{task}_*.jsonl')):
            subset = path.stem[len(task) + 1:]
            for record in _records(path):
                key = (subset, record['index'])
                if kind == 'reviews':
                    if key in reviewed:
                        raise ValueError('Duplicate EvalScope review')
                    reviewed.add(key)
                    row = rows.setdefault(key, {'subset': subset, 'index': record['index'],
                                               'completion_tokens': None, 'score': None})
                    row['score'] = _score(record)
                    continue
                if key in rows:
                    raise ValueError('Duplicate EvalScope prediction')
                if record.get('model') != model:
                    raise ValueError('Prediction model differs from evaluation config')
                output = record.get('model_output') or {}
                if output.get('model') not in (None, '', model):
                    raise ValueError('Response model differs from evaluation config')
                choices = output.get('choices') or []
                if not isinstance(choices, list) or len(choices) > 1:
                    raise ValueError('Expected at most one completion per sample')
                usage = output.get('usage')
                tokens = usage.get('output_tokens') if isinstance(usage, dict) else None
                if tokens is not None and (type(tokens) is not int or tokens < 0):
                    raise ValueError('Output token usage must be a nonnegative integer')
                rows[key] = {'subset': subset, 'index': record['index'], 'completion_tokens': tokens,
                             'score': None}
    samples = [rows[key] for key in sorted(rows)]
    reports = []
    for path in (directory / 'evalscope/reports' / model).glob('*.json'):
        report = json.loads(path.read_text())
        if report.get('dataset_name') == task and report.get('model_name') == model:
            reports.append(report)
    if len(reports) > 1:
        raise ValueError('Ambiguous benchmark report for token statistics')
    if reports:
        count = reports[0].get('num')
        if type(count) is not int or count < len(samples):
            raise ValueError('Report sample count is smaller than cached sample count')
    values = [row['completion_tokens'] for row in samples if row['completion_tokens'] is not None]
    metrics = {'mean': sum(values) / len(values) if values else None}
    for label, score in (('correct', 1), ('incorrect', 0)):
        group = [row['completion_tokens'] for row in samples
                 if row['score'] == score and row['completion_tokens'] is not None]
        metrics[f'{label}_mean'] = sum(group) / len(group) if group else None
    return {'metrics': {key: value for key, value in metrics.items() if value is not None}}
