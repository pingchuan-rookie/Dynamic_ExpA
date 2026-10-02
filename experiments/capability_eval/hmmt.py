"""Validate HMMT avg@4 against all four scored trials of every problem."""
import math
from pathlib import Path

from backend import protocol
from token_metrics import _records, _score


def validate_average(directory, config, count, score):
    expected = protocol('hmmt26')
    repeats = expected['repeats']
    generation = config.get('generation_config') or {}
    if (config.get('protocol') != expected or config.get('samples_per_problem') != repeats
            or generation.get('temperature') != expected['tasks']['temperature']
            or type(generation.get('seed')) is not int or generation.get('n', 1) != 1):
        raise ValueError('HMMT requires the recorded avg@4 sampling protocol')
    problems = min(config.get('limit') or expected['tasks']['expected_count'],
                   expected['tasks']['expected_count'])
    if count != problems * repeats:
        raise ValueError('HMMT avg@4 requires four answers per selected problem')
    root = Path(directory) / 'evalscope/reviews' / config['model']
    trials = {}
    for path in root.glob('hmmt26_*.jsonl'):
        for record in _records(path):
            sample = record.get('sample_score') or {}
            group, trial = sample.get('group_id'), sample.get('generation_index')
            value = _score(record)
            if (type(group) is not int or not 0 <= group < problems
                    or type(trial) is not int or not 0 <= trial < repeats
                    or record['index'] != group * repeats + trial or value is None
                    or (group, trial) in trials):
                raise ValueError('Invalid or duplicate HMMT avg@4 trial')
            trials[group, trial] = value
    if len(trials) != count:
        raise ValueError('Incomplete HMMT avg@4 review cache')
    if not math.isclose(sum(trials.values()) / count, score, rel_tol=0, abs_tol=1e-6):
        raise ValueError('HMMT report differs from average accuracy across four trials')
