"""Pair complete capability runs without inferring checkpoint identity from labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f'Non-finite JSON number: {value}')


def _json(path):
    path = Path(path)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError(f'Metadata exceeds 64 MiB: {path}')
    value = json.loads(path.read_text(), object_pairs_hook=_object, parse_constant=_nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _hashes(value):
    return (isinstance(value, dict) and bool(value)
            and all(isinstance(key, str) and isinstance(item, str) and len(item) == 64
                    and all(c in '0123456789abcdef' for c in item) for key, item in value.items()))


def _architecture(config):
    # Serialization provenance and loading precision do not change architecture.
    ignored = {'_name_or_path', 'transformers_version', 'torch_dtype', 'dtype',
               '_attn_implementation', 'attn_implementation'}
    if isinstance(config, dict):
        return {key: _architecture(value) for key, value in config.items() if key not in ignored}
    if isinstance(config, list):
        return [_architecture(value) for value in config]
    return config


def load_run(directory):
    """Return a validated score and its recorded protocol; never load model weights."""
    from backend import protocol

    directory = Path(directory).expanduser().resolve()
    config = _json(directory / 'config.json')
    status = _json(directory / 'status.json')
    if (config.get('evaluation_kind') != 'full' or config.get('limit') is not None
            or status.get('evaluation_kind') != 'full'):
        raise ValueError('Only full capability runs are eligible; smoke results are not table scores')
    if status.get('status') != 'completed' or status.get('scoring') != 'completed':
        raise ValueError('Run is incomplete or scoring failed')
    benchmark = config.get('benchmark')
    if benchmark not in ('mmlu_pro', 'hmmt26', 'livecodebench_v6'):
        raise ValueError('Unsupported capability benchmark')
    expected = protocol(benchmark)
    backend = _json(directory / 'backend_protocol.json')
    if (config.get('protocol') != expected or any(backend.get(k) != v for k, v in expected.items())
            or backend.get('status') != 'completed' or backend.get('full_benchmark') is not True):
        raise ValueError('Run protocol differs from the pinned complete benchmark protocol')
    if (config.get('single_turn') is not True or config.get('tools_enabled') is not False
            or config.get('samples_per_problem') != expected['repeats']):
        raise ValueError('Capability scores require single-turn policy-only evaluation without tools')
    generation = config.get('generation_config')
    if (not isinstance(generation, dict) or generation.get('temperature') != expected['tasks'].get('temperature', 0.0)
            or type(generation.get('max_tokens')) is not int or generation['max_tokens'] < 1
            or type(generation.get('seed')) is not int or generation.get('n', 1) != 1):
        raise ValueError('Missing or unsupported generation protocol')
    extra = generation.get('extra_body') or {}
    if (not isinstance(extra, dict) or not isinstance(extra.get('chat_template_kwargs'), dict)
            or type(extra['chat_template_kwargs'].get('enable_thinking')) is not bool
            or any(k in source for source in (generation, extra)
                   for k in ('tools', 'tool_choice', 'functions', 'function_call'))):
        raise ValueError('Missing thinking setting or forbidden model tools')
    dataset = backend.get('dataset') or {}
    spec = expected['tasks']
    count = spec['expected_count']
    if (dataset.get('count') != count or dataset.get('repo') != spec['repo']
            or dataset.get('revision') != spec['revision'] or not _hashes(dataset.get('source_sha256'))
            or set(dataset['source_sha256']) != set(spec['files'])):
        raise ValueError('Dataset count, revision or source hashes are invalid')
    ids = dataset.get('ordered_ids')
    if (not isinstance(ids, list) or len(ids) != count or any(not isinstance(i, str) for i in ids)
            or len(set(ids)) != count
            or dataset.get('ordered_ids_sha256') != hashlib.sha256(json.dumps(ids).encode()).hexdigest()):
        raise ValueError('Dataset ordered task identities are incomplete')
    count *= expected['repeats']
    task = 'capability_livecodebench_v6' if benchmark == 'livecodebench_v6' else spec['task']
    reports = []
    for path in (directory / 'evalscope/reports').rglob('*.json'):
        report = _json(path)
        if report.get('dataset_name') == task:
            reports.append((path, report))
    if len(reports) != 1:
        raise ValueError('Expected exactly one report for the benchmark; missing or ambiguous reports')
    path, report = reports[0]
    execution = report.get('execution_summary') or {}
    if (report.get('model_name') != config.get('model') or report.get('num') != count
            or execution.get('requested') != count or execution.get('succeeded') != count
            or execution.get('errored') != 0 or execution.get('incomplete') is not False):
        raise ValueError('Report model identity, sample count or execution completeness mismatch')
    primary = report.get('primary_metric_identity')
    wanted = {'name': 'accuracy', 'aggregation': 'mean', 'dimensions': {}}
    if benchmark == 'livecodebench_v6':
        wanted = {'name': 'accuracy', 'aggregation': 'pass_at_k', 'dimensions': {'k': 1}}
    if primary != wanted or report.get('primary_metric_unavailable_reason'):
        raise ValueError('Unsupported or unavailable primary metric')
    metrics = [m for m in report.get('metrics', []) if isinstance(m, dict) and m.get('identity') == primary]
    if len(metrics) != 1:
        raise ValueError('Missing or ambiguous primary metric')
    metric = metrics[0]
    score = metric.get('score')
    semantics = metric.get('semantics') or {}
    if (type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1
            or metric.get('num') != count or semantics.get('display_multiplier') != 100
            or semantics.get('value_range') != {'min': 0.0, 'max': 1.0}
            or semantics.get('direction') != 'higher_is_better'):
        raise ValueError('Primary score is not a complete finite ratio in [0, 1]')
    if benchmark == 'hmmt26':
        from hmmt import validate_average
        validate_average(directory, config, count, score)
    identity = config.get('model_identity') or {}
    metadata = identity.get('metadata_sha256')
    if not _hashes(metadata) or 'config.json' not in metadata:
        raise ValueError('Missing evaluated policy metadata identity')
    model = Path(identity.get('path', '')).expanduser().resolve()
    metadata_path = Path(identity.get('metadata_path', model)).expanduser().resolve()
    for name, digest in metadata.items():
        if Path(name).name != name or _digest(metadata_path / name) != digest:
            raise ValueError('Evaluated policy metadata has changed or contains a nonlocal filename')
    architecture = _json(metadata_path / 'config.json')
    if not architecture.get('model_type'):
        raise ValueError('Policy model configuration lacks model_type')
    tokenizers = {name: digest for name, digest in metadata.items()
                  if name in ('tokenizer.json', 'tokenizer.model') or name.endswith('.jinja')}
    if not any(name in tokenizers for name in ('tokenizer.json', 'tokenizer.model')):
        raise ValueError('Missing tokenizer identity')
    if 'tokenizer_config.json' in metadata:
        tokenizer_config = _json(metadata_path / 'tokenizer_config.json')
        tokenizer_config.pop('name_or_path', None)
        tokenizer_config.pop('_name_or_path', None)
        tokenizers['tokenizer_config.json'] = tokenizer_config
    return {'directory': str(directory), 'benchmark': benchmark, 'score': score, 'count': count // expected['repeats'],
            'config': config, 'dataset': dataset, 'model': model, 'architecture': _architecture(architecture),
            'tokenizers': tokenizers, 'metric': primary, 'report_path': str(path),
            'evidence_sha256': {str(p.relative_to(directory)): _digest(p) for p in
                               (directory / 'config.json', directory / 'status.json',
                                directory / 'backend_protocol.json', path)}}


def summarize(initial_run, trained_run, checkpoint):
    initial, trained = load_run(initial_run), load_run(trained_run)
    if initial['directory'] == trained['directory']:
        raise ValueError('Initial and trained runs must be distinct')
    for key in ('benchmark', 'count', 'dataset', 'metric', 'architecture', 'tokenizers'):
        if initial[key] != trained[key]:
            raise ValueError(f'Initial/trained {key} mismatch')
    for key in ('generation_config', 'protocol', 'single_turn', 'tools_enabled', 'samples_per_problem', 'sandbox_config'):
        if initial['config'].get(key) != trained['config'].get(key):
            raise ValueError(f'Initial/trained {key} mismatch')
    original = initial['config']['model_identity']
    exported = trained['config']['model_identity'].get('policy_export') or {}
    if original.get('policy_export') or (initial['model'] / 'policy_export.json').exists():
        raise ValueError('Initial run must use the initial model, not a post-training export')
    if initial['model'] == trained['model']:
        raise ValueError('Initial and trained runs cannot refer to the same policy directory')
    if trained['config']['model_identity'].get('native_checkpoint'):
        return summarize_native(initial, trained, checkpoint)
    manifest_path = trained['model'] / 'policy_export.json'
    if (exported.get('artifact_hashes_verified') is not True
            or exported.get('manifest_sha256') != _digest(manifest_path)):
        raise ValueError('Trained run lacks the verified policy export used during evaluation')
    manifest = _json(manifest_path)
    if (manifest.get('version') != 1 or any(manifest.get(k) is not True for k in
            ('policy_only', 'dtype_preserved', 'policy_strict_load', 'serialized_tensor_equality_verified'))
            or manifest.get('action_encoder_loaded') is not False or manifest.get('projector_loaded') is not False):
        raise ValueError('Incomplete trained policy export provenance')
    selected = str(Path(checkpoint).expanduser().resolve())
    source = manifest.get('source_checkpoint')
    algorithm = manifest.get('training_algorithm')
    if (not source or str(Path(source).resolve()) != selected or exported.get('source_checkpoint') != source
            or algorithm not in ('grpo_react', 'gigpo', 'dyad', 'dyad-grpo', 'dyad-gigpo')
            or exported.get('training_algorithm') != algorithm):
        raise ValueError('Trained policy export differs from the selected checkpoint or algorithm')
    source_hashes = manifest.get('source_files_sha256')
    if not _hashes(source_hashes) or not _hashes(manifest.get('artifact_files_sha256')):
        raise ValueError('Trained export lacks source/artifact content hashes')
    # Hash-bound training metadata proves which initial model was selected without
    # reading native pickle shards or relying on the evaluation label.
    source_config = Path(selected).parent / 'model_config.json'
    if source_hashes.get(str(source_config)) != _digest(source_config):
        raise ValueError('Source training metadata is missing or changed since policy export')
    training = _json(source_config)
    source_model = (training.get('model') or {}).get('MODEL_PATH')
    if not source_model or Path(source_model).expanduser().resolve() != initial['model']:
        raise ValueError('Initial policy does not match the source checkpoint training model path')
    if training.get('algo') != algorithm:
        raise ValueError('Training algorithm differs from export provenance')
    from policy_identity import training_identity
    method_identity = training_identity(training)
    for key, value in method_identity.items():
        if key in manifest and manifest[key] != value:
            raise ValueError('Export method axes differ from source training identity')
    if training.get('benchmark') not in ('alfworld', 'webshop'):
        raise ValueError('Main forgetting table only accepts ALFWorld or WebShop training sources')
    return {'benchmark': initial['benchmark'], 'unit': 'percentage_points',
            'initial_score': 100 * initial['score'], 'post_training_score': 100 * trained['score'],
            'forgetting': 100 * (initial['score'] - trained['score']), 'samples': initial['count'],
            'source_checkpoint': selected, 'training_algorithm': algorithm, **method_identity,
            'training_benchmark': training.get('benchmark'),
            'initial_run': initial['directory'], 'trained_run': trained['directory'],
            'initial_model_identity': original, 'trained_model_identity': trained['config']['model_identity'],
            'protocol': initial['config']['protocol'], 'generation_config': initial['config']['generation_config'],
            'metric': initial['metric'],
            'evidence_sha256': {'initial': initial['evidence_sha256'], 'trained': trained['evidence_sha256']},
            'verification': 'Recorded evaluation/export identities and current metadata verified; model weight values and native shards are not re-read'}


def summarize_native(initial: dict, trained: dict, checkpoint: str | Path) -> dict:
    """Bind a complete checkpoint evaluation to its actual serving identity."""
    from agent_system.policies.dyad.checkpoint_compat import read_saved_model_config
    from agent_system.inference.policy_identity import training_identity, inspect_policy_weights
    from agent_system.inference.export_policy import checkpoint_files, file_digest
    selected = Path(checkpoint).expanduser().resolve()
    identity = trained['config']['model_identity']
    endpoint = trained['config'].get('endpoint_identity', {})
    served = endpoint.get('identity', {})
    if (trained['model'] != selected or Path(endpoint.get('root', '')).resolve() != selected
            or served.get('restoration_complete') is not True):
        raise ValueError('Trained inference endpoint differs from the selected complete checkpoint')
    training = read_saved_model_config(selected.parent / 'model_config.json')
    method = training_identity(training)
    if method != identity.get('training_identity'):
        raise ValueError('Checkpoint method identity changed after evaluation')
    if identity.get('action_interface') == 'dyad':
        from types import SimpleNamespace
        from agent_system.policies.dyad.inference.source import source_metadata
        source = source_metadata(SimpleNamespace(checkpoint=str(selected), projector_init=None, model_config=None))
        if source != identity.get('source') or served.get('source_identity_sha256') != source['identity_sha256']:
            raise ValueError('Dyad serving source differs from the complete evaluated checkpoint')
        if served.get('action_interface') != 'dyad' or not served.get('projector_sha256'):
            raise ValueError('Dyad evaluation did not restore the action module')
    else:
        hashes = {p.name: file_digest(p) for p in checkpoint_files(selected)[0]}
        if hashes != identity.get('files_sha256'):
            raise ValueError('Native text checkpoint changed after evaluation')
        loaded = inspect_policy_weights(Path(served['policy_path']))
        if loaded != served.get('weights') or loaded.get('policy_export', {}).get('source_checkpoint') != str(selected):
            raise ValueError('Text serving policy export differs from the selected native checkpoint')
    if Path(training['model']['MODEL_PATH']).expanduser().resolve() != initial['model']:
        raise ValueError('Initial model differs from the checkpoint training source')
    if training.get('benchmark') not in ('alfworld', 'webshop'):
        raise ValueError('Main forgetting table only accepts ALFWorld or WebShop training sources')
    return {'benchmark': initial['benchmark'], 'unit': 'percentage_points',
            'initial_score': 100 * initial['score'], 'post_training_score': 100 * trained['score'],
            'forgetting': 100 * (initial['score'] - trained['score']), 'samples': initial['count'],
            'source_checkpoint': str(selected), 'training_algorithm': training['algo'], **method,
            'training_benchmark': training['benchmark'],
            'initial_run': initial['directory'], 'trained_run': trained['directory'],
            'initial_model_identity': initial['config']['model_identity'], 'trained_model_identity': identity,
            'protocol': initial['config']['protocol'], 'generation_config': initial['config']['generation_config'],
            'metric': initial['metric'],
            'evidence_sha256': {'initial': initial['evidence_sha256'], 'trained': trained['evidence_sha256']},
            'verification': 'Native checkpoint source and standalone inference restoration identities verified'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-run', type=Path, required=True)
    parser.add_argument('--trained-run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Exact native checkpoint expected for the trained run')
    args = parser.parse_args(argv)
    try:
        result = summarize(args.initial_run, args.trained_run, args.checkpoint)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'Capability summary rejected: {error}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
