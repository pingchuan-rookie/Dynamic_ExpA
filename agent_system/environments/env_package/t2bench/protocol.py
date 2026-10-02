"""Shared, model-free session protocol and isolated environment preflight."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]


def load_session(benchmark):
    if benchmark != 't2bench':
        raise ValueError('Unknown environment')
    return importlib.import_module(f'agent_system.environments.env_package.{benchmark}.envs').T2BenchEnv


def arguments(benchmark, config):
    # Read defaults without importing the benchmark or contacting a model.
    if benchmark != 't2bench':
        raise ValueError('Unknown benchmark')
    defaults = dict(model='Qwen3.5-2B', algorithm='grpo_react', agent_provider='openai',
                    seed=10, num_trials=1, max_steps=200,
                    max_errors=10, timeout=120, episode_timeout=1800, max_retries=0,
                    context_length=32768, temperature=0.0, max_tokens=2048, thinking='off',
                    task_split='base', domain='all', task_ids=None,
                    limit=None, debug=False, shuffle=False)
    for role in ('user', 'judge'):
        prefix = f'{benchmark.upper()}_{role.upper()}'
        defaults.update({f'{role}_model': os.getenv(prefix + '_MODEL'),
                         f'{role}_provider': os.getenv(prefix + '_PROVIDER'),
                         f'{role}_base_url': os.getenv(prefix + '_BASE_URL'),
                         f'{role}_api_key_env': prefix + '_API_KEY',
                         f'{role}_temperature': 0.0, f'{role}_max_tokens': 2048,
                         f'{role}_thinking': 'default'})
    defaults.update(config)
    if defaults['limit'] == -1:
        # Evaluation configs share TauEvaluationDataset's -1 sentinel for the full selection.
        defaults['limit'] = None
    args = SimpleNamespace(**defaults)
    from .config import resolve_environment_roles, thinking_settings
    resolve_environment_roles(args, default_base_url='http://127.0.0.1:8001/v1')
    for role in ('agent', 'user', 'judge'):
        settings = thinking_settings(args, role)
        if 'enable_thinking' in settings:
            field = 'thinking' if role == 'agent' else f'{role}_thinking'
            setattr(args, field, 'on' if settings['enable_thinking'] else 'off')
    return args


def raw_action(action):
    if isinstance(action, str):
        return action
    if not isinstance(action, dict):
        raise ValueError('Action must be raw ReAct text or a structured action envelope')
    raw = action.get('raw_text', action.get('raw_output'))
    if raw is not None:
        if not isinstance(raw, str):
            raise ValueError('raw_text must be a string')
        return raw
    value = {key: action[key] for key in ('name', 'arguments')}
    return '<think></think><action>' + json.dumps(value, ensure_ascii=False, allow_nan=False) + '</action>'


def verify_snapshot(benchmark, reference):
    if reference is None:
        return
    import subprocess
    root = (ROOT / 'data' / benchmark / 'source').resolve()
    manifest = json.loads((root / 'source_manifest.json').read_text())
    if manifest['source_commit'] != reference['source_commit']:
        raise ValueError('Environment source version differs from task snapshot')
    if (root / '.git').exists():
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != reference['source_commit']:
            raise ValueError('Official source commit differs from task snapshot')
    files = [{'path': reference['source_path'], 'sha256': reference['source_sha256']},
             *reference['resources']]
    for item in files:
        selected_root = ROOT / 'agent_system/environments/env_package/t2bench/source' if item['path'].startswith('src/') else root
        path = (selected_root / item['path']).resolve()
        if not path.is_relative_to(selected_root) or not path.is_file():
            raise ValueError('Snapshot resource is outside the official source tree or missing')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != item['sha256']:
            raise ValueError('Official resource differs from task snapshot')


def schema_hash(tools):
    return hashlib.sha256(json.dumps(tools, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def result_identity(benchmark, domain, split, task_id, trial, seed, episode_id):
    return dict(benchmark=benchmark, domain=domain, split=split, task_id=str(task_id), trial=trial,
                seed=seed, episode_id=episode_id, status='running', official_scored=False,
                official_reward=None, reward=None, attempt_reward=None, metric_valid=False,
                task_completed=False, termination_reason=None, format_error_count=0)


def finalize_incomplete(result, reason):
    """Count exhausted rollout budgets as failed attempts, not official scores."""
    reason = str(reason)
    budget = reason in {'max_assistant_turns', 'max_user_turns', 'max_tool_turns', 'response_length'}
    result.update(status='max_steps' if budget else 'incomplete', termination_reason=reason,
                  metric_valid=budget, reward=0.0 if budget else None,
                  attempt_reward=0.0 if budget else None)


def failure(result, exc, phase):
    result.update(status='service_error' if phase in ('reset', 'user', 'judge') else 'internal_error',
                  reward=None, official_reward=None, attempt_reward=None,
                  official_scored=False, metric_valid=False,
                  error={'phase': phase, 'type': type(exc).__name__,
                         'http_status': getattr(exc, 'status_code', None)})


def response(messages, tools, result, *, initial=False, delta=None, steps=0, official_steps=0,
             format_error=False):
    done = result['status'] != 'running'
    value = {'protocol_version': 1, 'agent_messages': copy.deepcopy(messages),
             'action_tools': copy.deepcopy(tools), 'schema_hash': schema_hash(tools),
             'done': done, 'assistant_steps': steps, 'official_steps': official_steps,
             'format_error': format_error, 'format_error_count': result.get('format_error_count', 0),
             'messages_delta': copy.deepcopy(delta or []),
             'episode_result': copy.deepcopy(result) if done else None}
    if initial:
        value['initial_messages'] = copy.deepcopy(messages)
    return value


def preflight(config, network=False):
    benchmark = config['benchmark']
    if benchmark != 't2bench':
        raise ValueError('Unknown benchmark')
    os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
    a = arguments(benchmark, config)
    if a.task_split != 'base':
        raise ValueError('tau2 supports only the pinned base split')
    if a.task_ids is not None and (not a.task_ids or len(a.task_ids) != len(set(a.task_ids))):
        raise ValueError('Task IDs must be nonempty and unique')
    from agent_system.environments.env_package.t2bench.runtime import inventory, probe
    _, info = inventory(a)
    for domain_info in info.values():
        domain_info.pop('full_task_ids', None)
    services = {}
    if network:
        roles = ['user'] + (['judge'] if any(v['judge_required_task_ids'] for v in info.values()) else [])
        for role in roles:
            if not getattr(a, f'{role}_model'):
                raise ValueError(f'{role}_model is required')
            services[role] = probe(a, role)
    return {'benchmark': benchmark, 'inventory': info, 'services': services, 'network': network}


def main():
    import sys
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    try:
        config = json.load(sys.stdin)
        # Official imports may log or print; keep stdout a single protocol document.
        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            result = preflight(config, network=a.preflight)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({'status': 'error', 'type': type(exc).__name__}), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
