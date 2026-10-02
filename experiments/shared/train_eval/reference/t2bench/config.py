"""Evaluation-only configuration for the pinned, tau3-containing t2bench release."""
from __future__ import annotations
from agent_system.environments.env_package.t2bench.config import resolve_environment_roles, thinking_settings

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[5]
BENCH = ROOT / 'agent_system/environments/env_package/t2bench/source'
COMMIT = 'a2c024725189473d2d7cea3a5cfdbcc67478e41f'
COUNTS = {'retail': 114, 'airline': 50, 'telecom': 114}
VERSION = '1.1.0'




class EnvironmentArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, role_env=None, role_base_url='https://api.openai.com/v1', **kwargs):
        super().__init__(*args, **kwargs)
        self.role_env = role_env
        self.role_base_url = role_base_url

    def parse_args(self, *args, **kwargs):
        return resolve_environment_roles(super().parse_args(*args, **kwargs), self.role_env, self.role_base_url)


def parser():
    p = EnvironmentArgumentParser(description='Official t2bench (includes tau3) text ReAct evaluation, without training.', allow_abbrev=False)
    p.add_argument('algorithm', nargs='?', choices=['grpo_react', 'gigpo', 'dyad'], default='grpo_react')
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from experiments.shared.train_eval.reference.t2bench.dyad_client import add_arguments
    add_arguments(p)
    p.add_argument('--version', action='version', version=VERSION)
    p.add_argument('--domain', choices=[*COUNTS, 'all', 'both'], default='all', help='both is an alias for all three domains')
    p.add_argument('--task-split', choices=['base'], default='base')
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--task-ids', nargs='+', help='Exact official string IDs; use one domain for domain-specific IDs')
    selection.add_argument('--limit', type=int)
    p.add_argument('--shuffle', action='store_true')
    p.add_argument('--debug', action='store_true', help='Strictly at most 3 tasks per domain and one trial')
    p.add_argument('--seed', type=int, default=300)
    p.add_argument('--num-trials', type=int, default=1)
    p.add_argument('--max-concurrency', type=int, choices=[1], default=1, help='Sequential execution isolates upstream judge configuration')
    p.add_argument('--max-steps', type=int, default=200, help='Official orchestrator steps, not just agent generations')
    p.add_argument('--max-errors', type=int, default=10)
    p.add_argument('--history-length', type=int, default=2, help='Recent observation/action pairs in each decision prompt; 0 disables history')
    p.add_argument('--timeout', type=float, default=120, help='Per-request timeout in seconds')
    p.add_argument('--episode-timeout', type=float, default=1800, help='Official wallclock bound, checked between steps')
    p.add_argument('--max-retries', type=int, default=0, help='Transport retries only, never restart an episode')
    p.add_argument('--model', '--agent-model', default=os.getenv('T2BENCH_AGENT_MODEL', 'Qwen3.5-2B'))
    weights = p.add_mutually_exclusive_group()
    weights.add_argument('--model-path', default=os.getenv('MODEL_PATH'))
    weights.add_argument('--checkpoint', help='Dyad: native global_step_N with model_config.json; GRPO: complete exported HF weights')
    p.add_argument('--checkpoint-source', help='Original checkpoint path/step, provenance only')
    p.add_argument('--tokenizer', help='Served tokenizer identity, provenance only')
    p.add_argument('--chat-template', default='server tokenizer default')
    p.add_argument('--context-length', type=int, default=65536, help='Declared serving limit; no prompt truncation')
    for role in ('agent', 'user', 'judge'):
        prefix = '' if role == 'agent' else f'{role}-'
        if role != 'agent':
            p.add_argument(f'--{role}-model', default=os.getenv(f'T2BENCH_{role.upper()}_MODEL'))
        p.add_argument(f'--{role}-provider', default=os.getenv(f'T2BENCH_{role.upper()}_PROVIDER', 'openai' if role == 'agent' else None))
        p.add_argument(f'--{role}-base-url', default=os.getenv(f'T2BENCH_{role.upper()}_BASE_URL', 'http://127.0.0.1:8000/v1' if role == 'agent' else None))
        p.add_argument(f'--{role}-api-key-env', default=f'T2BENCH_{role.upper()}_API_KEY')
        p.add_argument(f'--{prefix}temperature', type=float, default=0.0)
        p.add_argument(f'--{prefix}max-tokens', type=int, default=2048)
        p.add_argument(f'--{prefix}thinking', choices=['default', 'on', 'off'], default='off' if role == 'agent' else 'default')
    p.add_argument('--output-dir', '--save-to', type=Path, help='New absolute directory inside ARTIFACT_ROOT/outputs')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--check', '--dry-run', action='store_true', help='Offline task IDs/counts, environment and configuration check; no requests or output creation')
    mode.add_argument('--preflight', action='store_true', help='Probe required services only; do not execute tasks')
    return p




def validate_args(a):
    if a.algorithm != 'dyad' and (a.projector_init or a.model_config):
        raise ValueError('--projector-init/--model-config require algorithm dyad')
    for name in ('num_trials', 'max_steps', 'max_errors', 'context_length', 'max_tokens', 'user_max_tokens', 'judge_max_tokens'):
        if getattr(a, name) < 1:
            raise ValueError(f'{name} must be positive')
    for name in ('timeout', 'episode_timeout'):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if type(a.history_length) is not int or a.history_length < 0:
        raise ValueError('history-length must be a nonnegative integer')
    if a.max_retries < 0 or not 0 <= a.seed < 2**31:
        raise ValueError('max-retries must be nonnegative; seed must be in [0, 2^31)')
    if a.limit is not None and a.limit < 1:
        raise ValueError('limit must be positive')
    if a.debug and (a.num_trials != 1 or (a.limit is not None and a.limit > 3) or (a.task_ids is not None and len(a.task_ids) > 3)):
        raise ValueError('debug permits at most 3 tasks per domain and exactly 1 trial')
    if a.task_ids is not None and len(set(a.task_ids)) != len(a.task_ids):
        raise ValueError('Duplicate task IDs are not allowed')
    if a.checkpoint and a.model_path:
        raise ValueError('Unset MODEL_PATH when using --checkpoint')
    if a.max_tokens >= a.context_length:
        raise ValueError('max-tokens must be smaller than context-length')
    for role in ('agent', 'user', 'judge'):
        prefix = '' if role == 'agent' else f'{role}_'
        temperature = getattr(a, prefix + 'temperature')
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError(f'{role} temperature must be in [0, 2]')
        url = urlsplit(getattr(a, f'{role}_base_url'))
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError(f'{role} endpoint must be HTTP(S), without embedded credentials, query or fragment')
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', getattr(a, f'{role}_api_key_env')):
            raise ValueError(f'{role} api-key-env must name an environment variable')
        settings = thinking_settings(a, role)
        if 'enable_thinking' in settings:
            setattr(a, prefix + 'thinking', 'on' if settings['enable_thinking'] else 'off')


def validate_services(a, judge_required):
    """Offline checks intentionally do not require network credentials or simulator IDs."""
    for role in ('agent', 'user', *(['judge'] if judge_required else [])):
        model = a.model if role == 'agent' else getattr(a, f'{role}_model')
        if not model:
            raise ValueError(f'Set --{role}-model (agent also accepts --model)')
        if getattr(a, f'{role}_provider') == 'trapi':
            continue  # Dynamic Entra authentication occurs only in the requesting worker.
        key_env = getattr(a, f'{role}_api_key_env')
        host = urlsplit(getattr(a, f'{role}_base_url')).hostname
        if not os.getenv(key_env) and host not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError(f'Set credential environment variable {key_env} for {role}')


def output_path(a):
    import sys
    project = ROOT
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from agent_system.utils.artifact_paths import artifact_root, run_site

    site = run_site()
    root = (artifact_root(project) / 'outputs').resolve()
    if a.output_dir and not a.output_dir.is_absolute():
        raise ValueError('--output-dir / --save-to must be absolute')
    run = (a.output_dir or root / site / 'agentic_rl' / a.algorithm / 't2bench' / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8])).resolve()
    if not run.is_relative_to(root) or run == root:
        raise ValueError(f'Output must be inside {root}')
    if run.exists():
        raise ValueError('Output directory already exists; existing user files are never overwritten')
    return run, site


def git_identity(path):
    from agent_system.environments.env_package.source_bundle import source_identity
    return source_identity(path)


def weights_metadata(source):
    if not source:
        return {'source': 'endpoint model ID', 'local_files_verified': False, 'server_weights_verified': False}
    path = Path(source).expanduser().resolve()
    if not (path / 'config.json').is_file() or not any((path / n).is_file() for n in ('tokenizer.json', 'tokenizer.model')):
        raise ValueError('Weights require complete HF config, tokenizer and nonempty weight files; export raw verl shards first')
    files = sorted([*path.glob('model*.safetensors'), *path.glob('pytorch_model*.bin')])
    if not files or any(not f.is_file() or f.stat().st_size == 0 for f in files):
        raise ValueError('Missing or empty HF weight files')
    for index in path.glob('*.index.json'):
        for shard in json.loads(index.read_text()).get('weight_map', {}).values():
            # Validate the index's lexical path, not the symlink target: HF snapshots
            # intentionally link their files to ../../blobs outside the snapshot.
            if not isinstance(shard, str) or not shard or Path(shard).is_absolute() or '..' in Path(shard).parts:
                raise ValueError('Missing, empty or invalid HF shard')
            f = path / shard
            if not f.is_file() or f.stat().st_size == 0:
                raise ValueError('Missing, empty or invalid HF shard')
    return {'source': str(path), 'local_files_verified': True, 'server_weights_verified': False,
            'config_sha256': hashlib.sha256((path / 'config.json').read_bytes()).hexdigest(),
            'tokenizer_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in path.glob('tokenizer*') if f.is_file()},
            'files': [{'name': f.name, 'size': f.stat().st_size} for f in files]}
