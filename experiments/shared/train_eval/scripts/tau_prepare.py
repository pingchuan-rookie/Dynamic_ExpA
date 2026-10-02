"""Prepare runtime-initialized environments for the shared Ray evaluation runner."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
ROOT = PROJECT
CONFIG = HERE.parent / 'config/tau.yaml'


def require_benchmark(benchmark):
    if benchmark != 't2bench':
        raise ValueError(f'Unsupported benchmark: {benchmark!r}; use t2bench for tau evaluation.')


def parser(benchmark, env):
    require_benchmark(benchmark)
    document = yaml.safe_load(CONFIG.read_text())
    defaults = dict(document['defaults'], **document['environments'][benchmark])
    prefix = benchmark.upper()
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    from experiments.shared.train_eval.reference.t2bench.config import EnvironmentArgumentParser
    p = EnvironmentArgumentParser(description='Environment evaluation through the standalone vLLM/Dyad endpoint.',
                                  allow_abbrev=False, role_env=env,
                                  role_base_url='http://127.0.0.1:8001/v1')
    p.add_argument('algorithm', nargs='?', choices=['baseline', 'grpo_react', 'gigpo', 'dyad', 'dyad-grpo', 'dyad-gigpo'], default='grpo_react')
    p.add_argument('--backend', choices=['inference', 'reference'], default='inference')
    p.add_argument('--model', default=env.get('MODEL_NAME', env.get(f'{prefix}_AGENT_MODEL', defaults['model'])))
    p.add_argument('--hardware', default=env.get('HARDWARE_PROFILE'))
    weights = p.add_mutually_exclusive_group()
    weights.add_argument('--model-path', default=None)
    weights.add_argument('--checkpoint')
    weights.add_argument('--projector-init')
    p.add_argument('--model-config')
    p.add_argument('--api-url', default=env.get('EVAL_API_URL'))
    p.add_argument('--served-model', default=env.get('EVAL_SERVED_MODEL', 'evaluation-model'))
    p.add_argument('--domain', default=defaults['domain'])
    p.add_argument('--task-split', default=defaults['task_split'])
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--task-ids', nargs='+')
    selection.add_argument('--limit', type=int)
    p.add_argument('--shuffle', action='store_true')
    p.add_argument('--debug', action='store_true')
    for name in ('seed', 'num_trials', 'max_concurrency', 'max_steps', 'max_errors',
                 'max_retries', 'max_tokens', 'context_length', 'max_prompt_length', 'max_assistant_turns',
                 'history_length'):
        p.add_argument('--' + name.replace('_', '-'), type=int, default=defaults[name])
    for name in ('timeout', 'episode_timeout', 'temperature'):
        p.add_argument('--' + name.replace('_', '-'), type=float, default=defaults[name])
    p.add_argument('--thinking', choices=['default', 'on', 'off'], default=defaults['thinking'])
    for role in ('user', 'judge'):
        p.add_argument(f'--{role}-model', default=env.get(f'{prefix}_{role.upper()}_MODEL'))
        p.add_argument(f'--{role}-provider', default=env.get(f'{prefix}_{role.upper()}_PROVIDER'))
        p.add_argument(f'--{role}-base-url', default=env.get(f'{prefix}_{role.upper()}_BASE_URL'))
        p.add_argument(f'--{role}-api-key-env', default=f'{prefix}_{role.upper()}_API_KEY')
        p.add_argument(f'--{role}-temperature', type=float, default=defaults[f'{role}_temperature'])
        p.add_argument(f'--{role}-max-tokens', type=int, default=defaults[f'{role}_max_tokens'])
        p.add_argument(f'--{role}-thinking', choices=['default', 'on', 'off'], default=defaults[f'{role}_thinking'])
    p.add_argument('--output-dir', '--save-to', type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Check sources and task selection without model/service requests')
    mode.add_argument('--dry-run', action='store_true', help='Show the inference evaluation command without loading weights or starting Ray')
    mode.add_argument('--preflight', action='store_true', help='Check environment dependencies and user/judge services only')
    return p, defaults


def resolve_thinking(args):
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    from agent_system.utils.thinking import resolve_chat_template_kwargs

    for role in ('agent', 'user', 'judge'):
        prefix = '' if role == 'agent' else f'{role}_'
        value = getattr(args, prefix + 'thinking')
        model = args.model if role == 'agent' else getattr(args, f'{role}_model')
        sources = [model]
        if role == 'agent':
            sources.extend(getattr(args, key, None) for key in ('model_path', 'checkpoint'))
        settings = resolve_chat_template_kwargs(
            {} if value == 'default' else {'enable_thinking': value == 'on'}, model=sources)
        if settings:
            if role != 'agent' and getattr(args, f'{role}_provider') != 'openai':
                raise ValueError(f'{role} thinking override requires provider openai')
            setattr(args, prefix + 'thinking', 'on' if settings['enable_thinking'] else 'off')


def validate(args, benchmark):
    require_benchmark(benchmark)
    resolve_thinking(args)
    domains = ('retail', 'airline', 'telecom')
    if args.domain not in (*domains, 'all', 'both'):
        raise ValueError(f'Unsupported domain {args.domain!r}')
    if args.task_split != 'base':
        raise ValueError('Ray evaluation currently uses the pinned base evaluation snapshot')
    for name in ('num_trials', 'max_concurrency', 'max_steps', 'max_errors', 'max_tokens',
                 'context_length', 'max_prompt_length', 'max_assistant_turns', 'user_max_tokens', 'judge_max_tokens'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    for name in ('timeout', 'episode_timeout'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if args.history_length < 0:
        raise ValueError('history_length must be nonnegative')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    if args.max_retries < 0 or not 0 <= args.seed < 2**31:
        raise ValueError('Invalid seed or retry count')
    if args.max_prompt_length >= args.context_length or args.max_tokens >= args.context_length:
        raise ValueError('Prompt and per-decision generation limits must be smaller than context-length')
    if args.task_ids is not None and len(set(args.task_ids)) != len(args.task_ids):
        raise ValueError('Duplicate task IDs are not allowed')
    if args.debug and (args.num_trials != 1 or (args.limit or 0) > 3 or len(args.task_ids or []) > 3):
        raise ValueError('Debug permits at most three tasks per domain and one trial')
    for name in ('temperature', 'user_temperature', 'judge_temperature'):
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 2:
            raise ValueError(f'{name} must be finite and in [0, 2]')
    for role in ('user', 'judge'):
        url = urlsplit(getattr(args, f'{role}_base_url'))
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError(f'{role} endpoint must be HTTP(S), without embedded credentials, query or fragment')
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', getattr(args, f'{role}_api_key_env')):
            raise ValueError(f'{role} api-key-env must name an environment variable')
    if args.algorithm != 'dyad' and args.projector_init:
        raise ValueError('Projector requires Dyad')
    if args.algorithm != 'dyad' and args.model_config and not args.checkpoint:
        raise ValueError('Model configuration requires a checkpoint or Dyad projector')
    if args.algorithm == 'dyad' and not (args.projector_init or args.checkpoint):
        raise ValueError('Dyad requires an exact Alignment projector or full AgenticRL checkpoint')


def task_config(args, benchmark):
    require_benchmark(benchmark)
    resolve_thinking(args)
    from agent_system.environments.prompts.protocol import task_prompt_protocol
    excluded = {'backend', 'hardware', 'model_path', 'checkpoint', 'projector_init', 'model_config',
                'output_dir', 'check', 'dry_run', 'preflight', 'api_url', 'served_model'}
    config = {key: value for key, value in vars(args).items() if key not in excluded}
    config['benchmark'] = benchmark
    config['task_prompt_protocol'] = task_prompt_protocol(benchmark)
    config['domains'] = ['retail', 'airline', 'telecom'] if args.domain in ('all', 'both') else [args.domain]
    config['split'] = args.task_split
    config['limit'] = args.limit if args.limit is not None else (3 if args.debug else -1)
    config['session_config'] = {key: value for key, value in config.items()
                                if key not in {'domains', 'task_ids', 'limit', 'shuffle', 'num_trials'}}
    return config


def validate_saved_model_overrides(saved, defaults):
    """Accept only saved settings already enforced by the tau runtime command."""
    overrides = saved.get('model_overrides', [])
    if not isinstance(overrides, list):
        raise ValueError('Saved model_overrides must be a list')
    supported = 'actor_rollout_ref.model.override_config.attn_implementation'
    for override in overrides:
        if not isinstance(override, str) or '=' not in override:
            raise ValueError('Unsupported saved model override')
        key, value = override.split('=', 1)
        if key.lstrip('+') != supported:
            raise ValueError('Unsupported saved model override')
        if value != defaults['attn_impl']:
            raise ValueError('Saved attention override conflicts with the tau runtime attention implementation')


def configure_weights(args, env, defaults=None):
    from prepare import MODEL_KEYS, canonical_model, checkpoint_algorithm, experiment_recipes, resolve_alignment_projector
    if defaults is None:
        defaults = yaml.safe_load(CONFIG.read_text())['defaults']
    env.update(MODEL_NAME=args.model, RESUME_MODE='disable')
    for key in ('RESUME_CKPT', 'DYAD_NATIVE_RESTORE_DIR', 'DYAD_NATIVE_MODEL_CONFIG',
                'DYAD_ENCODER_PROJECTOR_INIT', 'EVAL_MODEL_CONFIG', 'EVAL_MODEL_OVERRIDES'):
        env.pop(key, None)
    if args.model_path:
        env['MODEL_PATH'] = str(Path(args.model_path).expanduser().resolve())
    if args.algorithm in {'grpo_react', 'gigpo'}:
        if args.checkpoint:
            # Baseline native checkpoint restoration stays on the existing loader.
            from prepare import checkpoint_model_config, load_model_config
            env['EVAL_CHECKPOINT'] = args.checkpoint
            source = Path(args.checkpoint).expanduser()
            if (source / 'config.json').is_file() and not (source / 'actor').is_dir():
                env['MODEL_PATH'] = str(source.resolve())
            else:
                config, saved = checkpoint_model_config(source, args.model_config)
                if config is None:
                    raise ValueError('Native text baseline checkpoint requires its saved model_config.json')
                env['EVAL_MODEL_CONFIG'] = str(config.resolve())
                if canonical_model(saved.get('model', {}).get('MODEL_NAME', '')) != canonical_model(args.model):
                    raise ValueError('--model must match the checkpoint model identity')
                validate_saved_model_overrides(saved, defaults)
                load_model_config(env, saved['benchmark'], args.algorithm)
        return None
    for key, value in experiment_recipes()['train']['defaults'].items():
        if key in MODEL_KEYS:
            env.setdefault(key, str(value))
    if args.model_config:
        path = Path(args.model_config).expanduser().resolve()
    elif args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        candidates = (checkpoint / 'model_config.json', checkpoint.parent / 'model_config.json')
        path = next((p for p in candidates if p.is_file()), candidates[-1])
    else:
        path = None
    if path:
        from agent_system.policies.dyad.checkpoint_compat import read_saved_model_config
        saved = read_saved_model_config(path)
        if saved.get('version') != 1 or saved.get('algo') != 'dyad' or not isinstance(saved.get('model'), dict):
            raise ValueError('Expected a saved version=1 Dyad model configuration')
        estimator = checkpoint_algorithm(saved).adv_estimator
        if env.get('DYAD_ADV_ESTIMATOR', estimator) != estimator:
            raise ValueError('Selected Dyad loss does not match checkpoint model configuration')
        env['DYAD_ADV_ESTIMATOR'] = estimator
        validate_saved_model_overrides(saved, defaults)
        if canonical_model(saved['model'].get('MODEL_NAME', '')) != canonical_model(args.model):
            raise ValueError('--model must match the checkpoint model identity')
        for key, value in saved['model'].items():
            if key not in MODEL_KEYS or not isinstance(value, str):
                raise ValueError(f'Unsupported model configuration key: {key}')
            if key in {'MODEL_NAME', 'MODEL_PATH', 'DYAD_ENCODER_MODEL_PATH'}:
                if key == 'MODEL_PATH' and args.model_path and args.checkpoint:
                    raise ValueError('A native checkpoint cannot replace its policy with --model-path')
                if key != 'MODEL_PATH' or not args.model_path:
                    env[key] = value
            elif key.startswith('DYAD_ENCODER_') or key in {'DYAD_TRAINING_SCHEDULE', 'DYAD_PROJECTOR_LR'}:
                env[key] = value
        env['EVAL_MODEL_CONFIG'] = str(path)
    if args.projector_init:
        env['DYAD_ENCODER_PROJECTOR_INIT'] = resolve_alignment_projector(args.projector_init, env)
        return None
    if not path:
        raise ValueError('Native checkpoint requires a saved model configuration')
    from agent_system.policies.dyad.inference.source import source_metadata
    source = source_metadata(argparse.Namespace(checkpoint=args.checkpoint, projector_init=None,
                                               model_config=str(path)))
    env.pop('DYAD_ENCODER_PROJECTOR_INIT', None)
    return source


def build_command(env, config, defaults, snapshot):
    from agent_system.evaluation.config import from_environment, command
    return command(from_environment(env, config['benchmark'], dataset=str(snapshot), task_config=config))


def reference(argv, benchmark, env):
    require_benchmark(benchmark)
    forwarded = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
        elif arg == '--backend':
            skip = True
        elif not arg.startswith('--backend='):
            forwarded.append(arg)
    python = env.get(f'{benchmark.upper()}_PYTHON_BIN', str(ROOT / f'.venvs/{benchmark}/bin/python'))
    return subprocess.call([python, str(ROOT / f'experiments/shared/train_eval/reference/{benchmark}/evaluate.py'), *forwarded], env=env)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print('[prepare] Expected t2bench', file=sys.stderr)
        return 2
    benchmark, argv = argv[0], argv[1:]
    try:
        require_benchmark(benchmark)
    except ValueError as exc:
        print(f'[prepare] {exc}', file=sys.stderr)
        return 2
    env = dict(os.environ)
    if '--backend=reference' in argv or any(argv[i:i+2] == ['--backend', 'reference'] for i in range(len(argv))):
        return reference(argv, benchmark, env)
    p, defaults = parser(benchmark, env)
    try:
        args = p.parse_args(argv)
        from prepare import apply_algorithm, baseline_checkpoint_algorithm
        if args.algorithm == 'baseline':
            args.algorithm = baseline_checkpoint_algorithm(args.checkpoint, args.model_config)
        if args.algorithm != 'dyad':
            args.algorithm = apply_algorithm(env, args.algorithm).algo
        validate(args, benchmark)
        return prepare(args, benchmark, defaults, env)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'[prepare] {exc}', file=sys.stderr)
        return 2


def environment_preflight(config, env, *, network=False):
    benchmark = config['benchmark']
    python = env.get(f'{benchmark.upper()}_PYTHON_BIN', str(ROOT / f'.venvs/{benchmark}/bin/python'))
    if not Path(python).is_file():
        raise ValueError(f'Missing environment interpreter: {python}')
    runtime = dict(env)
    runtime['PYTHONPATH'] = os.pathsep.join([str(ROOT), runtime.get('PYTHONPATH', '')])
    result = subprocess.run([python, '-m', 'agent_system.environments.env_package.t2bench.protocol', '--preflight' if network else '--check'],
                            input=json.dumps(config), text=True, capture_output=True, env=runtime,
                            timeout=config['timeout'] * (config['max_retries'] + 1) * 3 + 60)
    if result.returncode:
        # Model transports may contain provider credentials in exception text.
        raise ValueError(f'Environment {"service" if network else "offline"} preflight failed (exit {result.returncode})')
    for line in reversed(result.stdout.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get('benchmark') == benchmark:
            return value
    raise ValueError('Environment preflight returned no structured result')


def check_tau_resources(command, config, env):
    from prepare import effective_cpus
    required = 0.25 * config['max_concurrency']
    available = int(env.get('RAY_NUM_CPUS', effective_cpus()))
    if available < required:
        raise ValueError(f'Environment workers require at least {math.ceil(required)} CPUs')
    return {'runtime': 'inference_api', 'ray_cpus': available,
            'required_cpu_lower_bound': required,
            'reservations': {'tau environment actors': required}}, json.loads(command[-1])


def prepare(args, benchmark, defaults, env):
    from prepare import apply_algorithm, environment, effective_cpus, output_layout, resolve_model
    config = task_config(args, benchmark)
    if args.debug and args.limit is None and args.task_ids is None:
        config['limit'] = defaults['debug_limit']
    env['PATH'] = os.pathsep.join([str(Path(sys.executable).parent), env.get('PATH', '')])
    env.update(PYTHON_BIN=sys.executable, RUN_IS_EVAL='1', RUN_IS_DEBUG='1' if args.debug else '0',
               RUN_ALGO_BASE=args.algorithm, RUN_ALGO=args.algorithm, SCALE_PROFILE=args.hardware or 'evaluation',
               EFFECTIVE_CPUS=str(effective_cpus()), PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = os.pathsep.join([str(PROJECT), str(ROOT), env.get('PYTHONPATH', '')])
    for key in ('n_gpus_per_node', 'rollout_tp_size', 'gpu_mem_util', 'agent_num_workers'):
        env.setdefault(key.upper(), str(defaults[key]))
    source = configure_weights(args, env, defaults)
    if args.checkpoint:
        env['EVAL_CHECKPOINT'] = args.checkpoint
    if args.api_url:
        env['EVAL_API_URL'] = args.api_url
    env['EVAL_SERVED_MODEL'] = args.served_model
    selection = apply_algorithm(env, args.algorithm)
    env['RUN_ALGO'] = selection.public_alias
    resolve_model(env)
    if args.output_dir:
        if not args.output_dir.is_absolute():
            raise ValueError('--output-dir must be absolute')
        env['RUN_DIR'] = str(args.output_dir)
    output_layout(env, benchmark, args.algorithm)
    env.update(TAU_BENCHMARK=benchmark, TAU_EVAL_CONFIG=json.dumps(config),
               TAU_ENV_POOL_SIZE=str(args.max_concurrency),
               TAU_ENV_CALL_TIMEOUT=str(max(defaults['env_call_timeout'], args.timeout * (args.max_retries + 1) + 30)))
    if args.algorithm == 'dyad':
        env.update(DYAD_DYNAMIC_ACTIONS='1', DYAD_ACTION_CAPACITY=str(defaults['action_capacity']),
                   DYAD_ACTION_CONTEXT_DIR=str(Path(env['RUN_DIR']) / 'encoder_contexts'))
        env.setdefault('DYAD_ENCODER_MAX_LENGTH', str(defaults['encoder_max_length']))
        env.pop('DYAD_CODEGYM_ALL', None)
    snapshot = PROJECT / 'data' / benchmark / 'dataset' / (args.task_split + '.parquet')
    if not snapshot.is_file():
        raise ValueError(f'Missing task snapshot: {snapshot}')
    # Dataset only inspects public references; environment initialization happens in Ray.
    from omegaconf import OmegaConf
    from agent_system.environments.backends.tau.dataset import TauEvaluationDataset
    dataset = TauEvaluationDataset(str(snapshot), tokenizer=None, config=OmegaConf.create({'tau': config}))
    if not len(dataset):
        raise ValueError('Task selection is empty')
    resources, _ = check_tau_resources(build_command(env, config, defaults, snapshot), config, env)
    preflight = environment_preflight(config, env, network=not (args.check or args.dry_run))
    planned = {domain: {ep['task_id'] for ep in dataset.planned_episodes if ep['domain'] == domain}
               for domain in config['domains']}
    for domain, info in preflight['inventory'].items():
        selected = info.get('selected_task_ids', info.get('task_ids'))
        if selected is None or set(map(str, selected)) != planned.get(domain):
            raise ValueError(f'Official task selection disagrees with snapshot for {domain}')
    if args.check or args.preflight:
        print(json.dumps({'backend': 'inference_api', 'benchmark': benchmark, 'algorithm': args.algorithm,
                          'planned_episodes': len(dataset), 'snapshot': str(snapshot), 'preflight': preflight,
                          'resources': resources,
                          'weights': source['identity'] if source else env.get('DYAD_ENCODER_PROJECTOR_INIT', env['MODEL_PATH'])}, ensure_ascii=False))
        return 0
    command = build_command(env, config, defaults, snapshot)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    resources, values = check_tau_resources(command, config, env)
    import torch
    required_gpus = 0 if args.api_url else int(env['ROLLOUT_TP_SIZE'])
    if torch.cuda.device_count() < required_gpus:
        raise ValueError(f'Inference requires {required_gpus} visible GPUs')
    resources.update(required_gpus=required_gpus, visible_gpus=torch.cuda.device_count())
    directory = Path(env['RUN_DIR'])
    directory.mkdir(parents=True, exist_ok=False)
    from agent_system.environments.env_package.t2bench.trapi import TRAPI_ENV_VARS
    tool_config = {'tools': [{'class_name': 'agent_system.environments.backends.tau.tool.TauLocalEnvTool',
                             'config': {'type': 'native', 'env_type': benchmark, 'benchmark': benchmark,
                                        'tool_name': 'tau_session', 'pool_size': args.max_concurrency,
                                        'num_cpus_per_worker': 0.25,
                                        'python_executable': env.get(f'{benchmark.upper()}_PYTHON_BIN', str(ROOT / f'.venvs/{benchmark}/bin/python')),
                                        'forward_env_vars': [args.user_api_key_env, args.judge_api_key_env, *TRAPI_ENV_VARS],
                                        'reset_timeout_s': float(env['TAU_ENV_CALL_TIMEOUT']),
                                        'step_timeout_s': float(env['TAU_ENV_CALL_TIMEOUT']),
                                        'finalize_timeout_s': float(env['TAU_ENV_CALL_TIMEOUT'])}}]}
    (directory / 'environment_tool.yaml').write_text(yaml.safe_dump(tool_config, sort_keys=False))
    plan = {'mode': 'evaluation', 'benchmark': benchmark, 'algo': args.algorithm, 'env': env, 'command': command,
            'configuration': {'backend': 'inference_api', 'benchmark': benchmark, 'algorithm': args.algorithm,
                              'action_interface': selection.action_interface, 'adv_estimator': selection.adv_estimator,
                              'model': env['MODEL_NAME'], 'planned_episodes': len(dataset),
                              'history_length': config['history_length'],
                              'task_prompt_protocol': config['task_prompt_protocol'],
                              'tau': config, 'resources': resources, 'hydra': values}}
    if source:
        # The dynamic runtime builds target actions from each official task's schema;
        # the native restore discards the source action embeddings and encoder cache.
        plan['configuration'].update(
            native_source=source, source_benchmark=source['identity']['training_benchmark'],
            target_benchmark=benchmark,
            evaluation_checkpoint=source['source'], target_actions_rebuilt=True)
    with tempfile.TemporaryDirectory(prefix='dyad-eval-command-') as temporary:
        path = Path(temporary) / 'plan.json'
        path.write_text(json.dumps(plan))
        from run import run_process
        return run_process([sys.executable, str(HERE / 'run.py'), str(path)], env)
