"""Single-turn capability evaluation for capability_eval, through the shared generation endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse
from urllib.request import urlopen
from uuid import uuid4


DEFAULT_MAX_TOKENS = {'mmlu_pro': 2048, 'hmmt26': 8192, 'livecodebench_v6': 16384}


def resolve_max_tokens(benchmark, override=None):
    return DEFAULT_MAX_TOKENS[benchmark] if override is None else override


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('benchmark', choices=['mmlu_pro', 'hmmt26', 'livecodebench_v6'])
    p.add_argument('--model-path', type=Path,
                   help='Optional local HF model or native checkpoint identity; omit for an existing endpoint')
    p.add_argument('--model', default='capability_eval-policy', help='Served model name')
    p.add_argument('--api-url', default='http://127.0.0.1:8000/v1', help='vLLM/Dyad OpenAI-compatible endpoint')
    p.add_argument('--label', required=True, help='Checkpoint identity label, e.g. qwen3.5-4b-initial')
    p.add_argument('--output-dir', type=Path,
                   help='Exact new run directory; must not exist, including for --dry-run/--check. '
                        'Default: a unique timestamp directory under outputs/capability_eval/<benchmark>')
    p.add_argument('--wandb-managed', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--thinking', choices=['on', 'off'], default='off')
    p.add_argument('--max-tokens', type=positive,
                   help='Override generation budget (defaults: MMLU-Pro 2048; HMMT26 8192; LiveCodeBench v6 16384)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-batch-size', type=positive, default=1)
    p.add_argument('--limit', type=positive, help='Smoke test only; not a full capability_eval result')
    p.add_argument('--sandbox-config', type=Path, help='Explicit checker configuration; capability containers default to the image-integrated process checker')
    p.add_argument('--dry-run', action='store_true', help='Validate local inputs and print protocol without network/GPU work')
    p.add_argument('--check', action='store_true', help='Also verify evaluator dependency and serving identity, without evaluation')
    return p


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_identity(path):
    if path is None:
        return {"path": None, "identity_source": "endpoint"}
    path = path.expanduser().resolve()
    if (path / "actor").is_dir():
        project = Path(__file__).resolve().parents[2]
        if str(project) not in sys.path:
            sys.path.insert(0, str(project))
        from agent_system.policies.dyad.checkpoint_compat import read_saved_model_config
        saved = read_saved_model_config(path.parent / "model_config.json")
        from agent_system.inference.policy_identity import training_identity
        metadata_path = Path(saved['model']['MODEL_PATH']) if saved.get('algo') in {'dyad', 'dyad-grpo', 'dyad-gigpo'} else path / 'actor/huggingface'
        metadata = sorted({metadata_path / 'config.json', *metadata_path.glob('tokenizer*'), *metadata_path.glob('*.jinja')})
        identity = {"path": str(path), "native_checkpoint": True,
                    "metadata_path": str(metadata_path.resolve()),
                    "metadata_sha256": {p.name: digest(p) for p in metadata if p.is_file()},
                    "training_identity": training_identity(saved)}
        if saved.get("algo") in {"dyad", "dyad-grpo", "dyad-gigpo"}:
            from agent_system.policies.dyad.inference.source import source_metadata
            source = source_metadata(argparse.Namespace(checkpoint=str(path), projector_init=None, model_config=None))
            return {**identity, "action_interface": "dyad", "source": source}
        from agent_system.inference.export_policy import checkpoint_files, file_digest
        return {**identity, "action_interface": "text",
                "files_sha256": {p.name: file_digest(p) for p in checkpoint_files(path)[0]}}
    if not path.is_dir() or not (path / 'config.json').is_file():
        raise ValueError('Expected a complete HF policy directory with config.json; native actor/projector is unsupported')
    if (path / 'actor').exists() or (path / 'projector.pt').exists():
        raise ValueError('Use a verified pure-policy HF export, not a native checkpoint or projector directory')
    config = json.loads((path / 'config.json').read_text())
    if not config.get('model_type'):
        raise ValueError('HF config.json must declare model_type')
    from policy_identity import inspect_policy_weights
    inspected = inspect_policy_weights(path)
    indexes = list(path.glob('*.safetensors.index.json'))
    if not any((path / name).is_file() for name in ('tokenizer.json', 'tokenizer.model')):
        raise ValueError('HF policy directory must include its tokenizer')
    metadata = sorted({path / 'config.json', *indexes, *path.glob('tokenizer*'), *path.glob('*.jinja')})
    return {'path': str(path), 'metadata_sha256': {p.name: digest(p) for p in metadata if p.is_file()},
            **inspected}


def output_root():
    project = Path(__file__).resolve().parents[2]
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from agent_system.utils.artifact_paths import artifact_root

    return artifact_root(project) / 'outputs' / 'capability_eval'


def thinking_kwargs(thinking, *model_sources):
    project = Path(__file__).resolve().parents[2]
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from agent_system.utils.thinking import resolve_chat_template_kwargs

    return resolve_chat_template_kwargs({'enable_thinking': thinking == 'on'}, model=model_sources)


def generation_config(benchmark, max_tokens, seed, template_kwargs):
    from backend import protocol
    return {'temperature': protocol(benchmark)['tasks'].get('temperature', 0.0),
            'max_tokens': resolve_max_tokens(benchmark, max_tokens), 'seed': seed,
            'extra_body': {'chat_template_kwargs': template_kwargs, 'args': {}}}


def build_config(args):
    output_root()
    directory = args.output_dir.expanduser().absolute() if args.output_dir is not None else None
    # Do not resolve the final component: dangling symlinks are existing destinations too.
    if directory is not None and os.path.lexists(directory):
        raise ValueError(f'Output directory already exists: {directory}')
    url = urlparse(args.api_url)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('Use an HTTP(S) endpoint without credentials, query parameters or fragments')
    if url.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('This launcher only sends benchmark prompts to a loopback endpoint; use an SSH tunnel for remote serving')
    if url.path.rstrip('/') != '/v1':
        raise ValueError('--api-url must end in /v1')
    if not args.model.strip() or not args.label.strip():
        raise ValueError('Model and label must not be blank')
    identity = model_identity(args.model_path)
    template_kwargs = thinking_kwargs(args.thinking, args.model, args.model_path)
    sandbox = None
    if args.sandbox_config:
        sandbox = json.loads(args.sandbox_config.read_text())
        if not isinstance(sandbox, dict):
            raise ValueError('Sandbox configuration must be a JSON object')
    if args.benchmark != 'livecodebench_v6' and sandbox is not None:
        raise ValueError('--sandbox-config is only valid for livecodebench_v6')
    from backend import protocol, validate_sandbox
    from checker import integrated_runtime
    if sandbox is not None or (args.benchmark == 'livecodebench_v6' and integrated_runtime()):
        sandbox = validate_sandbox(sandbox)
    return {'benchmark': args.benchmark, 'model': args.model, 'api_url': args.api_url.rstrip('/'),
            'runtime_image': os.environ.get('CAPABILITY_IMAGE'),
            'runtime_image_id': os.environ.get('CAPABILITY_IMAGE_ID'),
            'label': args.label, 'model_identity': identity,
            'output_dir': str(directory) if directory is not None else None,
            'generation_config': generation_config(args.benchmark, args.max_tokens, args.seed, template_kwargs),
            'limit': args.limit, 'eval_batch_size': args.eval_batch_size,
            'sandbox_config': sandbox, 'protocol': protocol(args.benchmark),
            'evaluation_kind': 'smoke' if args.limit else 'full', 'single_turn': True,
            'tools_enabled': False, 'samples_per_problem': protocol(args.benchmark)['repeats']}


def verify_endpoint(config):
    with urlopen(config['api_url'] + '/models', timeout=15) as response:
        data = json.load(response)
    models = [m for m in data.get('data', []) if m.get('id') == config['model']]
    if len(models) != 1:
        raise ValueError('Endpoint must advertise exactly the requested served model')
    root = models[0].get('root')
    if not root or (config['model_identity'].get('path') and Path(root).resolve() != Path(config['model_identity']['path'])):
        raise ValueError('Endpoint model root differs from --model-path; start serve.sh with this exact HF directory')
    identity = models[0].get('identity', {})
    expected = config['model_identity'].get('source')
    if expected:
        if (identity.get('action_interface') != 'dyad' or identity.get('restoration_complete') is not True
                or identity.get('source_identity_sha256') != expected['identity_sha256']):
            raise ValueError('Endpoint did not restore the complete requested Dyad checkpoint')
    return {'id': models[0]['id'], 'root': root, 'identity': identity}


def code_version():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = build_config(args)
        if args.dry_run:
            print(json.dumps(config, indent=2, ensure_ascii=False))
            return 0
        from backend import preflight, run
        preflight(config)
        config['endpoint_identity'] = verify_endpoint(config)
        if args.check:
            print('capability_eval preflight passed; no model generation or benchmark scoring was run.')
            return 0
        if config['output_dir'] is not None:
            directory = Path(config['output_dir'])
        else:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            directory = output_root() / args.benchmark / f'{stamp}-{uuid4().hex[:12]}'
        # Recheck atomically at creation, before writing any evaluator-owned artifacts.
        directory.mkdir(parents=True, exist_ok=False)
        config['output_dir'] = str(directory)
        config['code_commit'] = code_version()
        config['evaluator_files_sha256'] = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        write_json(directory / 'config.json', config)
        print(f'capability_eval output: {directory}', flush=True)
        tracker = None
        state = 'failed'
        try:
            if not args.wandb_managed:
                from wandb_tracking import Tracker
                tracker = Tracker(config, directory)
            run(config, directory)
            write_json(directory / 'status.json', {'status': 'completed', 'scoring': 'completed',
                                                  'evaluation_kind': config['evaluation_kind']})
            if tracker is not None:
                result = tracker.result(directory)
                write_json(directory / 'token_metrics.json', result['metrics'])
            state = 'completed'
        except BaseException as error:
            write_json(directory / 'status.json', {'status': 'failed', 'error_type': type(error).__name__})
            raise
        finally:
            if tracker is not None:
                tracker.finish(state)
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f'capability_eval error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
