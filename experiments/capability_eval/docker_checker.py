"""Fail-closed Docker transport for the pinned LiveCodeBench checker."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import tempfile
import time
from uuid import uuid4


class CheckerInfrastructureError(RuntimeError):
    pass


def validate_config(value):
    if not isinstance(value, dict):
        raise ValueError('LiveCodeBench requires an explicit Docker sandbox configuration')
    defaults = {'engine': 'docker', 'python': '/opt/checker-python/bin/python3.12',
                'memory_mb': 1024, 'cpus': 1, 'pids_limit': 64,
                'timeout_seconds': 120, 'output_limit_bytes': 1048576}
    if set(value) - (set(defaults) | {'image'}):
        raise ValueError('Unknown sandbox configuration fields')
    config = {**defaults, **value}
    if config['engine'] != 'docker':
        raise ValueError('Sandbox engine must be docker')
    if not re.fullmatch(r'(?:[^\s]+@)?sha256:[0-9a-f]{64}', str(config.get('image', ''))):
        raise ValueError('Sandbox image must be an immutable sha256 image ID or repository digest')
    if not isinstance(config['python'], str) or not re.fullmatch(r'/[A-Za-z0-9_./-]+', config['python']):
        raise ValueError('Sandbox Python must be an absolute executable path')
    bounds = {'memory_mb': (128, 8192), 'cpus': (0.1, 8), 'pids_limit': (16, 256),
              'timeout_seconds': (1, 3600), 'output_limit_bytes': (1024, 4194304)}
    for key, (low, high) in bounds.items():
        number = config[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not low <= number <= high:
            raise ValueError(f'Invalid sandbox limit: {key}')
        if key != 'cpus' and not isinstance(number, int):
            raise ValueError(f'Sandbox {key} must be an integer')
    return config


def _docker(*args):
    try:
        result = subprocess.run(['docker', *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CheckerInfrastructureError('Docker is unavailable or unresponsive') from exc
    if result.returncode:
        raise CheckerInfrastructureError(f'Docker {args[0]} failed: {result.stderr[-2000:]}')
    return result.stdout


def preflight(config):
    config = validate_config(config)
    info = json.loads(_docker('info', '--format', '{{json .}}'))
    security = ' '.join(info.get('SecurityOptions') or [])
    if 'seccomp' not in security or any(info.get(key) is False for key in ('MemoryLimit', 'PidsLimit')):
        raise CheckerInfrastructureError('Docker must support seccomp, memory and PID limits')
    image = json.loads(_docker('image', 'inspect', config['image']))[0]
    image_config = image.get('Config', {})
    if image_config.get('Volumes'):
        raise CheckerInfrastructureError('Checker image must not declare writable volumes')
    allowed_env = {'PATH', 'HOME', 'LANG', 'PYTHONDONTWRITEBYTECODE', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'}
    if any(item.partition('=')[0] not in allowed_env for item in image_config.get('Env') or []):
        raise CheckerInfrastructureError('Checker image contains unexpected inherited environment variables; use checker.Dockerfile')
    return {'image_id': image['Id'], 'security_options': info.get('SecurityOptions', [])}


def create_command(config, name, directory):
    return ['docker', 'create', '--name', name, '--network', 'none', '--read-only',
            '--user', '65534:65534', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--memory', f"{config['memory_mb']}m", '--memory-swap', f"{config['memory_mb']}m",
            '--cpus', str(config['cpus']), '--pids-limit', str(config['pids_limit']),
            '--ulimit', 'nofile=128:128', '--ulimit', 'fsize=16777216:16777216',
            '--tmpfs', '/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777',
            '--ipc', 'none', '--log-driver', 'none', '--workdir', '/tmp',
            '--env', 'CAPABILITY_CHECKER_CONTAINER=1', '--env', 'OPENBLAS_NUM_THREADS=1',
            '--env', 'OMP_NUM_THREADS=1', '--env', 'PYTHONDONTWRITEBYTECODE=1',
            '--mount', f'type=bind,src={directory},dst=/checker,readonly',
            '--entrypoint', config['python'], config['image'], '-I', '/checker/checker_runner.py']


def _capture(command, timeout, limit):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    streams = selectors.DefaultSelector()
    for pipe in (process.stdout, process.stderr):
        streams.register(pipe, selectors.EVENT_READ)
    buffers = {process.stdout: bytearray(), process.stderr: bytearray()}
    deadline = time.monotonic() + timeout
    stop_reason = None
    try:
        while streams.get_map():
            if time.monotonic() >= deadline:
                stop_reason = 'timeout'
                break
            for key, _ in streams.select(min(0.2, max(0, deadline - time.monotonic()))):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    streams.unregister(key.fileobj)
                    continue
                buffers[key.fileobj].extend(chunk)
                if sum(map(len, buffers.values())) > limit:
                    stop_reason = 'output_limit'
                    break
            if stop_reason:
                break
        if stop_reason:
            process.kill()
        process.wait(timeout=10)
        return process.returncode, bytes(buffers[process.stdout]), bytes(buffers[process.stderr]), stop_reason
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        streams.close()
        process.stdout.close()
        process.stderr.close()


def score(config, code, evaluation_sample, checker_source, test_timeout=6):
    config = validate_config(config)
    samples = json.loads(evaluation_sample)
    inputs, outputs = samples.get('inputs'), samples.get('outputs')
    if not isinstance(inputs, list) or not inputs or not isinstance(outputs, list) or len(inputs) != len(outputs):
        raise CheckerInfrastructureError('Malformed or empty LiveCodeBench test cases')
    name = f'capability-checker-{uuid4().hex}'
    created = False
    with tempfile.TemporaryDirectory(prefix='capability-checker-') as temporary:
        directory = Path(temporary)
        directory.chmod(0o755)
        (directory / 'request.json').write_text(json.dumps({'code': code, 'evaluation_sample': evaluation_sample,
                                                          'test_timeout': test_timeout}))
        (directory / 'testing_util.py').write_bytes(Path(checker_source).read_bytes())
        (directory / 'checker_runner.py').write_bytes(
            (Path(__file__).resolve().parents[2] / 'agent_system/rewards/capability_checker_runner.py').read_bytes()
        )
        for path in directory.iterdir():
            path.chmod(0o444)
        try:
            command = create_command(config, name, directory)
            _docker(*command[1:])
            created = True
            returncode, stdout, stderr, stopped = _capture(
                ['docker', 'start', '--attach', name], config['timeout_seconds'], config['output_limit_bytes'])
            state = json.loads(_docker('inspect', '--format', '{{json .State}}', name))
            if stopped and state.get('Running'):
                _docker('kill', name)
                state = json.loads(_docker('inspect', '--format', '{{json .State}}', name))
            if state.get('Error'):
                raise CheckerInfrastructureError(f"Checker container error: {state['Error']}")
            if stopped or state.get('OOMKilled'):
                if b'CAPABILITY_CHECKER_EXECUTION_STARTED' not in stderr:
                    raise CheckerInfrastructureError('Checker initialization exceeded its resource limits')
                return False, {'solution_failure': stopped or 'memory_limit', 'execution_method': 'hardened_docker'}
            if returncode or state.get('ExitCode'):
                raise CheckerInfrastructureError(f'Checker process failed: {stderr[-2000:].decode(errors="replace")}')
            try:
                result = json.loads(stdout)
            except (ValueError, UnicodeError) as exc:
                raise CheckerInfrastructureError('Checker returned invalid result protocol') from exc
            if not isinstance(result, dict):
                raise CheckerInfrastructureError('Checker returned a non-object result protocol')
            results = result.get('results')
            if (not isinstance(results, list) or not results or len(results) > len(inputs)
                    or any(type(v) not in (bool, int) or type(v) is int and v not in {-4, -3, -2, -1, 0, 1}
                           for v in results)):
                raise CheckerInfrastructureError('Checker returned malformed test results')
            passed = all(value is True or type(value) is int and value == 1 for value in results)
            if passed and len(results) != len(inputs):
                raise CheckerInfrastructureError('Checker returned incomplete passing test results')
            return passed, {'execution_method': 'hardened_docker', 'results': results,
                            'checker_sha256': hashlib.sha256(Path(checker_source).read_bytes()).hexdigest(),
                            'metadata': result.get('metadata', {})}
        finally:
            if created:
                _docker('rm', '--force', name)
