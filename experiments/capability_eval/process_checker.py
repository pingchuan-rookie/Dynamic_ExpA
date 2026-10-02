"""Resource-limited checker processes for our own models inside the eval container.

This is NOT a strong security sandbox: there is no network/filesystem namespace,
no aggregate cgroup/PID limit, and candidates share the checker's interpreter.
CPU and address-space limits are per process, not Docker-equivalent limits.
Process-group cleanup cannot contain deliberate session escape or result tampering.
Never use this transport to execute arbitrary hostile code on the host.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import tempfile
import time

if __package__:
    from .backend import EVALSCOPE_COMMIT
    from .docker_checker import CheckerInfrastructureError
else:
    from backend import EVALSCOPE_COMMIT
    from docker_checker import CheckerInfrastructureError


_DEFAULTS = {
    'engine': 'process', 'python': '/opt/checker-python/bin/python3.12',
    'timeout_seconds': 120, 'output_limit_bytes': 1048576,
    'memory_mb': 1024, 'cpu_seconds': 120,
    'runtime_identity': '/opt/capability-checker/runtime.json',
}
_STARTED = b'CAPABILITY_CHECKER_EXECUTION_STARTED\n'


def validate_config(value):
    if not isinstance(value, dict):
        raise ValueError('LiveCodeBench requires an explicit process checker configuration')
    if set(value) - set(_DEFAULTS):
        raise ValueError('Unknown process checker configuration fields')
    config = {**_DEFAULTS, **value}
    if config['engine'] != 'process':
        raise ValueError('Checker engine must be process')
    for key in ('python', 'runtime_identity'):
        if not isinstance(config[key], str) or not config[key].startswith('/') or '\x00' in config[key]:
            raise ValueError(f'Process checker {key} must be an absolute path')
    bounds = {'timeout_seconds': (1, 3600), 'output_limit_bytes': (1024, 4194304),
              'memory_mb': (128, 8192), 'cpu_seconds': (1, 3600)}
    for key, (low, high) in bounds.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f'Invalid process checker limit: {key}')
    return config


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON field: {key}')
        result[key] = value
    return result


def _json(value):
    def reject_constant(constant):
        raise ValueError(f'Non-finite JSON value: {constant}')
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('Non-finite JSON number')
        return number
    return json.loads(value, object_pairs_hook=_unique_object,
                      parse_constant=reject_constant, parse_float=finite_float)


def _runtime(config):
    try:
        manifest_path = Path(config['runtime_identity'])
        if manifest_path.stat().st_size > 65536:
            raise ValueError('Runtime manifest is too large')
        identity = _json(manifest_path.read_text())
        required = {'python_version', 'numpy_version', 'checker_sha256', 'evalscope_commit'}
        if not isinstance(identity, dict) or not required <= identity.keys():
            raise ValueError('Incomplete runtime manifest')
        if any(not isinstance(identity[key], str) or not identity[key] for key in required):
            raise ValueError('Invalid runtime identity fields')
        if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', identity['python_version']):
            raise ValueError('Runtime Python version must be major.minor.patch')
        if not re.fullmatch(r'[0-9a-f]{64}', identity['checker_sha256']):
            raise ValueError('Invalid runtime checker hash')
        if identity['evalscope_commit'] != EVALSCOPE_COMMIT:
            raise ValueError(f'Runtime EvalScope commit must match the pinned backend commit {EVALSCOPE_COMMIT}')
        checker_path = identity.get('checker_path', str(manifest_path.with_name('testing_util.py')))
        if not isinstance(checker_path, str) or not Path(checker_path).is_absolute():
            raise ValueError('Runtime checker path must be absolute')
        checker = Path(checker_path).read_bytes()
        if hashlib.sha256(checker).hexdigest() != identity['checker_sha256']:
            raise ValueError('Installed checker does not match runtime manifest')
        if not Path(config['python']).is_file() or not os.access(config['python'], os.X_OK):
            raise ValueError('Checker Python is not executable')
    except (OSError, ValueError, TypeError) as exc:
        raise CheckerInfrastructureError(f'Invalid process checker runtime: {exc}') from exc
    return {key: identity[key] for key in required}, checker


def _kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _capture(command, directory, config):
    # Do not inherit credentials, PYTHONPATH, proxy settings or service tokens.
    environment = {'PATH': '/usr/bin:/bin', 'HOME': str(directory), 'TMPDIR': str(directory),
                   'LANG': 'C.UTF-8', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                   'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
                   'PYTHONDONTWRITEBYTECODE': '1'}
    try:
        process = subprocess.Popen(command, cwd=directory, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True, close_fds=True)
    except OSError as exc:
        raise CheckerInfrastructureError(f'Cannot launch checker Python: {exc}') from exc
    streams = selectors.DefaultSelector()
    buffers = {process.stdout: bytearray(), process.stderr: bytearray()}
    total = 0
    stopped = None
    deadline = time.monotonic() + config['timeout_seconds']
    try:
        for pipe in buffers:
            streams.register(pipe, selectors.EVENT_READ)
        while streams.get_map() or process.poll() is None:
            # Also clean up descendants on normal leader exit, even if they kept
            # stdout open. Waiting for EOF first would turn success into timeout.
            if process.poll() is not None:
                _kill_group(process)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stopped = 'timeout'
                break
            for key, _ in streams.select(min(0.05, remaining)):
                chunk = os.read(key.fileobj.fileno(), min(65536, config['output_limit_bytes'] - total + 1))
                if not chunk:
                    streams.unregister(key.fileobj)
                    continue
                buffers[key.fileobj].extend(chunk)
                total += len(chunk)
                if total > config['output_limit_bytes']:
                    stopped = 'output_limit'
                    break
            if stopped:
                break
        _kill_group(process)
        process.wait(timeout=10)
        return process.returncode, bytes(buffers[process.stdout]), bytes(buffers[process.stderr]), stopped
    finally:
        _kill_group(process)
        process.wait(timeout=10)
        streams.close()
        for pipe in buffers:
            pipe.close()


def _execute(config, identity, checker, request, *, probe=False):
    # Explicit /tmp avoids inaccessible pytest/user TMPDIR parents after setuid.
    with tempfile.TemporaryDirectory(prefix='capability-process-', dir='/tmp') as temporary:
        directory = Path(temporary)
        work = directory / 'work'
        work.mkdir(mode=0o700)
        directory.chmod(0o711 if os.geteuid() == 0 else 0o700)
        if os.geteuid() == 0:
            os.chown(work, 65534, 65534)
        runner = Path(__file__).resolve().parents[2] / 'agent_system/rewards/capability_checker_runner.py'
        (directory / 'checker_runner.py').write_bytes(runner.read_bytes())
        (directory / 'testing_util.py').write_bytes(checker)
        (directory / 'request.json').write_text(json.dumps({**request, 'runtime_identity': identity}, allow_nan=False))
        for path in directory.iterdir():
            if path.is_file():
                path.chmod(0o444 if os.geteuid() == 0 else 0o400)
        limits = {key: config[key] for key in ('memory_mb', 'cpu_seconds')}
        command = [config['python'], '-I', '-B', str(directory / 'checker_runner.py'),
                   '--request-directory', str(directory), '--process-limits', json.dumps(limits)]
        if probe:
            command.append('--probe')
        return _capture(command, work, config)


def _decode(stdout):
    try:
        result = _json(stdout)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise CheckerInfrastructureError('Checker returned invalid result protocol') from exc
    if not isinstance(result, dict):
        raise CheckerInfrastructureError('Checker returned a non-object result protocol')
    return result


def preflight(config):
    config = validate_config(config)
    identity, checker = _runtime(config)
    returncode, stdout, stderr, stopped = _execute(config, identity, checker, {}, probe=True)
    if returncode or stopped:
        raise CheckerInfrastructureError(
            f'Process checker preflight failed ({stopped or returncode}): {stderr[-2000:].decode(errors="replace")}')
    result = _decode(stdout)
    if result != {'runtime_identity': identity}:
        raise CheckerInfrastructureError('Process checker preflight returned an unexpected runtime identity')
    return {**identity, 'python': config['python'], 'execution_method': 'bounded_process',
            'strong_sandbox': False}


def _validate_request(code, evaluation_sample, test_timeout):
    if not isinstance(code, str) or not isinstance(evaluation_sample, str):
        raise CheckerInfrastructureError('Checker code and evaluation_sample must be strings')
    if (isinstance(test_timeout, bool) or not isinstance(test_timeout, (int, float))
            or not math.isfinite(test_timeout) or not 0 < test_timeout <= 3600):
        raise CheckerInfrastructureError('Invalid checker test timeout')
    try:
        samples = _json(evaluation_sample)
    except (ValueError, RecursionError) as exc:
        raise CheckerInfrastructureError('Malformed LiveCodeBench evaluation sample') from exc
    if not isinstance(samples, dict):
        raise CheckerInfrastructureError('LiveCodeBench evaluation sample must be an object')
    inputs, outputs = samples.get('inputs'), samples.get('outputs')
    if (not isinstance(inputs, list) or not inputs or not isinstance(outputs, list)
            or len(inputs) != len(outputs) or any(not isinstance(v, str) for v in inputs + outputs)):
        raise CheckerInfrastructureError('Malformed or empty LiveCodeBench test cases')
    fn_name = samples.get('fn_name')
    if fn_name is not None and (not isinstance(fn_name, str) or not fn_name.isidentifier()):
        raise CheckerInfrastructureError('Malformed LiveCodeBench function name')
    if fn_name is not None:
        try:
            for value in inputs:
                for line in value.split('\n'):
                    _json(line)
            for value in outputs:
                _json(value)
        except (ValueError, RecursionError) as exc:
            raise CheckerInfrastructureError('Malformed call-based LiveCodeBench JSON test case') from exc
    return inputs


def score(config, code, evaluation_sample, checker_source, test_timeout=6):
    config = validate_config(config)
    inputs = _validate_request(code, evaluation_sample, test_timeout)
    identity, _ = _runtime(config)
    try:
        checker = Path(checker_source).read_bytes()
    except (OSError, TypeError) as exc:
        raise CheckerInfrastructureError('Cannot read requested checker source') from exc
    if hashlib.sha256(checker).hexdigest() != identity['checker_sha256']:
        raise CheckerInfrastructureError('Requested checker source does not match runtime manifest')
    request = {'code': code, 'evaluation_sample': evaluation_sample, 'test_timeout': test_timeout}
    returncode, stdout, stderr, stopped = _execute(config, identity, checker, request)
    failure = stopped
    if returncode in (-signal.SIGXCPU, -signal.SIGKILL):
        failure = failure or 'cpu_or_memory_limit'
    elif returncode == -signal.SIGXFSZ or b'CAPABILITY_CHECKER_FILE_LIMIT\n' in stderr:
        failure = failure or 'file_limit'
    elif b'CAPABILITY_CHECKER_MEMORY_LIMIT\n' in stderr:
        failure = failure or 'memory_limit'
    if failure:
        if _STARTED not in stderr:
            raise CheckerInfrastructureError(f'Checker initialization exceeded its resource limits: {failure}')
        return False, {'solution_failure': failure, 'execution_method': 'bounded_process',
                       'checker_sha256': identity['checker_sha256']}
    if returncode:
        raise CheckerInfrastructureError(f'Checker process failed: {stderr[-2000:].decode(errors="replace")}')
    result = _decode(stdout)
    results = result.get('results')
    if (set(result) != {'results', 'metadata'} or not isinstance(result['metadata'], dict)
            or not isinstance(results, list) or not results or len(results) > len(inputs)
            or any(type(v) not in (bool, int) or type(v) is int and v not in {-4, -3, -2, -1, 0, 1}
                   for v in results)):
        raise CheckerInfrastructureError('Checker returned malformed test results')
    passed = all(value is True or type(value) is int and value == 1 for value in results)
    if passed and len(results) != len(inputs):
        raise CheckerInfrastructureError('Checker returned incomplete passing test results')
    return passed, {'execution_method': 'bounded_process', 'results': results,
                    'checker_sha256': identity['checker_sha256'], 'metadata': result['metadata']}
