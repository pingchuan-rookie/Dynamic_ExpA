"""Run the pinned checker in Docker or a resource-limited evaluation process.

The process mode is NOT a strong security sandbox and is only for our own
models in the evaluation container, never arbitrary hostile host execution.
Candidates and the checker share an interpreter and result channel in both
modes; neither mode provides a score-authenticity boundary.
"""
import argparse
import contextlib
import datetime
import errno
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import platform
import sys
import types


def _process_limits(value):
    """Apply limits in this fresh interpreter, never through preexec_fn."""
    import resource

    limits = json.loads(value)
    if (not isinstance(limits, dict) or set(limits) != {'memory_mb', 'cpu_seconds'}
            or type(limits['memory_mb']) is not int or not 128 <= limits['memory_mb'] <= 8192
            or type(limits['cpu_seconds']) is not int or not 1 <= limits['cpu_seconds'] <= 3600):
        raise ValueError('Invalid checker process limits')
    os.umask(0o077)
    # Both soft and hard limits prevent accidental increases by the candidate.
    for kind, maximum in ((resource.RLIMIT_AS, limits['memory_mb'] * 1024 * 1024),
                          (resource.RLIMIT_CPU, limits['cpu_seconds']),
                          (resource.RLIMIT_FSIZE, 16 * 1024 * 1024),
                          (resource.RLIMIT_NOFILE, 128), (resource.RLIMIT_CORE, 0)):
        _, hard = resource.getrlimit(kind)
        maximum = maximum if hard == resource.RLIM_INFINITY else min(maximum, hard)
        resource.setrlimit(kind, (maximum, maximum))
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    if not os.access(sys.executable, os.X_OK):
        raise RuntimeError('Checker Python is inaccessible to the execution user')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request-directory', type=Path)
    parser.add_argument('--process-limits')
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if args.process_limits is not None:
        if args.request_directory is None or not args.request_directory.is_absolute():
            raise RuntimeError('Process checker requires an explicit absolute request directory')
        _process_limits(args.process_limits)
    elif os.environ.get('CAPABILITY_CHECKER_CONTAINER') != '1':
        # Backward compatibility for Docker /checker. This variable is merely
        # an invocation guard, not evidence of isolation for either transport.
        raise RuntimeError('Docker runner requires the checker container invocation')
    if args.probe and args.process_limits is None:
        raise RuntimeError('Runtime probe requires process mode')
    directory = args.request_directory or Path('/checker')
    request = json.loads((directory / 'request.json').read_text())
    import numpy as np

    # testing_util imports only these two EvalScope helpers; keep the complete
    # pinned checker unchanged without installing the evaluator in the sandbox.
    helpers = types.ModuleType('evalscope.utils.io_utils')
    helpers.current_time = datetime.datetime.now
    logger = types.ModuleType('evalscope.utils.logger')
    logger.get_logger = lambda *args, **kwargs: logging.getLogger('checker')
    sys.modules['evalscope.utils.io_utils'] = helpers
    sys.modules['evalscope.utils.logger'] = logger
    checker_path = directory / 'testing_util.py'
    identity = request.get('runtime_identity')
    if args.process_limits is not None:
        measured = {'python_version': platform.python_version(), 'numpy_version': np.__version__,
                    'checker_sha256': hashlib.sha256(checker_path.read_bytes()).hexdigest(),
                    'evalscope_commit': identity['evalscope_commit']}
        if measured != identity:
            raise RuntimeError('Checker runtime identity mismatch')
    spec = importlib.util.spec_from_file_location('pinned_testing_util', checker_path)
    checker = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(checker)
    if not callable(getattr(checker, 'run_test', None)):
        raise RuntimeError('Pinned checker has no run_test entry point')
    if args.probe:
        print(json.dumps({'runtime_identity': measured}, allow_nan=False))
        return
    destination = sys.stdout
    dump = json.dumps
    sys.stderr.write('CAPABILITY_CHECKER_EXECUTION_STARTED\n')
    sys.stderr.flush()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            results, metadata = checker.run_test(
                {'input_output': request['evaluation_sample']},
                test=request['code'], debug=False, timeout=request['test_timeout'],
            )
        results = [value.item() if isinstance(value, np.generic) else value for value in results]
        result = {'results': results, 'metadata': metadata}
    except (SystemExit, KeyboardInterrupt):
        result = {'results': [False], 'metadata': {'solution_exit': True}}
    except MemoryError:
        sys.stderr.write('CAPABILITY_CHECKER_MEMORY_LIMIT\n')
        sys.stderr.flush()
        raise
    except OSError as exc:
        if exc.errno == errno.EFBIG:
            sys.stderr.write('CAPABILITY_CHECKER_FILE_LIMIT\n')
            sys.stderr.flush()
        raise
    destination.write(dump(result, default=str, allow_nan=False) + '\n')
    destination.flush()


if __name__ == '__main__':
    main()
