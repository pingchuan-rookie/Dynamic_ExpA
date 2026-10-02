"""Explicit execution backend selection for the pinned capability checker."""
from __future__ import annotations

import importlib
import os
from pathlib import Path

from docker_checker import CheckerInfrastructureError


def integrated_runtime():
    return os.environ.get('RUN_SITE') == 'lucia' or os.environ.get('CAPABILITY_CONTAINER') == '1'


def _backend(config):
    if config is None and integrated_runtime():
        config = {'engine': 'process'}
    if not isinstance(config, dict):
        raise ValueError('LiveCodeBench requires an explicit checker configuration')
    engine = config.get('engine', 'docker')
    if engine not in ('docker', 'process'):
        raise ValueError('Checker engine must be docker or process')
    if engine == 'docker' and integrated_runtime():
        raise ValueError('Capability containers use the image-integrated process checker, not nested Docker')
    return importlib.import_module(f'{engine}_checker'), config


def validate_config(config):
    backend, config = _backend(config)
    return backend.validate_config(config)


def preflight(config):
    backend, config = _backend(config)
    return backend.preflight(config)


def score(config, code, evaluation_sample, checker_source, test_timeout=6):
    backend, config = _backend(config)
    return backend.score(config, code, evaluation_sample, Path(checker_source), test_timeout)


def execution_method(config):
    return 'hardened_docker' if validate_config(config)['engine'] == 'docker' else 'bounded_process'
