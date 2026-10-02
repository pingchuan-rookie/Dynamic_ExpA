"""Offline dependency, official source, task count and environment checks (no LLM)."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def verify(benchmark, source=None, driver_python=None):
    if benchmark != 't2bench':
        raise ValueError('Only t2bench evaluation is supported')
    spec = json.loads((HERE / 'versions.json').read_text())[benchmark]
    assert sys.version_info[:2] == (3, 12), 'Evaluation environments require Python 3.12'
    for line in (HERE / f'{benchmark}-requirements.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        actual = importlib.metadata.version(name)
        assert actual == expected, f'{name}: expected {expected}, found {actual}'
    if driver_python is not None:
        import subprocess
        import ray

        probe = ('import json,sys,ray; '
                 'print(json.dumps({"python":list(sys.version_info[:3]),"ray":ray.__version__}))')
        driver = json.loads(subprocess.check_output([str(driver_python), '-c', probe], text=True))
        assert list(sys.version_info[:3]) == driver['python'], 'Ray driver/worker Python versions differ'
        assert ray.__version__ == driver['ray'], 'Ray driver/worker versions differ'
        print(f'[tau-ray] {benchmark}: driver/worker Python and Ray versions match', flush=True)
    print(f'[tau-dependencies] {benchmark}: exact dependency snapshot verified', flush=True)
    if source is None:
        return
    import subprocess
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from agent_system.environments.env_package.source_bundle import source_identity
    actual = source_identity(source)['commit']
    assert actual == spec['commit'], f'Unexpected upstream commit: {actual}'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / 'src'))
    rows = []
    os.environ.setdefault('TAU2_DATA_DIR', str(Path(__file__).resolve().parents[3] / 'data/t2bench/source/data'))
    import tau2
    assert Path(tau2.__file__).resolve().is_relative_to(source.resolve())
    from tau2.registry import registry
    for domain, count in [('retail', 114), ('airline', 50), ('telecom', 114)]:
        tasks = registry.get_tasks_loader(domain)('base')
        assert len(tasks) == len({t.id for t in tasks}) == count
        env = registry.get_env_constructor(domain)()
        task = next(t for t in tasks if t.evaluation_criteria.actions)
        state = task.initial_state
        env.set_state(state.initialization_data if state else None,
                      state.initialization_actions if state else None,
                      (state.message_history or []) if state else [])
        assert env.get_tools() and env.get_policy() and env.get_db_hash()
        before = env.get_db_hash()
        calls = 0
        for action in task.evaluation_criteria.actions:
            invoke = env.use_user_tool if action.requestor == 'user' else env.use_tool
            observation = invoke(action.name, **action.arguments)
            assert observation is not None
            calls += 1
        assert calls
        rows.append({'domain': domain, 'split': 'base', 'tasks': count,
                     'tools': len(env.get_tools()), 'reference_tool_calls': calls,
                     'state_changed': before != env.get_db_hash()})
    print(json.dumps({'benchmark': benchmark, 'commit': actual, 'offline_only': True, 'domains': rows}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=['t2bench'])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source', type=Path)
    group.add_argument('--dependencies-only', action='store_true')
    parser.add_argument('--driver-python', type=Path,
                        help='Verify Ray actor compatibility with the evaluation driver')
    args = parser.parse_args()
    verify(args.benchmark, args.source, args.driver_python)
