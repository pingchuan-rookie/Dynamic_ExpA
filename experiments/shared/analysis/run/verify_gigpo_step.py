#!/usr/bin/env python3
"""Verify real ALFWorld step-profile interactions, not legacy trajectory dumps."""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

from _env_checks import is_probe, lease_violations


def verify(directory: str) -> bool:
    from _env_checks import step_run_data, verify_environment_steps

    data = step_run_data(directory)
    if data["shared"]:
        return verify_environment_steps(directory, data)
    events = []
    for path in sorted(Path(directory).glob('alfworld_env_pool*.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():
                event = json.loads(line)
                event['_source'] = path.name
                events.append(event)
    pools = [e for e in events if e.get('event') == 'pool_init']
    real = [e for e in events if not is_probe(e.get('session_id'))]
    resets = [e for e in real if e.get('event') == 'env_reset']
    steps = [e for e in real if e.get('event') == 'env_step_bind']
    closes = [e for e in real if e.get('event') == 'env_close']
    key = lambda e: (e['_source'], e.get('session_id'))
    reset_counts = collections.Counter(map(key, resets))
    close_counts = collections.Counter(map(key, closes))
    step_counts = collections.Counter(map(key, steps))
    workers = {key(e): e.get('worker_idx') for e in resets}
    violations, active = lease_violations(pools, resets, closes)
    checks = {
        'healthy pools': bool(pools) and all(e.get('ok') for e in pools),
        'real environment decisions': bool(steps),
        'one reset and close per episode': bool(resets)
        and reset_counts == close_counts and all(n == 1 for n in reset_counts.values())
        and all(e.get('ok') for e in closes),
        'episode decisions within 1..50': set(step_counts) == set(reset_counts)
        and all(1 <= n <= 50 for n in step_counts.values()),
        'worker binding preserved': all(
            key(e) in workers and workers[key(e)] == e.get('worker_idx') for e in steps
        ),
        'all leases returned': violations == 0 and active == 0,
        'action and observation evidence': all(
            isinstance(e.get('action_in'), str) and isinstance(e.get('observation_out'), str)
            and bool(e['observation_out']) for e in steps
        ),
        'finite native rewards': all(
            isinstance(e.get('reward'), (int, float)) and math.isfinite(e['reward']) for e in steps
        ),
        'no environment aborts': not any(e.get('event') == 'env_abort' for e in real),
    }
    for label, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
    terminal = sum(bool(e.get('done')) for e in steps)
    rewarded = sum(float(e.get('reward', 0)) > 0 for e in steps)
    print(f'episodes={len(resets)} real_steps={len(steps)} terminal_steps={terminal} rewarded_steps={rewarded}')
    print('Native pool rewards are unscaled; GiGPO scales them by 10 before advantage computation.')
    print('Format validity, token masks, advantages and optimizer updates are not verified by pool logs.')
    print('Generated-token trajectory decoding is unavailable for this diagnostic format.')
    return all(checks.values())


if __name__ == '__main__':
    from decode_trajectories import save_report

    directory = sys.argv[1]
    with save_report('verify_gigpo_step', directory):
        ok = verify(directory)
    sys.exit(0 if ok else 1)
