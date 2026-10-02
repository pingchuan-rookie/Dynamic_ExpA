"""Prepare the pinned official source without overwriting an existing checkout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def prepare(benchmark: str, root: Path) -> Path:
    if benchmark != "t2bench":
        raise ValueError("Only t2bench evaluation is supported")
    import sys
    sys.path.insert(0, str(root))
    from agent_system.environments.env_package.source_bundle import source_identity
    target = root / "agent_system/environments/env_package/t2bench/source"
    expected = json.loads((HERE / "versions.json").read_text())[benchmark]["commit"]
    if source_identity(target)["commit"] != expected:
        raise ValueError("Packaged t2bench source differs from the declared version")
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=['t2bench'])
    parser.add_argument('--code-root', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.benchmark, args.code_root.resolve())
