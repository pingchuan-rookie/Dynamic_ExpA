"""Validate DIVE deployment without contacting tool or model services."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def verify_judge_dependencies() -> dict[str, str]:
    """Check that the pinned judge and authentication SDKs can be imported."""
    import importlib

    for name in ("httpx", "openai", "azure.identity"):
        importlib.import_module(name)
    return {name: importlib.metadata.version(name) for name in ("openai", "azure-identity")}


def verify(repo=None, driver_python=None, dataset=None):
    dependencies = verify_judge_dependencies()
    from agent_system.environments.env_package.dive.runtime import DiveToolRuntime, SOURCE_COMMIT
    runtime = DiveToolRuntime(repo)
    from agent_system.environments.env_package.source_bundle import source_identity
    actual = source_identity(runtime.root)["commit"]
    if actual != SOURCE_COMMIT:
        raise ValueError(f"DIVE source version mismatch: {actual}")
    if driver_python:
        import ray
        driver = json.loads(subprocess.check_output([driver_python, "-c",
            "import json,sys,ray;print(json.dumps([list(sys.version_info[:3]),ray.__version__]))"], text=True))
        if driver != [list(sys.version_info[:3]), ray.__version__]:
            raise ValueError("DIVE worker and driver Python/Ray versions differ")
    from agent_system.environments.backends.dive.selection import SELECTED_DOMAINS
    records = [record for record in runtime.registry.list()
               if record.id.split(".", 1)[0] in SELECTED_DOMAINS]
    failures = []
    for record in records:
        try:
            runtime._create(record.id)
        except Exception as exc:
            failures.append([record.id, type(exc).__name__])
    if failures:
        raise RuntimeError(f"DIVE tool import failures: {failures}")
    if dataset:
        unique = {}
        with Path(dataset).open() as stream:
            for line in stream:
                task = json.loads(line)
                if task["metadata"]["domain"] not in SELECTED_DOMAINS:
                    continue
                for tool in task["tools"]:
                    key = json.dumps(tool, sort_keys=True)
                    unique[key] = tool
        for tool in unique.values():
            runtime.validate_tools([tool], imports=False)
    print(json.dumps({"ok": True, "source_commit": actual, "tools": len(records), "domains": list(SELECTED_DOMAINS),
                      "python": sys.version.split()[0], "ray": importlib.metadata.version("ray"),
                      "judge_dependencies": dependencies, "network_checked": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo")
    parser.add_argument("--driver-python")
    parser.add_argument("--dataset")
    args = parser.parse_args()
    verify(args.repo, args.driver_python, args.dataset)
