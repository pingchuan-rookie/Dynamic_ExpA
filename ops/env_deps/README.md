# Environment dependency preparation

This directory contains dependency installation, pinned source versions and deployment checks for DIVE, t2bench and SWE-bench.
Run the installation scripts from the repository root. Environment implementations are in [env_package](../../agent_system/environments/env_package/README.md).
Installation scripts are separate from environment interaction. Missing dependencies never trigger a substitute dataset, model or environment.

- `dive/`: `prepare_source.sh` verifies the bundled source; `install.sh` installs isolated dependencies and `verify.py` checks available tools.
- `t2bench/`: `prepare_source.py` verifies the bundled source; `install.sh` and `verify.py` prepare and check dependencies and tasks.
- `swebench/`: `install.sh` installs the pinned official evaluator and mini-swe-agent.

Run each entrypoint from the repository root using its documented environment variables and command-line arguments.
Docker builds use the version declarations and dependency files here. Runtime source ships with `env_package`; neither building nor running requires `repos/`.
