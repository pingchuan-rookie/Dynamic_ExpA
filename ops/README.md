# Project image builds

This directory contains project-maintained image definitions and build tools.
The upstream verl image definitions remain in [`../docker/`](../docker/).

- `Dockerfile.dyad-verl` and `build_verl.sh`: training and environment dependencies.
- `Dockerfile.dive`: refresh DIVE dependencies on an existing training image.
- [`dive_sandbox/`](dive_sandbox/README.md): the derived SandboxFusion service image.
- `Dockerfile.capability-eval` and `build_capability_eval.sh`: capability-evaluation dependencies.
- `Dockerfile.swebench` and `Dockerfile.swebench-flask-offline`: SWE-bench environments.
- `Dockerfile`, `Dockerfile.dyad-agentenv`, and `build.sh`: retained earlier environment builds.

Moving an image definition does not change existing image tags, running containers,
or submitted jobs. Follow each build entrypoint's documented context and inputs.
