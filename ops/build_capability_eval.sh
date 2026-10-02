#!/usr/bin/env bash
# Build locally; publishing the resulting image is a separate operation.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base="${BASE_IMAGE:-pingchuan03/dynamic-expa-verl@sha256:5d7d7a74efea7c98cd24f68e0958c34910c78b391c029186aa90747dd9331d82}"
image="${IMAGE_TAG:-pingchuan03/dynamic-expa-verl:verl090-vllm024-cu130-20260918-capability}"
checker="${CHECKER_IMAGE:-capability-eval-checker:local}"
if ! docker image inspect "${checker}" >/dev/null 2>&1; then
    docker build -f "${root}/experiments/capability_eval/checker.Dockerfile" \
        --build-arg "BASE_IMAGE=${base}" -t "${checker}" "${root}"
fi
checker_id="$(docker image inspect "${checker}" --format '{{.Id}}')"
# BuildKit FROM expects a named reference, not a bare local sha256 image ID.
pinned_checker="capability-eval-checker:build-${checker_id#sha256:}"
docker tag "${checker_id}" "${pinned_checker}"
docker build -f "${root}/ops/Dockerfile.capability-eval" \
    --build-arg "BASE_IMAGE=${base}" --build-arg "CHECKER_IMAGE=${pinned_checker}" \
    -t "${image}" "${root}"
# Ordinary outer container only: no daemon socket, privileged mode or network.
docker run --rm --network none \
    -e RUN_SITE=lucia -e PATH=/opt/checker-python/bin \
    -v "${root}:/workspace/dynamic-expa:ro" \
    --entrypoint /opt/venv-capability-eval/bin/python "${image}" -c '
import shutil
import sys
assert shutil.which("docker") is None
sys.path.insert(0, "/workspace/dynamic-expa/experiments/capability_eval")
from backend import preflight
result = preflight({"benchmark": "livecodebench_v6", "model": "readiness", "api_url": "http://127.0.0.1:8000/v1"})
assert result["execution_backend"] == "bounded_process", result
print(result)
'
printf 'Built and checked locally: %s\n' "${image}"
docker image inspect "${image}" --format '{{.Id}}'
printf 'Not pushed; Cluster jobs were not submitted.\n'
