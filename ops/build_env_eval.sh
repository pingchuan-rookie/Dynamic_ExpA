#!/usr/bin/env bash
# Prepare the joint evaluator image locally; no registry push or job submission.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    printf '%s\n' 'Usage: bash ops/build_env_eval.sh' \
        'Build and check the joint environment evaluator image locally.' \
        'Overrides: BASE_IMAGE, IMAGE_TAG. Docker daemon and build network access are required.' \
        'Example: IMAGE_TAG=env-eval:local bash ops/build_env_eval.sh' \
        'SWE Docker endpoint, task images, assets and worker isolation are separate runtime prerequisites.'
    exit 0
fi
[[ $# == 0 ]] || { printf 'Unexpected arguments; use --help\n' >&2; exit 2; }
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base="${BASE_IMAGE:-pingchuan03/dynamic-expa-verl@sha256:c20cc65ccfdb3023866764033dbd1121c9cd357c3da88961e6f16e70e5681ffc}"
image="${IMAGE_TAG:-pingchuan03/dynamic-expa-verl:verl090-vllm024-cu130-20260920-env-eval}"
docker build -f "${root}/ops/Dockerfile.swebench" --target grader-runtime \
    --build-arg "BASE_IMAGE=${base}" -t "${image}" "${root}"
docker run --rm --network=none --entrypoint /bin/bash "${image}" -euc '
    for name in expa-verl webshop t2bench swebench; do
        test -x "/opt/venv-${name}/bin/python"
    done
    policy_ray="$(/opt/venv-expa-verl/bin/python -c "import ray; print(ray.__version__)")"
    /opt/venv-swebench/bin/python -c '\''import importlib.metadata as m, json, sys
import docker, ray, swebench.harness.run_evaluation
assert ray.__version__ == sys.argv[1]
source = json.loads(m.distribution("swebench").read_text("direct_url.json"))
assert source["vcs_info"]["commit_id"] == "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
print("SWE harness import, pinned commit and Ray compatibility: PASS")'\'' "$policy_ray"
'
printf 'Built and dependency-checked locally: %s\n' "${image}"
docker image inspect "${image}" --format '{{.Id}}'
printf 'Not published; Cluster Docker access, assets and isolation remain deployment prerequisites.\n'
