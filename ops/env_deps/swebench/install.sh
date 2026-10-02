#!/usr/bin/env bash
# Explicit online preparation; never called by the offline evaluator.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source_dir="${SWEBENCH_REPO:-${root}/agent_system/environments/env_package/swebench/source}"
venv="${SWE_VENV_DIR:-${root}/.venvs/swebench}"
PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN:-python3}" - "${source_dir}" <<'CHECK'
import sys
from agent_system.environments.env_package.source_bundle import source_identity
if source_identity(sys.argv[1])["commit"] != "726c5461e2ef52d83cf1ea2107870a8bb3328d57":
    raise SystemExit("Packaged SWE-bench source differs from the expected version")
CHECK
if [[ ! -x "${venv}/bin/python" ]]; then
    [[ ! -e "${venv}" ]] || { printf 'Incomplete venv exists: %s\n' "${venv}" >&2; exit 2; }
    uv venv --relocatable --python 3.12 "${venv}"
fi
policy_python="${PYTHON_BIN:-${root}/.venvs/expa-verl/bin/python}"
ray_version="$("${policy_python}" -c 'import ray; print(ray.__version__)')"
uv pip install --python "${venv}/bin/python" "${source_dir}" "mini-swe-agent==2.4.6" "ray[default]==${ray_version}"
PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}" "${venv}/bin/python" -c \
    'import docker; from agent_system.environments.env_package.swebench.assets import verify_harness_identity; print(verify_harness_identity())'
