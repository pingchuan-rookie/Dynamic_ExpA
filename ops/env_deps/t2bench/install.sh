#!/usr/bin/env bash
# Install dependencies only: no editable package or host interpreter in the image.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
driver_python="${DYAD_PYTHON_BIN:-/opt/venv-expa-verl/bin/python}"
ray_version="$("${driver_python}" -c 'import ray; print(ray.__version__)')"
venv="${T2BENCH_VENV:-/opt/venv-t2bench}"
uv venv --relocatable --python 3.12 "${venv}"
uv pip sync --python "${venv}/bin/python" "${HERE}/t2bench-requirements.txt"
uv pip install --python "${venv}/bin/python" "ray==${ray_version}"
uv pip check --python "${venv}/bin/python"
"${venv}/bin/python" "${HERE}/verify.py" t2bench --dependencies-only --driver-python "${driver_python}"
