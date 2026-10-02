#!/usr/bin/env bash
# Install only the isolated tool environment, not the upstream synthesis package.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
DRIVER="${DYAD_PYTHON_BIN:-${ROOT}/.venvs/expa-verl/bin/python}"
TARGET="${DIVE_VENV:-${ROOT}/.venvs/dive}"
"${DRIVER}" - "${TARGET}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).expanduser().resolve()
if root == Path(sys.prefix).resolve() or root in Path(sys.prefix).resolve().parents:
    raise SystemExit("Refusing to sync the driver environment or its parent")
if root.exists():
    marker = root / ".dyad-dive-managed"
    if not (root / "pyvenv.cfg").is_file() or not marker.is_file() or marker.read_text() != "dyad-dive-v1\n":
        raise SystemExit("Existing DIVE_VENV is not a managed DIVE environment; choose a new destination")
PY
if [ ! -x "${TARGET}/bin/python" ]; then
    uv venv --python "${DRIVER}" "${TARGET}"
    printf 'dyad-dive-v1\n' > "${TARGET}/.dyad-dive-managed"
fi
DRIVER_PYTHON="$("${DRIVER}" -c 'import sys; print(sys.version_info[:3])')"
TARGET_PYTHON="$("${TARGET}/bin/python" -c 'import sys; print(sys.version_info[:3])')"
if [ "${DRIVER_PYTHON}" != "${TARGET_PYTHON}" ]; then
    printf '%s\n' 'Existing DIVE environment has another Python version; refusing to sync it.' >&2
    exit 1
fi
uv pip sync --python "${TARGET}/bin/python" "${HERE}/requirements.lock.txt"
RAY_VERSION="$("${DRIVER}" -c 'import ray; print(ray.__version__)')"
uv pip install --python "${TARGET}/bin/python" "ray==${RAY_VERSION}"
uv pip check --python "${TARGET}/bin/python"
PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${TARGET}/bin/python" -m ops.env_deps.dive.verify --driver-python "${DRIVER}"
