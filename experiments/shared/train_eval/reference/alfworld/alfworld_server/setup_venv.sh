#!/usr/bin/env bash
# Optionally create an isolated ALFWorld server environment.
# start.sh can reuse the project environment; override VENV_DIR for separate deployments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_ROOT="${SCRIPT_DIR}"
while [ "${REPO_ROOT}" != "/" ]; do
    if [ -f "${REPO_ROOT}/pyproject.toml" ] && [ -d "${REPO_ROOT}/agent_system" ]; then
        break
    fi
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venvs/alfworld-server}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
ALFWORLD_VERSION="${ALFWORLD_VERSION:-0.4.2}"

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] 未找到 uv，先装：https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "[INFO] venv          : ${VENV_DIR} (python ${PYTHON_VERSION})"
echo "[INFO] alfworld      : ${ALFWORLD_VERSION}"
uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"

uv pip install --python "${VENV_DIR}/bin/python" \
    "alfworld==${ALFWORLD_VERSION}" fastapi uvicorn anyio pyyaml
# Keep opencv-python-headless as the sole cv2 provider.
uv pip uninstall --python "${VENV_DIR}/bin/python" opencv-python 2>/dev/null || true

export ALFWORLD_DATA="${ALFWORLD_DATA:-${HOME}/.cache/alfworld}"
if [ -d "${ALFWORLD_DATA}/json_2.1.1" ]; then
    echo "[INFO] ALFWORLD_DATA 已存在，跳过下载：${ALFWORLD_DATA}"
else
    echo "[INFO] 下载 ALFWorld 数据到 ${ALFWORLD_DATA}"
    "${VENV_DIR}/bin/alfworld-download"
fi

echo
echo "[INFO] 完成。起 server："
echo "         ALFWORLD_PYTHON=${VENV_DIR}/bin/python bash ${SCRIPT_DIR}/start.sh"
