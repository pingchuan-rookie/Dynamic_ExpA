#!/usr/bin/env bash
# Start the ALFWorld reference server.
# Use PORT, ALFWORLD_PYTHON, and ALFWORLD_INCLUDE_UNSEEN for explicit overrides.
set -euo pipefail

PORT="${PORT:-36001}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the repository via tracked pyproject.toml and agent_system/ markers.
REPO_ROOT="${SCRIPT_DIR}"
while [ "${REPO_ROOT}" != "/" ]; do
    if [ -f "${REPO_ROOT}/pyproject.toml" ] && [ -d "${REPO_ROOT}/agent_system" ]; then
        break
    fi
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done

# Interpreter precedence: explicit override, image environment, then project environment.
if [ -n "${ALFWORLD_PYTHON:-}" ]; then
    PYBIN="${ALFWORLD_PYTHON}"
elif [ -x "/opt/venv-alfworld/bin/python" ]; then
    PYBIN="/opt/venv-alfworld/bin/python"
elif [ -x "${REPO_ROOT}/.venvs/expa-verl/bin/python" ]; then
    PYBIN="${REPO_ROOT}/.venvs/expa-verl/bin/python"
else
    PYBIN="python3"
fi

echo "[INFO] Checking port ${PORT}..."
# An unused port makes lsof return nonzero.
PIDS=$(lsof -t -iTCP:${PORT} -sTCP:LISTEN 2>/dev/null || true)
if [ -n "${PIDS}" ]; then
    echo "[WARN] Port ${PORT} is occupied by PID(s): ${PIDS}"
    echo "[INFO] Killing process(es)..."
    kill ${PIDS} 2>/dev/null || true
    sleep 2
    PIDS_AFTER=$(lsof -t -iTCP:${PORT} -sTCP:LISTEN 2>/dev/null || true)
    if [ -n "${PIDS_AFTER}" ]; then
        echo "[WARN] Still alive, force killing: ${PIDS_AFTER}"
        kill -9 ${PIDS_AFTER} 2>/dev/null || true
    fi
    echo "[INFO] Port ${PORT} has been released."
else
    echo "[INFO] Port ${PORT} is free."
fi

# Resolve the reference server package through PYTHONPATH.
export PYTHONPATH="$(dirname "${SCRIPT_DIR}"):${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export ALFWORLD_SERVER_DIAG_PATH="${ALFWORLD_SERVER_DIAG_PATH:-${SCRIPT_DIR}/outputs/alfworld_server.jsonl}"

CFG_DIR="${ALFWORLD_CONFIGS_DIR:-${SCRIPT_DIR}/configs}"

# Generate unseen mappings when requested and absent.
if [ "${ALFWORLD_INCLUDE_UNSEEN:-}" = "1" ] && [ ! -f "${CFG_DIR}/mappings_unseen.json" ]; then
    echo "[INFO] ${CFG_DIR}/mappings_unseen.json 不存在，正在从 valid_unseen 生成..."
    "${PYBIN}" "$(dirname "${SCRIPT_DIR}")/alfworld_official_eval/make_unseen_mappings.py"
fi

echo "[INFO] python       : ${PYBIN}"
echo "[INFO] ALFWORLD_DATA: ${ALFWORLD_DATA}"
echo "[INFO] configs      : ${CFG_DIR}"
echo "[INFO] diagnostics  : ${ALFWORLD_SERVER_DIAG_PATH}"
if [ "${ALFWORLD_INCLUDE_UNSEEN:-}" = "1" ]; then
    echo "[INFO] ALFWORLD_INCLUDE_UNSEEN=1 -> valid_unseen（OOD，134 局）会追加到索引 3693 起"
fi
echo "[INFO] Starting ALFWorld server on port ${PORT}..."

exec "${PYBIN}" -m uvicorn alfworld_server:app --host 0.0.0.0 --port "${PORT}"
