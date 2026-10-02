#!/usr/bin/env bash
# Projector training and val selection (including --debug); never scores final test.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${HERE}/../../../.." && pwd)"
REPO="${PROJECT}"
export PYTHON="${PYTHON:-${REPO}/.venvs/expa-verl/bin/python}"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" "${HERE}/scripts/actenc_alignment_prepare.py" "$@"
