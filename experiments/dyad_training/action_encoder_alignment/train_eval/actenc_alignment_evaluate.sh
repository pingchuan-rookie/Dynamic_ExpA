#!/usr/bin/env bash
# Explicit complete test scoring of one saved checkpoint; never runs training or upload.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${HERE}/../../../.." && pwd)"
REPO="${PROJECT}"
export PYTHON="${PYTHON:-${REPO}/.venvs/expa-verl/bin/python}"
export PYTHONPATH="${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" -m agent_system.policies.dyad.training.action_encoder_alignment.actenc_alignment_evaluate "$@"
