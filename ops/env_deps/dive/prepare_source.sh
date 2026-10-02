#!/usr/bin/env bash
# Verify the environment source bundled with the project; never use reference checkouts.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
TARGET="${DIVE_REPO:-${ROOT}/agent_system/environments/env_package/dive/source}"
PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN:-python3}" - "${TARGET}" <<'CHECK'
import sys
from agent_system.environments.env_package.source_bundle import source_identity
from agent_system.environments.env_package.dive.runtime import SOURCE_COMMIT
if source_identity(sys.argv[1])["commit"] != SOURCE_COMMIT:
    raise SystemExit("Packaged DIVE source differs from the expected version")
CHECK
