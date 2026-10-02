#!/usr/bin/env bash
# Public training entrypoint.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/training_config.sh" "$@"
