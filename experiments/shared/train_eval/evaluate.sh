#!/usr/bin/env bash
# Public evaluation entrypoint.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/evaluation_config.sh" "$@"
