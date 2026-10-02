#!/usr/bin/env bash
# gigpo training entrypoint; configuration and execution are shared.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED="$(cd "${HERE}/../../shared/train_eval" && pwd)"
source "${SHARED}/scripts/method.sh"
status=0
select_method_args gigpo "$@" || status=$?
[ "$status" -ne 10 ] || exit 0
[ "$status" -eq 0 ] || exit "$status"
exec bash "${SHARED}/train.sh" "${METHOD_ARGS[@]}"
