#!/usr/bin/env bash
# Execute an inference evaluation command; no training argument builders or val_only.
set -euo pipefail
case "${1:-}" in
    --execute) shift; exec "$@" ;;
    *) echo 'Use experiments/shared/train_eval/evaluate.sh to resolve evaluation inputs.' >&2; exit 2 ;;
esac
