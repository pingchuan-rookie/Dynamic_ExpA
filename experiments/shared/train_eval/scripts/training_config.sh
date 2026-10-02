#!/usr/bin/env bash
# Select a training recipe and pass its configuration to shared preparation.
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPTS}/../../../.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venvs/expa-verl/bin/python}"
fail() { echo "[train] $*" >&2; exit 2; }
positions=(); overrides=(); options=(); experiment=train
export RUN_IS_DEBUG=0 RUN_IS_EVAL=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --model) export MODEL_NAME="${2:?--model requires a model name}"; shift 2 ;;
        --hardware) export HARDWARE_PROFILE="${2:?--hardware requires a profile}"; shift 2 ;;
        --debug) export RUN_IS_DEBUG=1; shift ;;
        --projector-init) export DYAD_ENCODER_PROJECTOR_INIT="${2:?--projector-init requires an exact Alignment run or file}"; shift 2 ;;
        --experiment) experiment="${2:?--experiment requires a name}"; shift 2 ;;
        --check|--dry-run) options+=("$1"); shift ;;
        -h|--help)
            echo '  --model NAME           Model name used for weights and configuration selection'
            echo '  --hardware PROFILE     a6000, h100, h200, or a configured hardware variant'
            echo 'Usage: train.sh <alfworld|codegym|webshop|dive> [dyad-grpo|dyad-gigpo|grpo_react|gigpo] [options] [Hydra overrides]'
            echo '  --projector-init PATH  Exact Alignment run (including timestamp) or projector.pt; bare run names use ckpt/lucia/alignment'
            echo '  --experiment NAME   train (default) or an ablation filename from ablation/*.yaml'
            echo '  --debug             Small micro-batch/step profile; preserve per-env sampling/update budget'
            echo '  --check             Validate configuration without starting Ray or loading models'
            echo '  --dry-run           Build and display the final command without starting Ray'
            exit 0 ;;
        *=*) overrides+=("$1"); shift ;;
        -*) fail "Unknown option: $1" ;;
        *) positions+=("$1"); shift ;;
    esac
done
[ "${#positions[@]}" -ge 1 ] && [ "${#positions[@]}" -le 2 ] || fail 'Expected benchmark and optional algorithm'
benchmark="${positions[0]}"; algo="${positions[1]:-dyad-grpo}"
case "$benchmark" in
    tbench) fail 'Unsupported benchmark: tbench has been removed.' ;;
    alfworld|codegym|webshop|dive) ;;
    *) fail "$benchmark is evaluation-only; use evaluate.sh. Training datasets: dive, codegym, alfworld, webshop" ;;
esac
selection="$("${PYTHON_BIN}" - "$SCRIPTS" "$algo" <<'PY'
import os, sys
sys.path.insert(0, sys.argv[1])
from prepare import resolve_algorithm
try:
    selection = resolve_algorithm(sys.argv[2], os.environ)
except ValueError as exc:
    sys.exit(str(exc))
print(selection.algo, selection.adv_estimator)
PY
)" || exit 2
read -r algo estimator <<< "$selection"
[ "$algo" != dyad ] || export DYAD_ADV_ESTIMATOR="$estimator"
experiment="${experiment#ablation/}"; experiment="${experiment%.sh}"; experiment="${experiment%.yaml}"
[ "$experiment" != main ] || experiment=train
case "$benchmark/$algo" in
    alfworld/dyad|alfworld/grpo_react|alfworld/gigpo|codegym/dyad|codegym/grpo_react|codegym/gigpo|webshop/dyad|webshop/grpo_react|webshop/gigpo|dive/dyad|dive/grpo_react|dive/gigpo) ;;
    *) fail "Unsupported benchmark/algorithm: $benchmark/$algo" ;;
esac
[ "$experiment" = train ] || [ "$algo" = dyad ] || fail 'Ablations require dyad'
recipe="$("${PYTHON_BIN}" - "$SCRIPTS" "$experiment" "$benchmark" "$algo" <<'PY'
import sys, os
sys.path.insert(0, sys.argv[1])
from prepare import experiment_recipes
name, benchmark, algo = sys.argv[2:]
if algo != 'dyad':
    sys.exit(0)
try:
    recipes = experiment_recipes()
except ValueError as exc:
    sys.exit(str(exc))
if name not in recipes or benchmark not in recipes[name]['envs']:
    sys.exit('Unsupported training experiment/benchmark: ' + name + '/' + benchmark)
if algo == 'dyad':
    values = dict(recipes['train']['defaults'])
    values.update(recipes[name].get('overrides', {}))
    values = {k: os.environ.get(k) or v for k, v in values.items()}
    if name == '2.8_policy_lm_only' and not os.environ.get('DYAD_ENCODER_PROJECTOR_INIT'):
        sys.exit('2.8 requires DYAD_ENCODER_PROJECTOR_INIT from Alignment')
    for k, v in values.items():
        print(k + '\t' + v)
PY
)" || exit 2
while IFS=$'\t' read -r key value; do
    [ -z "$key" ] || export "$key=$value"
done <<< "$recipe"
export TRAINING_EXPERIMENT="$experiment" MODEL_NAME="${MODEL_NAME:-Qwen2.5-0.5B-Instruct}"
exec "${PYTHON_BIN}" "${SCRIPTS}/prepare.py" launch training "$benchmark" "$algo" "${options[@]}" -- "${overrides[@]}"
