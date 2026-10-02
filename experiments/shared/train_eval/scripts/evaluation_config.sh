#!/usr/bin/env bash
# Benchmark configuration only: never execute training recipes or ablation scripts.
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPTS}/../../../.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venvs/expa-verl/bin/python}"
fail() { echo "[evaluate] $*" >&2; exit 2; }
# Environment sessions call the shared standalone inference API.
case "${1:-}" in
    tbench) fail 'Unsupported benchmark: tbench has been removed; use t2bench for tau evaluation.' ;;
    t2bench)
        exec "${PYTHON_BIN}" "${SCRIPTS}/prepare.py" tau-evaluation "$@"
        ;;
    swebench_verified)
        exec "${PYTHON_BIN}" "${SCRIPTS}/prepare.py" swebench-evaluation "${@:2}"
        ;;
esac
positions=(); overrides=(); options=()
explicit_checkpoint=0
export RUN_IS_DEBUG=0 RUN_IS_EVAL=1
while [ "$#" -gt 0 ]; do
    case "$1" in
        --api-url) export EVAL_API_URL="${2:?--api-url requires an endpoint}"; shift 2 ;;
        --served-model) export EVAL_SERVED_MODEL="${2:?--served-model requires a name}"; shift 2 ;;
        --model) export MODEL_NAME="${2:?--model requires a model name}"; shift 2 ;;
        --hardware) export HARDWARE_PROFILE="${2:?--hardware requires a profile}"; shift 2 ;;
        --debug) export RUN_IS_DEBUG=1; shift ;;
        --checkpoint) export EVAL_CHECKPOINT="${2:?--checkpoint requires a path}" EVAL_MODE=checkpoint; explicit_checkpoint=1; shift 2 ;;
        --projector-init) export EVAL_PROJECTOR_INIT="${2:?--projector-init requires a path}" EVAL_MODE=projector; shift 2 ;;
        --model-config) export EVAL_MODEL_CONFIG="${2:?--model-config requires a JSON path}"; shift 2 ;;
        --weights) export EVAL_MODE="${2:?--weights requires a mode}"; shift 2 ;;
        --check|--dry-run) options+=("$1"); shift ;;
        --experiment) fail 'Evaluation loads models, not training experiments. Use --checkpoint and its saved model_config.json.' ;;
        -h|--help)
            echo '  --api-url URL         Use an existing vLLM/Dyad endpoint'
            echo '  --served-model NAME    Model advertised by the endpoint'
            echo '  --model NAME           Model name used for weights and configuration selection'
            echo '  --hardware PROFILE     a6000, h100, h200, or a configured hardware variant'
            echo 'Usage: evaluate.sh <gsm8k|alfworld|codegym|webshop|dive> [baseline|grpo_react|gigpo|dyad-grpo|dyad-gigpo] [options] [Hydra overrides]'
            echo '       evaluate.sh t2bench [baseline|grpo_react|gigpo|dyad-grpo|dyad-gigpo] --help (standalone inference evaluation)'
            echo '  --projector-init PATH  Exact Alignment run (including timestamp) or projector.pt; bare run names use ckpt/lucia/alignment'
            echo '  --checkpoint PATH      Specific global_step_N checkpoint'
            echo '  --model-config PATH    Model configuration for a legacy checkpoint'
            echo '  --weights MODE         base, projector, checkpoint'
            echo '  --debug                Small evaluation batch/sample profile'
            echo '  --check                Validate selection without starting inference or loading models'
            echo '  --dry-run              Display the prepared command without starting inference'
            exit 0 ;;
        *=*) overrides+=("$1"); shift ;;
        -*) fail "Unknown option: $1" ;;
        *) positions+=("$1"); shift ;;
    esac
done
[ "${#positions[@]}" -ge 1 ] && [ "${#positions[@]}" -le 2 ] || fail 'Expected benchmark and optional algorithm'
[ -z "${EVAL_EXPERIMENT:-}" ] || fail 'EVAL_EXPERIMENT is retired; evaluation reads checkpoint model configuration'
benchmark="${positions[0]}"
algo="${positions[1]:-${EVAL_ALGO:-}}"
if [ "$algo" = baseline ] && [ "$explicit_checkpoint" != 1 ]; then
    fail 'baseline evaluation requires an explicit --checkpoint'
fi
if [ -z "${EVAL_MODE:-}" ]; then
    if [ -n "${EVAL_CHECKPOINT:-}" ]; then export EVAL_MODE=checkpoint
    elif [ -n "${EVAL_PROJECTOR_INIT:-}${DYAD_ENCODER_PROJECTOR_INIT:-}" ]; then export EVAL_MODE=projector
    else export EVAL_MODE=base; fi
fi
case "$EVAL_MODE" in
    alignment_dyad|agentic_rl_projector) export EVAL_MODE=projector ;;
    base|projector|checkpoint) ;;
    *) fail 'EVAL_MODE must be base, projector, or checkpoint' ;;
esac
if [ -z "$algo" ]; then
    if [ "$EVAL_MODE" = projector ]; then algo=dyad
    elif [ "$EVAL_MODE" = checkpoint ]; then algo=auto
    else algo=grpo_react; fi
fi
exec "${PYTHON_BIN}" "${SCRIPTS}/prepare.py" launch evaluation "$benchmark" "$algo" "${options[@]}" -- "${overrides[@]}"
