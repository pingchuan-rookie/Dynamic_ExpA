#!/usr/bin/env bash
# Method entrypoints select identity; shared configuration owns all other options.
select_method_args() {
    local family="${1:?method family required}" benchmark algorithm
    shift
    if [ "$#" -eq 0 ] || [ "$1" = --help ] || [ "$1" = -h ]; then
        printf 'Usage: train.sh <environment> [algorithm] [options] [Hydra overrides]\n'
        printf 'Method: %s; use shared/train_eval/train.sh --help for common options.\n' "$family"
        return 10
    fi
    benchmark="$1"
    shift
    case "$family" in
        dyad) algorithm=dyad-grpo ;;
        grpo) algorithm=grpo_react ;;
        gigpo) algorithm=gigpo ;;
        *) printf 'Unknown method family: %s\n' "$family" >&2; return 2 ;;
    esac
    if [ "$#" -gt 0 ] && [[ "$1" != -* && "$1" != *=* ]]; then
        case "$family/$1" in
            dyad/dyad-grpo|dyad/dyad-gigpo|grpo/grpo_react|gigpo/gigpo)
                algorithm="$1"; shift ;;
            *) printf 'Algorithm %s does not belong to %s training\n' "$1" "$family" >&2; return 2 ;;
        esac
    fi
    if [ "$family" != dyad ]; then
        local argument
        for argument in "$@"; do
            if [ "$argument" = --projector-init ]; then
                printf 'Projector initialization requires dyad training\n' >&2
                return 2
            fi
        done
        unset DYAD_ENCODER_PROJECTOR_INIT ALIGNMENT_PROJECTOR_INIT
    fi
    METHOD_ARGS=("$benchmark" "$algorithm" "$@")
}
