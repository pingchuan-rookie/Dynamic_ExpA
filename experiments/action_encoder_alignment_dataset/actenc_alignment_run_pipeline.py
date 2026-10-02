#!/usr/bin/env python3
"""Build the Alignment dataset end to end.

Thin wrapper over `experiments.action_encoder_alignment_dataset.actenc_alignment_generate_dataset`. The steps are the
pipeline of section 1.13 of
`dynamic_dyad_training_strategy_mcp_updated.md`, and they follow the file-update rule of section
1.12: each one reads only its upstream products, so changing `n_per_size` needs `--step case-plan`
onward and leaves the catalogues alone.

Usage (cwd = repository root):

    export ANTHROPIC_AUTH_TOKEN=...            # the strong LLM gateway credential
    .venvs/expa-verl/bin/python experiments/action_encoder_alignment_dataset/actenc_alignment_run_pipeline.py --all
    .venvs/expa-verl/bin/python experiments/action_encoder_alignment_dataset/actenc_alignment_run_pipeline.py --step
    dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.action_encoder_alignment_dataset.actenc_alignment_generate_dataset import STEPS, run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", action="append", choices=list(STEPS),
                        help="run one step; repeatable, order follows the pipeline")
    parser.add_argument("--all", action="store_true", help="run every step")
    args = parser.parse_args()
    if not args.step and not args.all:
        parser.error("pass --all or at least one --step")
    run(args.step or list(STEPS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
