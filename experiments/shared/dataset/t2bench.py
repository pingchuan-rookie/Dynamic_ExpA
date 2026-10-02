#!/usr/bin/env python3
"""Export native tau2 task snapshots, defaulting to the 278-task base split."""
from utils.tau_snapshot import main


if __name__ == "__main__":
    raise SystemExit(main("t2bench"))
