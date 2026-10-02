"""Compatibility entry for the shared vLLM/Dyad inference service."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent_system.inference.server import main as serve_main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    for flag, value in (("--model", "capability_eval-policy"), ("--gpu-memory-utilization", "0.9")):
        if not any(arg == flag or arg.startswith(flag + "=") for arg in argv):
            argv += [flag, value]
    return serve_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
