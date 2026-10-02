"""Entrypoint for the reference ALFWorld env server.

Equivalent to the submodule's `alfworld` console script, but the app is imported by module path
rather than by installed package name, so no pip install is needed -- PYTHONPATH pointing at
experiments/shared/train_eval/reference/alfworld is enough. start.sh does exactly that.
"""

import argparse

import uvicorn


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=36001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run("alfworld_server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    launch()
