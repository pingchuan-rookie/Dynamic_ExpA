"""JSONL process transport for the standalone WebShop environment."""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agent_system.environments.env_package.webshop.envs import FORMAT_VERSION, WebShopEnv

def serve():
    # Preserve a dedicated protocol fd and redirect OS stdout as well as Python
    # stdout. The JVM/native libraries can print without going through sys.stdout.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    backend = None
    try:
        for line in sys.stdin:
            request = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict) or "id" not in request:
                    raise ValueError("Each request must be an object with id and op")
                op = request["op"]
                if op == "init":
                    if backend is not None:
                        raise RuntimeError("Backend is already initialized")
                    backend = WebShopEnv(request.get("config", {}))
                    result = backend.health()
                elif op == "health" and backend is None:
                    result = {"ready": False, "protocol_version": FORMAT_VERSION, "active_sessions": 0}
                else:
                    if backend is None:
                        raise RuntimeError("init is required before this operation")
                    if op == "health":
                        result = backend.health()
                    elif op == "create":
                        result = backend.create(request["session_id"], request["task_id"], request["split"])
                    elif op == "step":
                        result = backend.step(request["session_id"], request["action"])
                    elif op == "close":
                        result = backend.close(request["session_id"])
                    elif op == "export_tasks":
                        result = backend.export_tasks(**{k: request[k] for k in
                            ("split", "offset", "limit", "output_path") if k in request})
                    else:
                        raise ValueError(f"Unknown operation: {op}")
                response = {"id": request["id"], "ok": True, "result": result}
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                response = {"id": request.get("id") if isinstance(request, dict) else None,
                            "ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
            protocol.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
            protocol.flush()
    finally:
        if backend is not None:
            backend.shutdown()
        protocol.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    serve()
