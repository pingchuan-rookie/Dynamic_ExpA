from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


_WRITE_LOCK = threading.Lock()


def log_event(event: str, **fields: Any) -> None:
    """Append one structured diagnostic event when logging is enabled."""
    path = os.getenv("ALFWORLD_SERVER_DIAG_PATH", "").strip()
    if not path:
        return

    payload = {
        "time": time.time(),
        "event": event,
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        **fields,
    }
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
