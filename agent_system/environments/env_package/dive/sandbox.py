"""Adapt the pinned SandboxFusion notebook response without altering upstream tools.

Import after ``load_upstream`` has established the isolated DIVE source paths.
"""
from tools.general.utils.code_sandbox_utils import SandboxFusionClient as UpstreamSandboxFusionClient, clip_text


def _clip_output(value, limit):
    if isinstance(value, str):
        return clip_text(value, limit)
    if isinstance(value, list):
        return [_clip_output(item, limit) for item in value]
    if isinstance(value, dict):
        return {key: _clip_output(item, limit) for key, item in value.items()}
    return value


class SandboxFusionClient(UpstreamSandboxFusionClient):
    """Keep upstream transport, but read actual ``cells`` and kernel error data."""

    def __init__(self, *, base_url, **kwargs):
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("SandboxFusion requires an explicit service URL")
        super().__init__(base_url=base_url, **kwargs)

    def run_jupyter(self, cells, kernel="python3", cell_timeout=None, total_timeout=None):
        if not isinstance(cells, list) or not cells or not all(isinstance(cell, str) for cell in cells):
            raise ValueError("Notebook execution requires a nonempty list of code strings")
        payload = {"cells": cells, "kernel": kernel,
                   "total_timeout": self.run_timeout * len(cells) if total_timeout is None else total_timeout}
        if cell_timeout is not None:
            payload["cell_timeout"] = cell_timeout
        try:
            data = self._post("/run_jupyter", payload)
        except Exception as exc:
            # Do not expose URLs or credentials embedded in transport exception text.
            return False, {"status": "Error", "cells": [],
                           "stderr": f"SandboxFusion notebook transport failed ({type(exc).__name__})"}
        try:
            if not isinstance(data, dict) or data.get("status") != "Success":
                raise ValueError("SandboxFusion notebook request failed")
            driver = data.get("driver")
            if (not isinstance(driver, dict) or driver.get("status") != "Finished"
                    or type(driver.get("return_code")) is not int or driver["return_code"] != 0):
                raise ValueError("SandboxFusion notebook driver failed")
            results = data.get("cells")
            if not isinstance(results, list) or len(results) != len(cells):
                raise ValueError("SandboxFusion notebook cell count mismatch")
            output = []
            for cell in results:
                if (not isinstance(cell, dict)
                        or not all(isinstance(cell.get(key), str) for key in ("stdout", "stderr"))
                        or not all(isinstance(cell.get(key), list)
                                   and all(isinstance(item, dict) for item in cell[key])
                                   for key in ("display", "error"))):
                    raise ValueError("SandboxFusion notebook cell payload is malformed")
                output.append({
                    "status": "Error" if cell["error"] else "Success",
                    **{key: _clip_output(cell[key], self.max_output_length)
                       for key in ("stdout", "stderr", "display", "error")},
                })
        except ValueError as exc:
            return False, {"status": "Error", "cells": [], "stderr": str(exc)}
        ok = all(not cell["error"] for cell in output)
        return ok, {"status": "Finished" if ok else "CellError", "cells": output}
