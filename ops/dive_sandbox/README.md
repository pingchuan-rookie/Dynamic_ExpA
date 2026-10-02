# Local DIVE SandboxFusion

This derived image fixes notebook startup in the pinned official `server-20250609` image.
Its Python 3.10 `sandbox-runtime` contains IPython 9.3, which requires Python 3.11 and fails before kernel startup.
The upstream notebook driver retries that failure until the outer execution timeout.
The Dockerfile replaces only IPython with a SHA256-pinned Python-3.10-compatible wheel; the exact base digest fixes the remaining dependency closure.
It does not modify SandboxFusion or DIVE source, bypass their execution paths, or relax container permissions.

```bash
docker build --pull=false -t expa-dive-sandbox:server-20250609-ipython830 ops/dive_sandbox
docker run --rm --pull never \
  --name expa-dive-sandbox-local \
  --cpus 4 --memory 8g --pids-limit 512 \
  --cap-drop ALL --security-opt no-new-privileges \
  -p 127.0.0.1:18352:8080 \
  expa-dive-sandbox:server-20250609-ipython830
```

The build downloads the pinned public wheel and requires the official base image to be available.
Use an unoccupied port and container name; do not delete an existing service automatically.
No source, model cache, credentials, GPU or Docker socket is mounted into this sandbox.
The default bridge permits outbound networking; these settings are not a complete hostile-code security certification.

Set `SANDBOX_FUSION_URL=http://127.0.0.1:18352` only for clients that share the host network namespace.
Probe both `/run_code` and `/run_jupyter` through the DIVE overlay, then its `jupyter_execute_code_cell` mapping.
The pinned upstream client expects `cells_result`, but this actual server returns `cells` with structured `display` and `error` lists; an upstream `Finished` result with no cells is not a successful execution check.
Multiple cells in one request share notebook state; this does not establish state persistence between requests or change DIVE's single-cell request semantics.

The image also contains Python-version metadata inconsistencies for NetworkX 3.5 and tifffile 2025.6.1.
Their runtime behavior is not certified here, and `pip check` alone does not verify Python-version compatibility for every installed distribution.
This targeted fix and the elementary notebook probes do not certify arbitrary packages, all DIVE tools or the training matrix.
Environment integration and configuration are documented in the [DIVE runtime](../../agent_system/environments/env_package/dive/README.md).
