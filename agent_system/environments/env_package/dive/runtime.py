"""Execute published DIVE tool calls through the pinned upstream registry."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import types

SOURCE_COMMIT = "106331d36ccdc34b81fd238615a36914a0d66330"
_REPAIRS = {
    "protparam_analysis": "ca1363d943ea4f3758d3d90e7893e0f855d0d066387b7d79b857995457c4e2bd",
    "protparam_aromaticity": "7e7b3de78b75c1308c89c442c613302b536dffbfefc274ff1092a7dfdc83c982",
    "protparam_isoelectric_point": "21b45fc5520412c2afa3d12e975b899327c7ff40a4b115f37341ed50b325f4e9",
    "protparam_molecular_weight": "4e3ecf5b6f8efec0922745f259d3d131ba375036271a9fc7f8256aa25b1470a0",
}


class DiveInfrastructureError(RuntimeError):
    """An attempt cannot be scored because its execution infrastructure failed."""


class DiveDeadline(BaseException):
    """Bypass upstream broad Exception handlers; an interrupted call is not reusable."""


def source_root(path=None):
    root = Path(path or os.environ.get("DIVE_REPO") or Path(__file__).resolve().parent / "source").resolve()
    if not (root / "tools/configs").is_dir() or not (root / "dive/tool_runner.py").is_file():
        raise DiveInfrastructureError(f"Missing DIVE source at {root}")
    return root


@lru_cache(maxsize=4)
def validate_source(root):
    from agent_system.environments.env_package.source_bundle import source_identity
    try:
        identity = source_identity(root)
    except (OSError, ValueError) as exc:
        raise DiveInfrastructureError(f"Invalid DIVE environment source bundle: {exc}") from None
    if identity["commit"] != SOURCE_COMMIT:
        raise DiveInfrastructureError("DIVE environment source version differs from the pin")


def load_upstream(path=None):
    root = source_root(path)
    validate_source(root)
    for name, expected in (("dive", root / "dive"), ("tools", root / "tools/tools")):
        module = sys.modules.get(name)
        if module is not None:
            module_path = Path(module.__file__).resolve().parent
            if module_path != expected:
                raise DiveInfrastructureError(f"Conflicting {name} package; use the isolated DIVE interpreter")
    for path in (root, root / "tools"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root


@contextmanager
def deadline(seconds):
    """Interrupt synchronous tool calls even when upstream catches Exception."""
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("DIVE deadline must be finite and positive")
    if threading.current_thread() is not threading.main_thread():
        raise DiveInfrastructureError("DIVE tool execution requires a synchronous actor main thread")
    previous = signal.getsignal(signal.SIGALRM)
    timer = signal.getitimer(signal.ITIMER_REAL)
    if timer[0] > 0:
        raise DiveInfrastructureError("Nested DIVE execution deadlines are unsupported")
    def expired(signum, frame):
        raise DiveDeadline()
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    except DiveDeadline:
        raise DiveInfrastructureError("DIVE execution deadline exceeded") from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def compatible_module(root, module_name):
    package = "tools.biological.bio_sequtils"
    if module_name.startswith(package + ".") and package not in sys.modules:
        # This pinned initializer imports two nonexistent, unregistered tools.
        # Load only its namespace; concrete registered tool modules remain upstream.
        path = root / "tools/tools/biological/bio_sequtils/__init__.py"
        if hashlib.sha256(path.read_bytes()).hexdigest() != "5fd78e3d487c34c46c9511cce45562a9d6bb29b5ff7b6f082187a49aaccd7b9b":
            raise DiveInfrastructureError("Unrecognized bio_sequtils package initializer")
        importlib.import_module("tools.biological")
        module = types.ModuleType(package)
        module.__file__ = str(path)
        module.__package__ = package
        module.__path__ = [str(path.parent)]
        sys.modules[package] = module
    stem = module_name.rsplit(".", 1)[-1]
    if stem not in _REPAIRS or module_name != f"tools.biological.protparam.{stem}":
        return importlib.import_module(module_name)
    path = root / "tools/tools/biological/protparam" / (stem + ".py")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _REPAIRS[stem]:
        raise DiveInfrastructureError(f"Unrecognized upstream source for {stem}")
    if module_name not in sys.modules:
        source = raw.decode().removesuffix("    @classmethod\n")
        module = types.ModuleType(module_name)
        module.__file__ = str(path)
        module.__package__ = module_name.rpartition(".")[0]
        exec(compile(source, str(path), "exec"), module.__dict__)
        sys.modules[module_name] = module
    return sys.modules[module_name]


@contextmanager
def browse_thinking_policy(tool_id):
    """Adapt only the pinned browse LLM call, never global requests or source files.

    Tool calls are synchronous on the actor main thread (enforced by deadline).
    Clone the upstream function with a private transport binding so its prompt,
    truncation and request defaults remain unchanged; restore even on failure.
    """
    if tool_id != "general.browse.browse":
        yield
        return
    project = Path(__file__).resolve().parents[4]
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from agent_system.utils.thinking import resolve_chat_template_kwargs
    from tools.general.utils import search_browse_utils as source

    provider = os.environ.get("BROWSE_LLM_PROVIDER", "openai_compatible")
    if provider not in ("openai_compatible", "trapi"):
        raise DiveInfrastructureError("Unsupported DIVE browse LLM provider")
    template_kwargs = resolve_chat_template_kwargs(model=source.BROWSE_LLM_MODEL)
    if provider != "trapi" and not template_kwargs:
        yield
        return
    original = source._query_with_llm
    transport = original.__globals__["requests"]
    bindings = dict(original.__globals__)

    if provider == "trapi":
        from .model_api.providers.trapi import normalize_trapi_request, resolve_trapi_config, trapi_client
        config = resolve_trapi_config(
            model=os.environ.get("BROWSE_LLM_MODEL"), base_url=os.environ.get("BROWSE_LLM_BASE_URL"),
        )
        # Upstream caches defaults at import time and requires a static key. Only
        # this cloned function sees the resolved role settings and synthetic key;
        # the bridge discards its headers and uses fresh Entra authentication.
        bindings.update(BROWSE_LLM_MODEL=config["model"], BROWSE_LLM_BASE_URL=config["base_url"],
                        BROWSE_LLM_API_KEY="TRAPI_DYNAMIC_AUTH")

        def post(url, *, json, headers=None, timeout=60):
            payload = normalize_trapi_request(config["model"], json)
            template = resolve_chat_template_kwargs(model=payload.get("model"))
            if template:
                payload["extra_body"] = {"chat_template_kwargs": template}
            with trapi_client(config["base_url"], timeout, max_retries=0) as client:
                response = client.chat.completions.create(**payload)
                body = response.model_dump(mode="json")
            return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)
    else:
        def post(url, *, json, **kwargs):
            payload = dict(json)
            payload["chat_template_kwargs"] = resolve_chat_template_kwargs(
                payload.get("chat_template_kwargs"), model=payload.get("model"),
            )
            return transport.post(url, json=payload, **kwargs)

    bindings["requests"] = types.SimpleNamespace(post=post)
    source._query_with_llm = types.FunctionType(
        original.__code__, bindings, original.__name__, original.__defaults__, original.__closure__,
    )
    try:
        yield
    finally:
        source._query_with_llm = original


class QuietLogger:
    def info(self, *args, **kwargs):
        pass
    warning = info
    error = info


class DiveToolRuntime:
    def __init__(self, repo=None, tool_timeout_s=60, sandbox_url=None):
        self.root = load_upstream(repo)
        from tools.core.registry import Registry
        self.registry = Registry.load(str(self.root / "tools/configs"))
        self.names = {r.id.rsplit(".", 1)[-1]: r.id for r in self.registry.list()}
        self.tool_timeout_s = float(tool_timeout_s)
        if not math.isfinite(self.tool_timeout_s) or self.tool_timeout_s <= 0:
            raise ValueError("tool_timeout_s must be finite and positive")
        self.sandbox_url = sandbox_url or os.environ.get("SANDBOX_FUSION_URL")

    def resolve(self, name):
        internal = "code_execution" if name == "jupyter_execute_code_cell" else name
        if name.startswith("semantic_scholar_"):
            candidate = name.removeprefix("semantic_scholar_")
            full = self.names.get(candidate, "")
            if full.startswith("academic.semantic_scholar."):
                internal = candidate
        if internal not in self.names:
            raise ValueError(f"Unregistered DIVE tool: {name}")
        return self.names[internal]

    def _create(self, tool_id):
        record = self.registry.get(tool_id)
        module_name, class_name = record.module.split(":")
        cls = getattr(compatible_module(self.root, module_name), class_name)
        return cls()

    def validate_tools(self, tools, *, imports=True):
        from jsonschema.validators import validator_for
        seen = set()
        for tool in tools:
            fn = tool["function"]
            name = fn["name"]
            if name in seen:
                raise ValueError(f"Duplicate DIVE tool: {name}")
            seen.add(name)
            schema = fn["parameters"]
            validator_for(schema).check_schema(schema)
            tool_id = self.resolve(name)
            if imports:
                self._create(tool_id)
        if not seen:
            raise ValueError("A DIVE task must expose at least one tool")
        return {"tools": len(seen), "source_commit": SOURCE_COMMIT}

    def execute(self, name, arguments):
        """Task schema validation belongs to the session, before compatibility mapping."""
        from tools.core.types import ExecutionContext
        tool_id = self.resolve(name)
        params = deepcopy(arguments)
        try:
            with deadline(self.tool_timeout_s):
                if name == "jupyter_execute_code_cell" or tool_id == "general.sandbox.code_execution":
                    if not self.sandbox_url:
                        raise DiveInfrastructureError("Code execution requires explicit SANDBOX_FUSION_URL")
                    from .sandbox import SandboxFusionClient
                    limit = params.get("timeout_seconds", self.tool_timeout_s)
                    if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
                        raise ValueError("timeout_seconds must be finite and positive")
                    limit = min(float(limit), self.tool_timeout_s)
                    code = params.get("cell_source") if name == "jupyter_execute_code_cell" else params.get("code")
                    client = SandboxFusionClient(base_url=self.sandbox_url, timeout=limit + 1, run_timeout=limit)
                    ok, result = client.run_jupyter([code], cell_timeout=limit, total_timeout=limit)
                    if not ok and result.get("status") == "Error":
                        raise DiveInfrastructureError("SandboxFusion execution infrastructure failed")
                    return str(result)
                tool = self._create(tool_id)
                auth = None
                env_key = "SEMANTIC_SCHOLAR_API_KEY" if tool_id.startswith("academic.semantic_scholar.") else "NCBI_API_KEY" if tool_id.startswith("biological.ncbi_entrez.") else None
                if env_key and os.environ.get(env_key):
                    auth = {"api_key": os.environ[env_key]}
                context = ExecutionContext(request_id="dive-tool", logger=QuietLogger(), auth=auth,
                                           timeout_ms=int(self.tool_timeout_s * 1000))
                # Upstream search/browse can swallow transport exceptions and print URLs
                # or credentials instead. Suppress those logs and invalidate such attempts.
                captured = io.StringIO()
                with redirect_stdout(captured), redirect_stderr(captured), browse_thinking_policy(tool_id):
                    result = tool.execute(context, params)
                diagnostic = captured.getvalue().lower()
                if tool_id.startswith("general.") and any(marker in diagnostic for marker in (
                        "attempt 1 error:", "attempt 2 error:", "attempt 3 error:",
                        "browse error with empty", "browse llm query failed:")):
                    raise DiveInfrastructureError("DIVE web service failed during tool execution")
                self._check_infrastructure(result)
                return self._format(result, tool_id)
        except DiveInfrastructureError:
            raise
        except (ImportError, OSError, TimeoutError) as exc:
            raise DiveInfrastructureError(f"DIVE tool infrastructure failed ({type(exc).__name__})") from None

    @staticmethod
    def _check_infrastructure(result):
        # Many upstream tools swallow HTTP exceptions. Only explicit error envelopes are
        # inspected; ordinary search results may legitimately contain these words.
        parsed = result
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except (ValueError, TypeError):
                return
        if not isinstance(parsed, dict) or not parsed.get("error"):
            return
        error = str(parsed["error"]).lower()
        markers = ("http 401", "http 403", "http 429", "http 500", "http 502", "http 503", "http 504",
                   "timed out", "timeout", "connection", "name resolution", "api key", "api_key",
                   "token not found", "unauthorized", "rate limit", "permission", "权限", "积分")
        import re
        http_failure = re.search(r"\b(?:401|403|429|5\d\d)\s+(?:client|server)\s+error\b", error)
        if http_failure or any(marker in error for marker in markers):
            raise DiveInfrastructureError("DIVE upstream returned an infrastructure error")

    @staticmethod
    def _format(result, tool_id):
        text = "" if result is None else result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        if not tool_id.startswith("general") and len(text) > 2000:
            import tiktoken
            tokens = tiktoken.get_encoding("cl100k_base").encode(text)
            cap = int(2000 * len(text) / len(tokens))
            if len(text) > cap:
                text = text[:cap] + "..."
        return text
