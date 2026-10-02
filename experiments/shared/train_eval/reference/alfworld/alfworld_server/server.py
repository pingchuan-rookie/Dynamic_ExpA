import os
import time

import anyio

from fastapi import FastAPI, HTTPException, Request

from .diagnostics import log_event
from .env_wrapper import server
from .model import *

app = FastAPI()


@app.on_event("startup")
async def configure_threadpool():
    """Allow many blocking TextWorld sessions to run over one HTTP server."""
    limit = int(os.getenv("ALFWORLD_SERVER_THREADPOOL_LIMIT", "256"))
    anyio.to_thread.current_default_thread_limiter().total_tokens = limit
    log_event("threadpool_configured", total_tokens=limit)


def _return_or_raise(payload):
    if isinstance(payload, dict) and "error" in payload:
        raise HTTPException(status_code=500, detail=payload)
    return payload


@app.middleware("http")
async def log_request(request: Request, call_next):
    started = time.perf_counter()
    status_code = 500
    error = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        log_event(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
            error=error,
            active_sessions=server.active_session_count(),
        )


@app.get("/")
def hello():
    return "This is environment AlfWorld."


# These routes are intentionally synchronous: FastAPI runs blocking TextWorld
# operations in its thread pool instead of blocking Uvicorn's event loop.
@app.post("/create")
def create():
    return _return_or_raise(server.create())


@app.post("/step")
def step(body: StepRequestBody):
    return _return_or_raise(server.step(body.id, body.action))


@app.post("/reset")
def reset(body: ResetRequestBody):
    return _return_or_raise(server.reset(body.id, body.game, body.world_type))


@app.post("/close")
def close(body: EnvRequestBody):
    return _return_or_raise(server.close(body.id))


@app.get("/stats")
def stats():
    return server.stats()


@app.get("/available_actions")
def get_available_actions(id: int):
    return server.get_available_actions(id)


@app.get("/observation")
def get_observation(id: int):
    return server.get_observation(id)


@app.get("/detail")
def get_detailed_info(id: int):
    return server.get_detailed_info(id)
