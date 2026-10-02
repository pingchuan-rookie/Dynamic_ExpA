"""Self-contained ALFWorld HTTP env server (reference evaluation service).

Same wire protocol as upstream agentenv's alfworld server, but imports nothing from
external reference checkouts: the single-game env class, the config loader and the mappings all live here.
See README.md for what differs from upstream.
"""

from .server import app
from .launch import launch

__all__ = ["app", "launch"]
