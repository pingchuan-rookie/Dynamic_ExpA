"""DIVE transport using the shared fail-closed native session lifecycle."""
from agent_system.environments.core.native_tool import NativeToolLocalEnvTool
from agent_system.environments.backends.dive.pool import DiveEnvPool


class DiveLocalEnvTool(NativeToolLocalEnvTool):
    protocol = "dive"
    environment = "dive"
    supports_gigpo = True
    tool_name = "dive_session"
    pool_class = DiveEnvPool
