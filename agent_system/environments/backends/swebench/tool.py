"""Evaluation-only native SWE-bench session transport."""
from agent_system.environments.core.native_tool import NativeToolLocalEnvTool
from agent_system.environments.backends.swebench.pool import SwebenchEnvPool


class SwebenchLocalEnvTool(NativeToolLocalEnvTool):
    environment = "swebench_verified"
    tool_name = "swebench_session"
    pool_class = SwebenchEnvPool
