"""Ray worker binding for the standalone codegym environment."""
from agent_system.environments.core.worker import BaseEnvWorker
from agent_system.environments.env_package.codegym.envs import CodeGymEnv

from agent_system.environments.env_package.codegym.envs import (
    ACTION_RPC_HEADROOM_S, ACTION_TIMEOUT_DEFAULT_S, parse_codegym_env_str, source_compatibility,
)


class CodeGymEnvWorker(CodeGymEnv, BaseEnvWorker):
    """Expose environment operations through the shared worker protocol."""
