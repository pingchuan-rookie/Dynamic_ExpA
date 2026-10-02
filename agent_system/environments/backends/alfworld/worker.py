"""Ray worker binding for the standalone alfworld environment."""
from agent_system.environments.core.worker import BaseEnvWorker
from agent_system.environments.env_package.alfworld.envs import AlfworldEnv


class AlfworldEnvWorker(AlfworldEnv, BaseEnvWorker):
    """Expose environment operations through the shared worker protocol."""
