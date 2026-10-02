from __future__ import annotations

from typing import Any

from agent_system.environments.backends.alfworld.adapter import AlfworldAdapter
from agent_system.environments.core.adapter import BaseEnvAdapter
from agent_system.environments.backends.calc.adapter import CalcAdapter
from agent_system.environments.backends.codegym.adapter import CodeGymAdapter
from agent_system.environments.core.generic import GenericAgentGymAdapter
from agent_system.environments.backends.tau.adapter import TauAdapter
from agent_system.environments.backends.webshop.adapter import WebShopAdapter

_ADAPTER_REGISTRY: dict[str, type[BaseEnvAdapter]] = {}


def register_adapter(name: str, adapter_cls: type[BaseEnvAdapter]) -> None:
    _ADAPTER_REGISTRY[name.lower()] = adapter_cls


def build_adapter(env_type: str, config: dict[str, Any]) -> BaseEnvAdapter:
    env_type = env_type.lower()
    adapter_cls = _ADAPTER_REGISTRY.get(env_type)
    if adapter_cls is None:
        if not config.get("allow_generic_adapter", True):
            raise ValueError(f"Unknown Dyad HTTP env adapter: {env_type}")
        adapter_cls = GenericAgentGymAdapter
    return adapter_cls(config)


register_adapter("agentgym", GenericAgentGymAdapter)
register_adapter("generic", GenericAgentGymAdapter)
register_adapter("text", GenericAgentGymAdapter)
register_adapter("alfworld", AlfworldAdapter)
register_adapter("codegym", CodeGymAdapter)
register_adapter("gsm8k_calc", CalcAdapter)
register_adapter("calc", CalcAdapter)
register_adapter("calculator", CalcAdapter)
register_adapter("t2bench", TauAdapter)
register_adapter("webshop", WebShopAdapter)
