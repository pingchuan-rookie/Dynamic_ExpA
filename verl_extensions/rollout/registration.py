"""Project rollout registrations attached to the verl v0.9.0 registries.

Keep targets lazy: Ray workers must resolve the same registrations independently,
without importing vLLM while the framework packages are still initializing.
"""


def rollout_targets() -> dict[tuple[str, str], str]:
    """Return project rollout targets for the upstream string-based registry."""
    return {
        ("dyadvllm", "async"): "agent_system.policies.dyad.rollout.dyad_vllm_rollout.ExpavLLMAsyncRollout",
    }


def load_dyad_replica():
    """Load the replica only when the framework requests this backend."""
    from agent_system.policies.dyad.rollout.dyad_vllm_async_server import ExpavLLMReplica

    return ExpavLLMReplica


def register_replicas(registry: type) -> None:
    """Register project loaders through the upstream public replica API."""
    registry.register("dyadvllm", load_dyad_replica)
