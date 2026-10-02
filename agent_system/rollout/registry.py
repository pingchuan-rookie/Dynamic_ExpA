"""Lazy collector targets for the shared environment-step protocol."""

LOOP_TARGETS = {
    "environment_step_agent": "agent_system.rollout.environment_step_agent_loop.EnvironmentStepAgentLoop",
}


def loop_config(name):
    target = LOOP_TARGETS.get(name)
    return {"_target_": target} if target is not None else None
