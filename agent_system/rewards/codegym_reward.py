# Reward function for CodeGym (agent_env style: the reward comes entirely from env.step).
#
# Like ALFWorld, CodeGym uses the data_source reward path of the Dyad/agent loop: when a
# trajectory's output.reward_score is None (e.g. a degenerate/aborted trajectory that carries no
# turn_scores), the framework's reward manager calls this function by data_source. CodeGym's real
# reward is a terminal 0/1 that the env writes back into the tool metrics through the bridge shim,
# from where the agent loop aggregates it into extra_fields.
#
# Shape normalisation (bare value / [x] / tensor(x)) lives in agent_system/rewards/base_reward.py.

from agent_system.rewards.base_reward import BaseEnvReward


class CodeGymReward(BaseEnvReward):
    SUCCESS_KEYS = ("won", "success")
    CONTAINERS = ("info", "infos", "last_env_metrics")
    NUMERIC_KEYS = ("reward", "score", "env_reward", "total_reward")

    def compute_score(self, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
        """Return the CodeGym episode reward (terminal 0/1) from environment metrics.

        Reward source priority (env-only):
          1. sum(turn_scores)        -- sum of the per-turn env rewards (normally completed trajectory)
          2. won / success           -- env success flag -> 1.0 / 0.0
          3. reward / score / env_reward -- direct numeric reward
          4. 0.0                     -- no signal at all (degenerate trajectory)
        """
        if not isinstance(extra_info, dict):
            return 0.0

        turn_scores = extra_info.get("turn_scores")
        if turn_scores:
            try:
                return float(sum(float(s) for s in turn_scores))
            except (TypeError, ValueError):
                pass

        success = self.find_flag(extra_info, self.SUCCESS_KEYS)
        if success is not None:
            return 1.0 if success else 0.0

        # Env metrics may be nested; unlike ALFWorld every container is walked, not just the first.
        for name in self.CONTAINERS:
            sub = extra_info.get(name)
            if isinstance(sub, dict):
                success = self.find_flag(sub, self.SUCCESS_KEYS)
                if success is not None:
                    return 1.0 if success else 0.0
                for key in self.NUMERIC_KEYS:
                    if key in sub:
                        return self.to_float(sub[key])

        for key in self.NUMERIC_KEYS:
            if key in extra_info:
                return self.to_float(extra_info[key])

        return 0.0


compute_score = CodeGymReward().compute_score
