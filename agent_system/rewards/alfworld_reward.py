# Reward function for ALFWorld (dispatched from verl/utils/reward_score/__init__.py by data_source).
#
# Does not compare against a GT action -- it only looks at whether the env reported success/won,
# and returns 1.0 on success, 0.0 otherwise.
# Shape normalisation (bare bool / [True] / tensor(True)) lives in agent_system/rewards/base_reward.py.

from agent_system.rewards.base_reward import BaseEnvReward


class AlfworldReward(BaseEnvReward):
    SUCCESS_KEYS = ("success", "won")
    # "won": [True] is the field ALFWorld usually emits; the env payload may instead sit nested
    # under info/infos.
    CONTAINERS = ("info", "infos")

    def compute_score(self, solution_str, ground_truth=None, extra_info=None, **kwargs):
        if extra_info is None:
            return 0.0

        success = self.find_flag(extra_info, self.SUCCESS_KEYS)
        if success is None:
            # Only the first container that is actually a dict is consulted; if it holds neither key
            # the result is "no signal" rather than falling through to the next one.
            for name in self.CONTAINERS:
                if name in extra_info and isinstance(extra_info[name], dict):
                    success = self.find_flag(extra_info[name], self.SUCCESS_KEYS)
                    break

        if success is None:
            return 0.0
        return 1.0 if success else 0.0


compute_score = AlfworldReward().compute_score
