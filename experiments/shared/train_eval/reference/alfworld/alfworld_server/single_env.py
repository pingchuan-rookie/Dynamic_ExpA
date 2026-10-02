"""Single-game ALFWorld TextWorld env.

Subclasses the official AlfredTWEnv but skips the automatic game scan: game_files stays empty and
the concrete game is chosen at reset time by ALFWorld_Wrapper. Scanning all 8810 games on every
session would otherwise dominate create() latency.

Self-contained on purpose -- this file exists so the reference server needs nothing from
external reference checkouts. It mirrors agent_system/environments/env_package/alfworld/envs.py:_build_single_game_env,
which does the same thing for the in-process training pool.

alfworld 0.4.2 no longer re-exports AlfredTWEnv from alfworld.agents.environment, so it has to be
imported from the alfred_tw_env submodule.
"""

from __future__ import annotations

from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv


class SingleAlfredTWEnv(AlfredTWEnv):
    """One game_file per env instance; the file is assigned at reset, not at construction."""

    def __init__(self, config, train_eval: str = "train"):
        self.config = config
        self.train_eval = train_eval
        self.goal_desc_human_anns_prob = self.config["env"]["goal_desc_human_anns_prob"]
        self.get_game_logic()
        self.random_seed = 42

        self.game_files = []
        self.num_games = 0
