# Copyright 2025 ExpA_sys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Standalone alfworld environment: reset, step, close and environment health checks."""

from __future__ import annotations

import os
from typing import Any, Optional



def _process_ob(ob: str) -> str:
    """Same as the legacy utils.process_ob: strip TextWorld's 'You arrive at loc ...' prefix."""
    ob = str(ob)
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2:]
    return ob


def _first(value: Any) -> Any:
    """Take the first element of a batch_size=1 return value."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _build_single_game_env(config: dict):
    """Build an AlfredTWEnv bound to a single game (skips the full 8810-game scan, cheap).

    Mirrors the legacy environment.SingleAlfredTWEnv: __init__ is overridden so games are not loaded
    automatically, the single game is picked at reset time instead.
    alfworld 0.4.2: AlfredTWEnv is not re-exported from alfworld.agents.environment, it has to be
    imported from the alfred_tw_env submodule.
    """
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos, AlfredTWEnv

    class _SingleGameTWEnv(AlfredTWEnv):
        def __init__(self, cfg: dict, train_eval: str = "train"):
            self.config = cfg
            self.train_eval = train_eval
            self.goal_desc_human_anns_prob = cfg["env"]["goal_desc_human_anns_prob"]
            self.get_game_logic()
            self.random_seed = 42
            self.game_files = []
            self.num_games = 0

        def init_env(self, batch_size):
            # Installed ALFWorld 0.4.2 has no use_expert switch: its dagger/train
            # branch always adds AlfredExpert. Match verl-agent's expert-free
            # policy environment explicitly, retaining the original wrappers,
            # randomization and method-specific episode limit.
            method = self.config["general"]["training_method"]
            if method not in {"dagger", "dqn"}:
                raise NotImplementedError(f"Unsupported ALFWorld training method: {method}")
            section = "rl" if method == "dqn" else "dagger"
            limit = self.config[section]["training"]["max_nb_steps_per_episode"]
            shuffle = self.train_eval == "train" and self.config["env"]["domain_randomization"]
            env_id = textworld.gym.register_games(
                self.game_files,
                textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"]),
                batch_size=batch_size,
                asynchronous=True,
                max_episode_steps=limit,
                wrappers=[AlfredDemangler(shuffle=shuffle), AlfredInfos],
            )
            return textworld.gym.make(env_id)

    return _SingleGameTWEnv(config)


# =========================================================================
# Single-env worker (plain Python; usable directly, or wrapped as an actor via ray.remote)
# =========================================================================
class AlfworldEnv:
    """Holds one ALFWorld TextWorld env, can be reset to different games repeatedly and stepped.

    A worker serves at most one trajectory at a time (lease semantics); TextWorld's process-global
    state is isolated by giving each worker its own env, and under Ray actors that is a separate
    process anyway, which naturally sidesteps the threading-lock serialisation the old server needed.
    """

    # The engine import is deferred to __init__, so whether it actually loaded is worth reporting.
    IMPORT_PROBES = ("torch", "verl", "alfworld")

    def __init__(self, config: dict, alfworld_data: str, games: list[str]):
        os.environ["ALFWORLD_DATA"] = os.path.expanduser(os.path.expandvars(alfworld_data))
        self.config = config
        self.games = games
        self._base = _build_single_game_env(config)
        self._env = None
        self._step_count = 0
        self._game_idx: Optional[int] = None

    def reset(self, game_idx: int, world_type: str = "Text") -> dict:
        if world_type != "Text":
            raise ValueError(f'world_type must be "Text", got {world_type!r}')
        if not (0 <= int(game_idx) < len(self.games)):
            raise IndexError(f"game_idx {game_idx} out of range (0..{len(self.games) - 1})")
        game_idx = int(game_idx)
        game_file = self.games[game_idx]

        self._base.game_files = [game_file]
        self._base.num_games = 1
        if self._env is not None:
            # Failed cleanup leaves unknown engine state; let the pool discard this actor.
            self._env.close()
        self._env = self._base.init_env(batch_size=1)
        obs, info = self._env.reset()
        self._step_count = 0
        self._game_idx = game_idx

        ob = _process_ob(_first(obs))
        available = list(info.get("admissible_commands", [[]])[0]) if info.get("admissible_commands") else []
        return {
            "observation": ob,
            "reward": 0.0,
            "won": False,
            "available_actions": available,
            "done": False,
            "step_count": 0,
            "game": game_idx,
        }

    def step(self, action: str) -> dict:
        if self._env is None:
            raise RuntimeError("step called before reset")
        obs, score, done, info = self._env.step([str(action)])
        self._step_count += 1
        ob = _process_ob(_first(obs))
        # ALFWorld's success signal is info["won"], and reward is derived from it (success = 1.0).
        # won is returned separately so callers can track success rate: reward gets reshaped by
        # reward_mode, won never does.
        won = bool(_first(info.get("won"))) if info.get("won") is not None else False
        reward = 1.0 if won else 0.0
        d = bool(_first(done))
        available = list(info.get("admissible_commands", [[]])[0]) if info.get("admissible_commands") else []
        return {
            "observation": ob,
            "reward": reward,
            "won": won,
            "available_actions": available,
            "done": d,
            "step_count": self._step_count,
        }

    def close(self) -> dict:
        if self._env is not None:
            # Failed cleanup leaves unknown engine state; let the pool discard this actor.
            self._env.close()
            self._env = None
        self._game_idx = None
        return {"closed": True}

    def health_check(self) -> dict:
        """Round-trip self-check create->reset(game0)->step(look)->close; returns ok / error."""
        try:
            r = self.reset(0, "Text")
            s = self.step("look")
            self.close()
            return {
                "ok": True,
                "n_available": len(r.get("available_actions", [])),
                "step_obs_len": len(s.get("observation", "")),
            }
        except Exception as exc:
            return {"ok": False, "error": repr(exc)}
