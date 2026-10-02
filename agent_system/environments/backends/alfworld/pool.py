# ALFWorld leases one Ray worker per trajectory; unknown actor state blocks new leases.
# BaseEnvPool owns startup, leases and cleanup; this class supplies environment hooks.
# game_idx follows the train+test mappings order returned by load_alfworld_games(),
# preserving tools_kwargs.create_kwargs.game. reset_spec is passed to worker.reset.
# reset/step return observation, reward, available_actions, done and step_count;
# reset also returns game, and close returns {closed: True}.

from __future__ import annotations

import json
import math
import os

import yaml

# The single-env worker (AlfworldEnvWorker plus its helpers _process_ob/_first/_build_single_game_env)
# lives in the import-light agent_system.environments.backends.alfworld.worker: the Ray actor process imports only that
# (never triggering verl/__init__.py -> torch), avoiding the raylet registration timeout storm when N
# actors start concurrently (worker_pool.cc:590 ... not registered within the timeout). The re-export
# keeps historical usage working.
from agent_system.environments.backends.alfworld.worker import AlfworldEnvWorker  # noqa: F401
from agent_system.environments.core.pool import TrajectoryEnvPool


# =========================================================================
# Game list (train + test, in mappings order) -- identical to agentenv_alfworld/env_wrapper.py
# =========================================================================
def load_alfworld_games(
    alfworld_data: str,
    train_mapping: str,
    test_mapping: str,
    unseen_mapping: str | None = None,
) -> list[str]:
    """Assemble the list of game.tw-pddl paths: train first (alfworld_mappings_train.json), then test
    (alfworld_mappings_test.json); optionally append unseen (alfworld_mappings_unseen.json, the
    valid_unseen OOD games).

    game_idx is the index into this list; the order matches the old HTTP server's self.games, so the
    game indices baked into the datasets stay valid.
    unseen is always appended at the end (its indices start after train+test), leaving the existing
    0-based train/test indices untouched.
    """
    data = os.path.expanduser(os.path.expandvars(alfworld_data))
    train_root = os.path.join(data, "json_2.1.1", "train")
    test_root = os.path.join(data, "json_2.1.1", "valid_seen")
    unseen_root = os.path.join(data, "json_2.1.1", "valid_unseen")

    sources = [(train_mapping, train_root), (test_mapping, test_root)]
    # Append only when unseen_mapping is passed explicitly and the file exists (backward compatible: off by
    # default = unchanged behavior).
    if unseen_mapping and os.path.exists(unseen_mapping):
        sources.append((unseen_mapping, unseen_root))

    games: list[str] = []
    for mapping_file, root in sources:
        with open(mapping_file, encoding="utf-8") as f:
            mappings = json.load(f)
        for m in mappings:
            games.append(os.path.join(root, str(m["task_type"]), str(m["task_id"]), "game.tw-pddl"))
    return games


class AlfworldEnvPool(TrajectoryEnvPool):
    """ALFWorld actors managed by BaseEnvPool.

    Automatic capacity follows the actual batch assigned to this agent worker,
    including validation repeats. Each live trajectory owns one isolated actor.
    Explicit pool sizes must cover that demand; resources are never silently reduced.
    """

    WORKER_CLS = AlfworldEnvWorker
    LOG_NAME = "alfworld_env_pool"
    RPC_TIMEOUT_DEFAULTS = {"reset": 120.0, "step": 60.0, "close": 30.0, "lease": 300.0}

    @classmethod
    def resolve_timeouts(cls, config: dict) -> dict[str, float]:
        values = {method: float(config.get(f"{method}_timeout_s", default))
                  for method, default in cls.RPC_TIMEOUT_DEFAULTS.items()}
        if any(not math.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError("ALFWorld timeouts must be finite and positive")
        return values

    def __init__(
        self,
        config_path: str,
        alfworld_data: str,
        train_mapping: str,
        test_mapping: str,
        pool_size: int | None,
        num_cpus_per_worker: float = 0.1,
        unseen_mapping: str | None = None,
        timeouts: dict | None = None,
    ):
        super().__init__(pool_size, num_cpus_per_worker)
        self._timeouts = self.resolve_timeouts(timeouts or {})
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.alfworld_data = os.path.expanduser(os.path.expandvars(alfworld_data))
        os.environ.setdefault("ALFWORLD_DATA", self.alfworld_data)
        self.games = load_alfworld_games(alfworld_data, train_mapping, test_mapping, unseen_mapping)

    def _rpc_timeout(self, method: str) -> float | None:
        return self._timeouts.get(method)

    def _lease_timeout(self) -> float:
        return self._timeouts["lease"]

    def _worker_init_args(self) -> tuple:
        return (self.config, self.alfworld_data, self.games)

    def _reset_log_extra(self, reset_spec: dict, result: dict) -> dict:
        return {
            "game": reset_spec.get("game_idx"),
            "obs_len": len(result.get("observation", "")),
            "n_available": len(result.get("available_actions", []) or []),
        }

    def _pool_init_log_extra(self) -> dict:
        return {"num_games": len(self.games)}
