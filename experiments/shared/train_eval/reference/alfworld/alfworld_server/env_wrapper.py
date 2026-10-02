import os
import json
import threading
import time

from .single_env import SingleAlfredTWEnv
from .diagnostics import log_event
from .utils import load_config, process_ob


def configs_dir() -> str:
    """Directory holding base_config.yaml and mappings_{train,test,unseen}.json.

    Defaults to this package's own configs/. The upstream agentenv mappings are not usable here:
    upstream train has 2420 entries against our 3553, and upstream test is built from
    json_2.1.1/valid_train while we use valid_seen, so the two share no task_id at all. The copy in
    this package is the complete one; agent_system/environments/configs/ holds a byte-identical copy for
    the in-process training pool (check_alfworld_mappings.py enforces that).

    ALFWORLD_CONFIGS_DIR overrides it (e.g. configs mounted from blob on the cluster).
    """
    return os.environ.get("ALFWORLD_CONFIGS_DIR") or os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs"
    )


class ALFWorld_Wrapper:
    def __init__(self, **kwargs):
        # load data_path
        self.data_path = kwargs.get("data_path", None)
        if self.data_path is None:
            raise Exception("missing parameter data_path")
        os.environ["ALFWORLD_DATA"] = self.data_path

        # load config for alfworld benchmark
        self.config_path = kwargs.get("config_path", None)
        if self.config_path is None:
            raise Exception("missing parameter config_path")
        self.config = load_config(self.config_path)

        self._max_id = 0
        self.ls = []
        self.env = {}  # dict[id, env_item]
        self.env_init = {}  # dict[id, env_item]
        self.info = {}  # dict[id, env_info]
        self.games = []  # list[game_file]
        self._lock = threading.RLock()
        self._env_locks = {}
        # TextWorld touches process-global Gym and parser state during reset, step, and
        # close. HTTP sessions remain concurrent, but calls into that engine must
        # be serialized to prevent parser corruption.
        self._engine_lock = threading.Lock()
        self._created_sessions = 0
        self._closed_sessions = 0

        cfg_dir = configs_dir()

        train_games_root = os.path.join(
            os.environ["ALFWORLD_DATA"], "json_2.1.1", "train"
        )
        test_games_root = os.path.join(
            os.environ["ALFWORLD_DATA"], "json_2.1.1", "valid_seen"
        )

        train_mapping_file = os.path.join(cfg_dir, "mappings_train.json")
        test_mapping_file = os.path.join(cfg_dir, "mappings_test.json")

        with open(train_mapping_file, "r") as f:
            mappings = json.load(f)
            for mapping in mappings:
                self.games.append(
                    os.path.join(
                        train_games_root,
                        mapping["task_type"],
                        mapping["task_id"],
                        "game.tw-pddl",
                    )
                )

        with open(test_mapping_file, "r") as f:
            mappings = json.load(f)
            for mapping in mappings:
                self.games.append(
                    os.path.join(
                        test_games_root,
                        mapping["task_type"],
                        mapping["task_id"],
                        "game.tw-pddl",
                    )
                )

        # Append optional unseen games after train/valid_seen to preserve existing game indices.
        self.unseen_start_index = len(self.games)
        self.num_unseen_games = 0
        if os.environ.get("ALFWORLD_INCLUDE_UNSEEN", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            unseen_games_root = os.path.join(
                os.environ["ALFWORLD_DATA"], "json_2.1.1", "valid_unseen"
            )
            unseen_mapping_file = os.path.join(cfg_dir, "mappings_unseen.json")
            if os.path.exists(unseen_mapping_file):
                with open(unseen_mapping_file, "r") as f:
                    mappings = json.load(f)
                    for mapping in mappings:
                        self.games.append(
                            os.path.join(
                                unseen_games_root,
                                mapping["task_type"],
                                mapping["task_id"],
                                "game.tw-pddl",
                            )
                        )
                    self.num_unseen_games = len(mappings)

    def create(self):
        """Create a new session without selecting a game yet."""
        started = time.perf_counter()
        idx = None
        try:
            # TODO extend to other kinds of environments
            with self._lock:
                idx = self._max_id
                self._max_id += 1
                self._env_locks[idx] = threading.RLock()

            env = SingleAlfredTWEnv(self.config, train_eval="train")
            with self._lock:
                self.env[idx] = env
                self.info[idx] = {"done": False, "reward": 0, "deleted": False, "step_count": 0}
                self.ls.append(idx)
                self._created_sessions += 1
            payload = {"id": idx}
        except Exception as exc:
            payload = {"error": str(exc), "error_type": type(exc).__name__}
        finally:
            log_event(
                "env_create",
                env_id=idx,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
                active_sessions=self.active_session_count(),
                result=self._summarize_payload(payload),
            )
        return payload

    def __del__(self):
        for idx in list(getattr(self, "ls", [])):
            try:
                if not self.info.get(idx, {}).get("deleted", True):
                    self.close(idx)
            except Exception:
                pass

    def step(self, idx: int, action: str):
        """Execute one action while serializing operations for the same session."""
        started = time.perf_counter()
        payload = None
        try:
            with self._get_env_lock(idx):
                self._check_id(idx)
                with self._engine_lock:
                    ob, _, done, info = self.env_init[idx].step([action])
                ob = process_ob(ob[0])
                reward = float(info["won"][0])
                done = bool(done[0])
                available_actions = info.get("admissible_commands", [[]])[0]
                step_count = int(self.info[idx].get("step_count", 0)) + 1
                payload = {
                    "observation": ob,
                    "reward": reward,
                    "available_actions": available_actions,
                    "done": done,
                    "step_count": step_count,
                }
                self.info[idx].update(payload)
        except Exception as exc:
            payload = {"error": str(exc), "error_type": type(exc).__name__}
        finally:
            log_event(
                "env_step",
                env_id=idx,
                action=action,
                game=self.info.get(idx, {}).get("game"),
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
                active_sessions=self.active_session_count(),
                result=self._summarize_payload(payload),
            )
        return payload

    def reset(self, idx: int, game: int, world_type: str):
        """Reset a session to one specific game."""
        started = time.perf_counter()
        payload = None
        if world_type not in ["Text", "Embody", "Hybrid"]:
            payload = {"error": 'world_type must be one of "Text", "Embody" and "Hybrid"'}
        else:
            try:
                with self._get_env_lock(idx):
                    self._check_id(idx, True)
                    old_env = self.env_init.pop(idx, None)
                    if old_env is not None:
                        with self._engine_lock:
                            old_env.close()

                    self.env[idx].game_files = [self.games[game]]
                    self.env[idx].num_games = 1
                    with self._engine_lock:
                        self.env_init[idx] = self.env[idx].init_env(batch_size=1)
                        ob, info = self.env_init[idx].reset()
                    ob = "\n".join(ob[0].split("\n\n")[1:])
                    available_actions = info.get("admissible_commands", [[]])[0]
                    payload = {
                        "id": idx,
                        "observation": ob,
                        "available_actions": available_actions,
                        "task_type": "/".join(info["extra.gamefile"][0].split("/")[-3:-1]),
                    }
                    self.info[idx] = {
                        "world_type": world_type,
                        "game": game,
                        "observation": ob,
                        "available_actions": available_actions,
                        "done": False,
                        "reward": 0,
                        "deleted": False,
                        "step_count": 0,
                    }
            except (Exception, SystemExit) as exc:
                # Treat planner SystemExit as a load failure so one broken game cannot stop the server.
                payload = {"error": str(exc), "error_type": type(exc).__name__}
        log_event(
            "env_reset",
            env_id=idx,
            game=game,
            world_type=world_type,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
            active_sessions=self.active_session_count(),
            result=self._summarize_payload(payload),
        )
        return payload

    def close(self, idx: int):
        """Close and mark one session deleted."""
        started = time.perf_counter()
        payload = {"id": idx, "closed": False}
        try:
            with self._get_env_lock(idx):
                self._check_id(idx, True)
                env_init = self.env_init.pop(idx, None)
                if env_init is not None:
                    with self._engine_lock:
                        env_init.close()
                self.env.pop(idx, None)
                self.info.pop(idx, None)
                with self._lock:
                    self._env_locks.pop(idx, None)
                    if idx in self.ls:
                        self.ls.remove(idx)
                    self._closed_sessions += 1
                payload = {"id": idx, "closed": True}
        except Exception as exc:
            payload = {"id": idx, "closed": False, "error": str(exc), "error_type": type(exc).__name__}
        finally:
            log_event(
                "env_close",
                env_id=idx,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
                active_sessions=self.active_session_count(),
                result=self._summarize_payload(payload),
            )
        return payload

    @staticmethod
    def _summarize_payload(payload):
        if not isinstance(payload, dict):
            return {"value_type": type(payload).__name__}
        keys = ("id", "reward", "done", "step_count", "task_type", "closed", "error", "error_type")
        summary = {key: payload[key] for key in keys if key in payload}
        if "available_actions" in payload:
            summary["available_action_count"] = len(payload["available_actions"] or [])
        if "observation" in payload:
            summary["observation_chars"] = len(str(payload["observation"]))
        return summary

    def active_session_count(self):
        with self._lock:
            return sum(not state.get("deleted", False) for state in self.info.values())

    def stats(self):
        with self._lock:
            return {
                "active_sessions": self.active_session_count(),
                "total_created_sessions": self._created_sessions,
                "total_closed_sessions": self._closed_sessions,
                "initialized_sessions": len(self.env_init),
                "done_sessions": sum(
                    state.get("done", False) and not state.get("deleted", False)
                    for state in self.info.values()
                ),
                "total_games": len(self.games),
                "unseen_start_index": getattr(self, "unseen_start_index", len(self.games)),
                "num_unseen_games": getattr(self, "num_unseen_games", 0),
            }

    def _get_env_lock(self, idx: int):
        with self._lock:
            lock = self._env_locks.get(idx)
        if lock is None:
            raise NameError(f"The id {idx} is not valid.")
        return lock

    def get_observation(self, idx: int):
        try:
            with self._get_env_lock(idx):
                self._check_id(idx)
                return self.info[idx]["observation"]
        except Exception as e:
            return {"error": str(e)}

    def get_available_actions(self, idx: int):
        try:
            with self._get_env_lock(idx):
                self._check_id(idx)
                return self.info[idx]["available_actions"]
        except Exception as e:
            return {"error": str(e)}

    def get_detailed_info(self, idx: int):
        try:
            with self._get_env_lock(idx):
                self._check_id(idx)
                return dict(self.info[idx])
        except Exception as e:
            return {"error": str(e)}

    def _check_id(self, idx: int, is_reset: bool = False):
        if idx not in self.info:
            raise NameError(f"The id {idx} is not valid.")

        if self.info[idx]["deleted"]:
            raise NameError(f"The task with environment {idx} has been deleted.")

        if not is_reset and self.info[idx]["done"]:
            raise NameError(f"The task with environment {idx} has finished.")


# Preserve externally supplied ALFWORLD_DATA; use the local cache only when it is unset.
_alfworld_data = os.environ.get("ALFWORLD_DATA") or "~/.cache/alfworld"
_alfworld_data = os.path.expanduser(os.path.expandvars(_alfworld_data))
os.environ["ALFWORLD_DATA"] = _alfworld_data

server = ALFWorld_Wrapper(
    data_path=_alfworld_data,
    config_path=os.path.join(configs_dir(), "base_config.yaml"),
)