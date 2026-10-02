"""Precompute the walkthrough (gt action) of every game and fill the .cache/agentic_rl/alfworld/walkthroughs cache.

Covers train + test(valid_seen) + unseen(valid_unseen), 3827 games in total (taken strictly from the
three mappings in agent_system/environments/configs, matching the game index contract of training/generation).

- Games already in the cache are skipped (the run is resumable); only uncached games run expert(planner) and are written out.
- Parallel (spawn process pool), the worker count is overridden with ALFWORLD_PRECOMPUTE_WORKERS (default 12).
- Cache path = .cache/agentic_rl/alfworld/walkthroughs/<data_split>/<task_type>/<task_id>.json (one json per game,
  walkthrough + reset merged, shared by the cache readers and workers below).

Usage:
    cd Dynamic_ExpA
    ALFWORLD_DATA=~/.cache/alfworld .venvs/expa-verl/bin/python experiments/shared/dataset/alfworld_precompute_gt.py
    # choose the worker count: ALFWORLD_PRECOMPUTE_WORKERS=8 ...
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


HERE = Path(__file__).resolve().parent                 # experiments/shared/dataset
DYAD_ROOT = HERE.parents[2]                            # Dynamic_ExpA
# The ALFWorld config/mappings sharing one source with the env pool at training time (after the repo
# reorganisation they live in agent_system/environments/configs, no longer in the old verl/local_env/configs).
CFG_DIR = DYAD_ROOT / "agent_system" / "environments" / "configs"
CACHE_DIR = os.environ.get(
    "ALFWORLD_WALKTHROUGH_CACHE_DIR", str(DYAD_ROOT / ".cache" / "agentic_rl" / "alfworld" / "walkthroughs")
)
EXPERT_TYPE = os.environ.get("ALFWORLD_WALKTHROUGH_EXPERT", "planner")
DATA = os.path.expanduser(os.path.expandvars(os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld")))
# Whether to precompute the reset cache too (initial observation + available_actions) -- this lets RL data generation start no environment at all.
# On by default (=1); ALFWORLD_PRECOMPUTE_RESET=0 computes only the walkthrough. reset needs an env (each game compiles TextWorld on the fly, slow).
RESET_TOO = os.environ.get("ALFWORLD_PRECOMPUTE_RESET", "1").strip().lower() not in ("0", "false", "no", "off")
CFG_PATH = str(CFG_DIR / "alfworld_base_config.yaml")

# logical split -> data directory name + mappings file
SPLITS = {
    "train": ("train", "alfworld_mappings_train.json"),
    "test": ("valid_seen", "alfworld_mappings_test.json"),
    "unseen": ("valid_unseen", "alfworld_mappings_unseen.json"),
}


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Read the YAML config and expand the environment variables and home directory inside it."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return expand_vars(config)


def expand_vars(obj: Any) -> Any:
    """Recursively expand ``~`` and environment variables inside config strings."""
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    if isinstance(obj, list):
        return [expand_vars(x) for x in obj]
    if isinstance(obj, dict):
        return {k: expand_vars(v) for k, v in obj.items()}
    return obj


def load_game_json(game_file: Path) -> Dict[str, Any]:
    """Read the JSON content of one game.tw-pddl."""
    with open(game_file, "r", encoding="utf-8") as f:
        return json.load(f)


def solve_walkthrough_with_expert(game_file: Path, expert_type: str = "handcoded") -> Optional[List[str]]:
    """
    Solve the walkthrough on the fly with the ALFWorld expert when game.tw-pddl has none.
    Most game.tw-pddl files after a standard download/generation already have a walkthrough and never reach here.
    """
    import random
    import textworld
    from textworld.envs import PddlEnv

    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler,
        AlfredExpert,
        AlfredExpertType,
        AlfredInfos,
    )

    # Ask TextWorld for the admissible commands and gamefile info the expert needs to plan/execute.
    request_infos = textworld.EnvInfos(admissible_commands=True, extras=["gamefile"])
    env = PddlEnv(request_infos)
    env = AlfredDemangler(env, shuffle=False)
    env = AlfredInfos(env)
    env = AlfredExpert(env, expert_type=expert_type)

    done = False
    steps = 0
    trajectory: List[str] = []

    try:
        env.load(str(game_file))
        game_state = env.reset()

        if env.expert_type == AlfredExpertType.PLANNER:
            # The planner expert returns the complete plan directly, no stepping needed.
            return list(game_state["extra.expert_plan"])

        while not done:
            # The handcoded expert gives the best action for the current step; after executing it we query the next one.
            expert_action = game_state["extra.expert_plan"][0]
            _ = random.choice(game_state.admissible_commands)

            game_state, _, done = env.step(expert_action)
            trajectory.append(str(expert_action))
            steps += 1

            if steps > 250:
                # Guard against an infinite loop caused by a broken environment or a stuck expert.
                return None

        return trajectory
    except Exception:
        return None


def _walkthrough_cache_key_parts(game_file: Path) -> Tuple[str, str, str]:
    """Parse (split, task_type, task_id) out of the game path
    .../json_2.1.1/<split>/<task_type>/<task_id>/game.tw-pddl, used as the stable key of the walkthrough cache."""
    task_id = game_file.parent.name
    task_type = game_file.parent.parent.name
    split = game_file.parent.parent.parent.name
    return split, task_type, task_id


def _sample_cache_path(cache_dir: Any, game_file: Path) -> Path:
    """Per-game merged cache file: <cache_dir>/<split>/<task_type>/<task_id>.json.
    One json per game: the walkthrough (gt actions + expert_type) and the reset (initial observation +
    available_actions) live in the same file, no longer split into expert / reset sub-directories."""
    split, task_type, task_id = _walkthrough_cache_key_parts(game_file)
    return Path(cache_dir) / split / task_type / f"{task_id}.json"


def _load_sample_cache(cache_dir: Any, game_file: Path) -> Optional[Dict[str, Any]]:
    """Read the raw dict of the per-game merged cache; returns None on miss/corruption."""
    if not cache_dir:
        return None
    path = _sample_cache_path(cache_dir, game_file)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        # A corrupted cache (e.g. a half-written file) counts as a miss; the caller recomputes and overwrites.
        return None


def _merge_save_sample_cache(cache_dir: Any, game_file: Path, updates: Dict[str, Any]) -> None:
    """Merge updates into the per-game cache (read-modify-write + atomic replace).

    walkthrough and reset write their own fields: existing content is read first and merged, so writing
    reset never overwrites an already stored walkthrough (and vice versa). One game is only ever processed
    sequentially by one worker, so no cross-process concurrent write to the same file exists."""
    if not cache_dir:
        return
    split, task_type, task_id = _walkthrough_cache_key_parts(game_file)
    path = _sample_cache_path(cache_dir, game_file)
    try:
        payload = _load_sample_cache(cache_dir, game_file) or {}
        payload.update(updates)
        # Constant keys (easier for humans to read / debug).
        payload["split"] = split
        payload["task_type"] = task_type
        payload["task_id"] = task_id
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)  # atomic replace, so nobody reads a half-written file
    except Exception:
        # A failed cache write (e.g. a read-only disk) must not affect the main flow; skip silently.
        pass


def load_cached_walkthrough(cache_dir: Any, game_file: Path, expert_type: str) -> Optional[List[str]]:
    """Return the gt action sequence directly on a cache hit, avoiding re-running the expert. Returns None on miss/corruption/expert mismatch."""
    data = _load_sample_cache(cache_dir, game_file)
    if not data:
        return None
    # The walkthrough depends on the expert: a cached expert different from the requested one counts as a miss and triggers a recompute with the right expert.
    cached_expert = data.get("expert_type")
    if cached_expert is not None and str(cached_expert) != str(expert_type):
        return None
    wt = data.get("walkthrough")
    if isinstance(wt, list) and len(wt) > 0:
        return [str(x) for x in wt]
    return None


def save_cached_walkthrough(cache_dir: Any, game_file: Path, expert_type: str, walkthrough: List[str]) -> None:
    """Merge a freshly computed walkthrough into the per-game cache (without overwriting an existing reset field)."""
    if not cache_dir or not walkthrough:
        return
    _merge_save_sample_cache(
        cache_dir,
        game_file,
        {"walkthrough": [str(x) for x in walkthrough], "expert_type": str(expert_type)},
    )


def load_cached_reset(cache_dir: Any, game_file: Path) -> Optional[Tuple[str, List[str]]]:
    """On a hit returns (raw_observation, available_actions), directly usable for the first RL prompt turn (no env needed)."""
    data = _load_sample_cache(cache_dir, game_file)
    if not data:
        return None
    obs = data.get("observation")
    acts = data.get("available_actions")
    if isinstance(obs, str) and isinstance(acts, list):
        return obs, [str(a) for a in acts]
    return None


def save_cached_reset(cache_dir: Any, game_file: Path, observation: str, available_actions: List[str]) -> None:
    """Merge the initial observation + available_actions obtained from env reset into the per-game cache (without overwriting an existing walkthrough)."""
    if not cache_dir:
        return
    _merge_save_sample_cache(
        cache_dir,
        game_file,
        {"observation": str(observation), "available_actions": [str(a) for a in available_actions]},
    )


def get_walkthrough(
    game_file: Path,
    *,
    solve_missing_walkthrough: bool,
    expert_type: str,
    cache_dir: Any = None,
) -> Optional[List[str]]:
    """Get the gt action sequence of one game; raw cache -> walkthrough shipped with the game -> run the expert now (result written back to the cache).

    game.tw-pddl is never written back (writing back pollutes the ALFWorld raw data; it was historically polluted by handcoded exploration steps).
    """
    # Highest priority: take the gt action straight from the persistent cache (default .cache/agentic_rl/alfworld/walkthroughs).
    # On a hit game.tw-pddl is never touched and no expert is started (the fastest path).
    cached = load_cached_walkthrough(cache_dir, game_file, expert_type)
    if cached is not None:
        return cached

    game_data = load_game_json(game_file)

    if game_data.get("solvable") is False:
        return None

    walkthrough = game_data.get("walkthrough")
    # Next, use the walkthrough shipped with the game file (usually empty in this dataset).
    if isinstance(walkthrough, list) and len(walkthrough) > 0:
        return [str(x) for x in walkthrough]

    if not solve_missing_walkthrough:
        # When automatic solving is disallowed, samples without a walkthrough are skipped by the caller.
        return None

    walkthrough = solve_walkthrough_with_expert(game_file, expert_type=expert_type)

    # On a successful recompute write the persistent cache so the next run hits it and avoids recomputing.
    if walkthrough:
        save_cached_walkthrough(cache_dir, game_file, expert_type, walkthrough)

    return walkthrough


def resolve_environment_cls(env_type: str):
    """
    ALFWorld releases differ slightly: upstream examples use
    alfworld.agents.environment.get_environment, while some installed wheels
    expose only the concrete environment classes.
    """
    try:
        # Recent/upstream ALFWorld usually exposes get_environment.
        from alfworld.agents.environment import get_environment

        return get_environment(env_type)
    except (ImportError, AttributeError):
        # Some installed packages expose only the concrete environment classes; map them here for compatibility.
        from alfworld.agents.environment import AlfredHybrid, AlfredThorEnv, AlfredTWEnv

        env_map = {
            "AlfredTWEnv": AlfredTWEnv,
            "AlfredThorEnv": AlfredThorEnv,
            "AlfredHybrid": AlfredHybrid,
        }
        if env_type not in env_map:
            raise ValueError(f"unsupported ALFWorld env type: {env_type}")
        return env_map[env_type]


def get_split_builder_and_games(config: Dict[str, Any], env_split: str):
    """Initialise the ALFWorld builder of the given split and collect its game file list."""
    env_type = config["env"]["type"]
    builder = resolve_environment_cls(env_type)(config, train_eval=env_split)
    game_files = [Path(p) for p in builder.game_files]
    return builder, game_files


def reset_single_game(builder: Any, game_file: Path) -> Tuple[str, List[str]]:
    """
    Force the builder to load exactly one game so the reset observation corresponds one-to-one with game_file.
    """
    builder.game_files = [str(game_file)]
    builder.num_games = 1

    env = None
    try:
        # batch_size=1 makes sure the reset result only corresponds to the current game_file.
        env = builder.init_env(batch_size=1)
        obs, info = env.reset()

        observation = obs[0] if isinstance(obs, (list, tuple)) else obs
        # info["admissible_commands"] is usually a two-dimensional list with a batch dimension.
        admissible = info.get("admissible_commands", [[]])
        available_actions = admissible[0] if len(admissible) > 0 else []

        return str(observation), [str(a) for a in available_actions]
    finally:
        if env is not None and hasattr(env, "close"):
            # Release the underlying environment resources even when reset or reading info failed.
            env.close()


def _game_path(data_split: str, task_type: str, task_id: str) -> str:
    return os.path.join(DATA, "json_2.1.1", data_split, task_type, task_id, "game.tw-pddl")


def _cache_path(game_file: str) -> Path:
    return _sample_cache_path(CACHE_DIR, Path(game_file))


def load_all_games():
    games = []
    for _split, (data_split, mf) in SPLITS.items():
        mpath = CFG_DIR / mf
        mappings = json.loads(mpath.read_text(encoding="utf-8"))
        if not isinstance(mappings, list) or not mappings:
            raise ValueError(f"Empty or invalid ALFWorld {_split} mapping: {mpath}")
        for m in mappings:
            games.append(_game_path(data_split, str(m["task_type"]), str(m["task_id"])))
    if not games:
        raise ValueError("No ALFWorld games found for GT precomputation")
    return games


# Each spawn worker owns its config and lazily initialized reset builder.
_CFG = None
_BUILDER = None


def _init_worker():
    global _CFG, _BUILDER
    # Direct script execution puts this directory first, where alfworld.py would
    # shadow the installed alfworld package needed by expert/reset imports.
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != str(HERE)]
    _CFG = load_yaml_config(CFG_PATH) if RESET_TOO else None
    _BUILDER = None


def _get_builder():
    """Lazily build and reuse this worker's builder (reset_single_game overrides game_files, so it works for any split)."""
    global _BUILDER
    if _BUILDER is None:
        _BUILDER, _ = get_split_builder_and_games(_CFG, "train")
    return _BUILDER


def _solve_one(game_file: str):
    try:
        wt = get_walkthrough(
            Path(game_file),
            solve_missing_walkthrough=True,
            expert_type=EXPERT_TYPE,
            cache_dir=CACHE_DIR,
        )
        # Also backfill the reset cache (needs an env), so later RL data generation starts no environment at all.
        if RESET_TOO and load_cached_reset(CACHE_DIR, Path(game_file)) is None:
            obs, acts = reset_single_game(_get_builder(), Path(game_file))
            save_cached_reset(CACHE_DIR, Path(game_file), obs, acts)
        return ("ok", game_file, len(wt) if wt else 0)
    except BaseException as exc:  # noqa: BLE001 - the planner may raise SystemExit
        return ("err", game_file, type(exc).__name__ + ": " + str(exc)[:60])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    workers = int(os.environ.get("ALFWORLD_PRECOMPUTE_WORKERS", "12"))
    games = load_all_games()
    total = len(games)

    # Resumable run: drop games that are already fully cached (the main process reads small json files, fast). With RESET_TOO on, both walkthrough and reset must be present to count as done.
    def _done(g: str) -> bool:
        game_file = Path(g)
        if load_cached_walkthrough(CACHE_DIR, game_file, EXPERT_TYPE) is None:
            return False
        return not RESET_TOO or load_cached_reset(CACHE_DIR, game_file) is not None

    todo = [g for g in games if not _done(g)]
    already = total - len(todo)

    print(f"[precompute] cache_dir={CACHE_DIR} expert={EXPERT_TYPE} reset_too={RESET_TOO}")
    print(f"[precompute] total={total}  already_cached={already}  todo={len(todo)}  workers={workers}")
    if not todo:
        print("[precompute] everything is cached, nothing to compute.")
        return

    t0 = time.time()
    ok = fail = 0
    failures = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_init_worker) as pool:
        for i, (tag, gf, info) in enumerate(pool.imap_unordered(_solve_one, todo, chunksize=2), 1):
            if tag == "ok" and info:
                ok += 1
            else:
                fail += 1
                failures.append((gf, info))
            if i % 200 == 0 or i == len(todo):
                el = time.time() - t0
                rate = i / el if el else 0
                eta = (len(todo) - i) / rate if rate else 0
                print(f"  {i}/{len(todo)} ok={ok} fail={fail} | {rate:.1f} games/s ETA {eta/60:.1f}min", flush=True)

    print(f"\n[precompute] done in {(time.time()-t0)/60:.1f}min: computed_ok={ok} failed={fail}")
    print(f"[precompute] cached now = {sum(1 for g in games if _cache_path(g).exists())}/{total}")
    if failures:
        print("[precompute] failure samples (first 10):")
        for gf, info in failures[:10]:
            print(f"   {info}  <- {gf}")


if __name__ == "__main__":
    main()
