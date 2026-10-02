"""Generate mappings_unseen.json for ALFWorld valid_unseen games.

Keep games whose game.tw-pddl solvable flag is true, matching the official
AlfredTWEnv filter. Start item ids after the train and test mappings so they
match the server's appended game order. Write to the reference server configs/;
ALFWORLD_CONFIGS_DIR overrides that directory.

Usage (does not require importing ALFWorld):
    ALFWORLD_DATA=~/.cache/alfworld python make_unseen_mappings.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent


def find_repo_root() -> Path:
    """Find the repository by its pyproject.toml file and agent_system/ directory."""
    for d in HERE.parents:
        if (d / "pyproject.toml").is_file() and (d / "agent_system").is_dir():
            return d
    raise RuntimeError(
        f"从 {HERE} 向上找不到 Dynamic_ExpA 根目录（标记：同级存在 pyproject.toml 和 agent_system/）。"
        "请用 ALFWORLD_CONFIGS_DIR 指向 server 的 configs/ 目录。"
    )


def _configs_dir() -> Path:
    override = os.environ.get("ALFWORLD_CONFIGS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return find_repo_root() / "experiments/shared/train_eval/reference/alfworld/alfworld_server/configs"


CONFIGS_DIR = _configs_dir()

MAPPINGS_TRAIN = CONFIGS_DIR / "mappings_train.json"
MAPPINGS_TEST = CONFIGS_DIR / "mappings_test.json"
OUTPUT_UNSEEN = CONFIGS_DIR / "mappings_unseen.json"

OFFICIAL_PREFIXES = [
    "pick_and_place",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "look_at_obj_in_light",
    "pick_two_obj_and_place",
]


def _alfworld_data_root() -> Path:
    raw = os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld")
    return Path(os.path.expanduser(os.path.expandvars(raw)))


def _count_mappings(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return len(json.load(f))


def _is_solvable(game_file: Path) -> bool:
    """Read the official solvable flag from game.tw-pddl."""
    try:
        with open(game_file, "r", encoding="utf-8") as f:
            return json.load(f).get("solvable") is True
    except Exception:
        return False


def enumerate_solvable_unseen(unseen_root: Path):
    if not unseen_root.is_dir():
        raise FileNotFoundError(
            f"valid_unseen directory not found: {unseen_root}\n"
            f"请确认 ALFWORLD_DATA 指向包含 json_2.1.1/valid_unseen 的数据目录。"
        )
    games = sorted(unseen_root.glob("*/*/game.tw-pddl"))
    if not games:
        raise RuntimeError(f"no game.tw-pddl found under {unseen_root}")
    return games


def main() -> None:
    data_root = _alfworld_data_root()
    unseen_root = data_root / "json_2.1.1" / "valid_unseen"
    games = enumerate_solvable_unseen(unseen_root)

    start_id = _count_mappings(MAPPINGS_TRAIN) + _count_mappings(MAPPINGS_TEST)

    mappings = []
    skipped = 0
    for game_path in games:
        if not _is_solvable(game_path):
            skipped += 1
            continue
        mappings.append(
            {
                "item_id": start_id + len(mappings),
                # .../valid_unseen/<task_type>/<task_id>/game.tw-pddl
                "task_type": game_path.parent.parent.name,
                "task_id": game_path.parent.name,
            }
        )

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_UNSEEN, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=4)

    end_id = start_id + len(mappings) - 1 if mappings else start_id
    print(f"ALFWORLD_DATA        : {data_root}")
    print(f"valid_unseen games   : {len(games)}（文件系统枚举）")
    print(f"kept (solvable=True) : {len(mappings)}")
    print(f"skipped (unsolvable) : {skipped}")
    print(f"global index range   : [{start_id}, {end_id}]")
    print(f"saved                : {OUTPUT_UNSEEN}")

    from collections import Counter

    counts: "Counter[str]" = Counter()
    for m in mappings:
        tt = str(m["task_type"])
        counts[next((pref for pref in OFFICIAL_PREFIXES if tt.startswith(pref)), "UNKNOWN")] += 1
    print("by task-type prefix  :")
    for pref in OFFICIAL_PREFIXES:
        print(f"  {pref:<32} {counts.get(pref, 0)}")
    if counts.get("UNKNOWN"):
        print(f"  {'UNKNOWN':<32} {counts['UNKNOWN']}")


if __name__ == "__main__":
    main()
