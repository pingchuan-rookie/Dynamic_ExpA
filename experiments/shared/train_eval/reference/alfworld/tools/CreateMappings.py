"""Build mappings_{train,test}.json from exported game-path lists.

Run export_game_files.py first. Read train_file.json and test_file.json from
this directory and write to the reference server's configs/ directory, overridable
with ALFWORLD_CONFIGS_DIR. Use the same lists as the server to preserve task ids.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


# =========================
# Global manual settings
# =========================

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


def configs_dir() -> Path:
    """Return the configs directory read by the reference server."""
    override = os.environ.get("ALFWORLD_CONFIGS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return find_repo_root() / "experiments/shared/train_eval/reference/alfworld/alfworld_server/configs"


INPUTS = find_repo_root() / "data/alfworld/mappings_source"
TRAIN_FILE = INPUTS / "train_file.json"
TEST_FILE = INPUTS / "test_file.json"

OUTPUT_TRAIN_MAPPING = configs_dir() / "mappings_train.json"
OUTPUT_TEST_MAPPING = configs_dir() / "mappings_test.json"


def load_json_list(path: str | Path) -> List[Any]:
    input_path = Path(path)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"{input_path} must be a JSON list")

    return data


def extract_task_fields(game_path: str) -> Tuple[str, str]:
    parts = Path(str(game_path)).parts

    if len(parts) < 3:
        raise ValueError(f"Cannot parse task_type/task_id from path: {game_path}")

    if parts[-1] != "game.tw-pddl":
        raise ValueError(f"Game path must end with game.tw-pddl: {game_path}")

    return parts[-3], parts[-2]


def build_mappings(
    game_paths: List[Any],
    *,
    start_item_id: int,
) -> List[Dict[str, Any]]:
    mappings = []

    for offset, game_path in enumerate(game_paths):
        task_type, task_id = extract_task_fields(str(game_path))

        mappings.append(
            {
                "item_id": start_item_id + offset,
                "task_type": task_type,
                "task_id": task_id,
            }
        )

    return mappings


def save_json(data: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Saved json: {output_path}")


def create_mapping_files() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_paths = load_json_list(TRAIN_FILE)
    test_paths = load_json_list(TEST_FILE)

    train_mappings = build_mappings(train_paths, start_item_id=0)
    test_mappings = build_mappings(test_paths, start_item_id=len(train_mappings))

    save_json(train_mappings, OUTPUT_TRAIN_MAPPING)
    save_json(test_mappings, OUTPUT_TEST_MAPPING)

    print("=== train mappings size ===", len(train_mappings))
    print("=== test mappings size ===", len(test_mappings))

    if train_mappings:
        print("=== train last ===", train_mappings[-1])

    if test_mappings:
        print("=== test first ===", test_mappings[0])

    return train_mappings, test_mappings


if __name__ == "__main__":
    create_mapping_files()