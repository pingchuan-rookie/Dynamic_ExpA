"""Export ALFWorld game paths to train_file.json and test_file.json.

CreateMappings.py converts these lists into the server's game mappings.
The train split reads json_2.1.1/train; eval_in_distribution reads valid_seen,
not valid_train. Resolve configuration from the repository root.

Usage with an interpreter that has ALFWorld installed:
    ALFWORLD_DATA=~/.cache/alfworld python export_game_files.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

SPLIT_TO_OUTPUT = {
    "train": "train_file.json",
    "eval_in_distribution": "test_file.json",
}


def find_repo_root() -> Path:
    """Find the repository by its pyproject.toml file and agent_system/ directory."""
    for d in HERE.parents:
        if (d / "pyproject.toml").is_file() and (d / "agent_system").is_dir():
            return d
    raise RuntimeError(
        f"从 {HERE} 向上找不到 Dynamic_ExpA 根目录（标记：同级存在 pyproject.toml 和 agent_system/）。"
    )


def default_config_path() -> Path:
    """Resolve base_config.yaml, defaulting to the reference server configuration."""
    override = os.environ.get("ALFWORLD_CONFIGS_DIR")
    cfg_dir = (
        Path(override).expanduser().resolve()
        if override
        else find_repo_root() / "experiments/shared/train_eval/reference/alfworld/alfworld_server/configs"
    )
    return cfg_dir / "base_config.yaml"


def alfworld_data_root() -> Path:
    raw = os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld")
    return Path(os.path.expanduser(os.path.expandvars(raw)))


def main() -> None:
    output_root = find_repo_root() / "data/alfworld/mappings_source"
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = alfworld_data_root()
    os.environ["ALFWORLD_DATA"] = str(data_root)

    config_path = default_config_path()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Defer the ALFWorld import until scanning is requested.
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    print("=" * 78)
    print(f"ALFWORLD_DATA : {data_root}  (exists={data_root.exists()})")
    print(f"config        : {config_path}")
    print(f"输出目录       : {output_root}")
    print("=" * 78)

    for split, filename in SPLIT_TO_OUTPUT.items():
        print(f"\n[{split}]")
        env = AlfredTWEnv(config, train_eval=split)
        game_files = sorted(env.game_files)
        del env

        print(f"  扫到 {len(game_files)} 局")
        if not game_files:
            print("  跳过（空）")
            continue

        # Store paths relative to ALFWORLD_DATA for portability.
        prefix = f"{data_root}{os.sep}"
        rel = [g[len(prefix):] if g.startswith(prefix) else g for g in game_files]

        output_path = output_root / filename
        with open(output_path, "w") as f:
            json.dump(rel, f, indent=2)

        print(f"  首条: {rel[0]}")
        print(f"  末条: {rel[-1]}")
        print(f"  写入: {output_path}")

    print("\nDone. 接下来跑 CreateMappings.py 生成 mappings_{train,test}.json。")


if __name__ == "__main__":
    main()
