# Extract CodeGym envs_en rows as <source>__<env_name>.py, retaining upstream naming.
# Write generated sources under data/codegym/envs/codegym_v1, overridable with CODEGYM_ENVS_DIR.
# External source checkouts remain unchanged.
import os
import glob
import argparse
import pyarrow.parquet as pq


def _find_repo_root():
    """Find the repository via pyproject.toml and agent_system/; return None if absent."""
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(d, "pyproject.toml")) and os.path.isdir(os.path.join(d, "agent_system")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def _default_envs_dir():
    override = os.environ.get("CODEGYM_ENVS_DIR")
    if override:
        return override
    root = _find_repo_root()
    if root is None:
        return None
    return os.path.join(root, "data", "codegym", "envs", "codegym_v1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        default=_default_envs_dir(),
        help="target dir for codegym_v1 env .py files (默认取 $CODEGYM_ENVS_DIR，"
        "否则 data/codegym/envs/codegym_v1)",
    )
    ap.add_argument(
        "--parquet",
        default=None,
        help="path to envs_en parquet; default = resolve from HF cache",
    )
    ap.add_argument("--limit", type=int, default=0, help="only extract first N envs (0 = all)")
    args = ap.parse_args()

    if not args.out_dir:
        ap.error(
            "找不到 Dynamic_ExpA 根目录（标记：同级存在 pyproject.toml 和 agent_system/），"
            "请显式指定 --out_dir 或设置 CODEGYM_ENVS_DIR"
        )

    parquet = args.parquet
    if parquet is None:
        # Respect HF_HOME and HF_HUB_CACHE on both local and mounted storage.
        hf_home = os.environ.get("HF_HOME")
        hub = os.environ.get("HF_HUB_CACHE") or (
            os.path.join(hf_home, "hub") if hf_home else os.path.expanduser("~/.cache/huggingface/hub")
        )
        base = os.path.join(hub, "datasets--VanishD--CodeGym", "snapshots")
        matches = glob.glob(base + "/*/envs_en/*.parquet")
        assert matches, f"envs_en parquet not found under {base}; download dataset first"
        parquet = matches[0]

    os.makedirs(args.out_dir, exist_ok=True)
    table = pq.read_table(parquet, columns=["env_code", "env_name", "source"])
    n = table.num_rows
    code_col = table.column("env_code")
    name_col = table.column("env_name")
    src_col = table.column("source")

    written = 0
    collisions = 0
    seen = set()
    limit = args.limit or n
    for i in range(min(limit, n)):
        env_code = code_col[i].as_py()
        env_name = name_col[i].as_py()
        source = src_col[i].as_py()
        fname = f"{source}__{env_name}.py"
        if fname in seen:
            collisions += 1
            continue
        seen.add(fname)
        with open(os.path.join(args.out_dir, fname), "w", encoding="utf-8") as f:
            f.write(env_code)
        written += 1
        if written % 2000 == 0:
            print(f"  written {written}/{limit} ...")

    # ensure package marker so importlib-by-path still works regardless
    open(os.path.join(args.out_dir, "__init__.py"), "a").close()
    print(f"Done. wrote {written} env files to {args.out_dir} (collisions skipped: {collisions})")


if __name__ == "__main__":
    main()
