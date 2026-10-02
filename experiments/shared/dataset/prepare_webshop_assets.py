"""Prepare the pinned official full/human WebShop assets and isolated backend.

Run with any Python 3.10+; --install creates only .venvs/webshop (never dyad-verl).
Downloads are checksum-pinned public mirrors of the upstream setup.sh files.
Indexing uses the official product loader, document fields and Lucene flags.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
BACKEND_PATH = ROOT / "agent_system/environments/env_package/webshop/envs.py"
spec = importlib.util.spec_from_file_location("webshop_backend", BACKEND_PATH)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
MIRROR = "https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/ce990fff5aee388db2706f07820c578ab68e0453"
DRIVE_IDS = {"items_shuffle.json": "1A2whVgOO0euk5O13n2iYDM0bQRkkRduB",
             "items_ins_v2.json": "1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi",
             "items_human_ins.json": "14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O"}
JAVA_URL = "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.32.1%2B1/OpenJDK11U-jdk_x64_linux_hotspot_11.0.32.1_1.tar.gz"
JAVA_SHA256 = "5c3f68887c325d36d852ba534303e1f5f1f5cae7d6cc1e951d73e0d8e98a058d"
REQUIREMENTS = Path(__file__).with_name("webshop_requirements.txt")
LOCKFILE = Path(__file__).with_name("webshop_requirements.lock.txt")


def run(command, **kwargs):
    print("Running: " + " ".join(map(str, command)), file=sys.stderr, flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def download(url, target, sha256):
    target = Path(target)
    if target.is_file():
        if backend.sha256_file(target) != sha256:
            raise ValueError(f"Existing file checksum mismatch; move it aside explicitly: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    # Adopt interrupted/manual downloads only after a complete checksum match.
    if partial.is_file() and backend.sha256_file(partial) == sha256:
        partial.replace(target)
        return
    print(f"Downloading {url} to {target}", file=sys.stderr, flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as stream:
        shutil.copyfileobj(response, stream, 8 * 1024 * 1024)
    if backend.sha256_file(partial) != sha256:
        raise ValueError(f"Downloaded checksum mismatch: {partial}")
    partial.replace(target)


def install(venv):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Pinned Java archive supports Linux x86_64 only")
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required to prepare the isolated Python 3.10 environment")
    if not (venv / "bin/python").is_file():
        run([uv, "venv", "--python", "3.10", venv])
    version = subprocess.check_output([str(venv / "bin/python"), "-c",
                                      "import sys; print('.'.join(map(str, sys.version_info[:2])))"], text=True).strip()
    if version != "3.10":
        raise RuntimeError(f"WebShop requires isolated Python 3.10, got {version}")
    run([uv, "pip", "sync", "--python", venv / "bin/python", LOCKFILE])
    java = venv / "java"
    if not (java / "bin/java").is_file():
        archive = venv / "java11.tar.gz"
        download(JAVA_URL, archive, JAVA_SHA256)
        temp = venv / "java.part"
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir()
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe member in Java archive: {member.name}")
                if member.issym() or member.islnk():
                    link = (temp / path.parent / member.linkname if member.issym()
                            else temp / member.linkname).resolve()
                    if not link.is_relative_to(temp.resolve()):
                        raise ValueError(f"Unsafe Java archive link: {member.name}")
            tar.extractall(temp)
        entries = list(temp.iterdir())
        if len(entries) != 1:
            raise ValueError("Unexpected Java archive layout")
        entries[0].replace(java)
        temp.rmdir()
    run([java / "bin/java", "-version"])


def file_record(path):
    return {"sha256": backend.sha256_file(path), "size": path.stat().st_size}


def source_record(source):
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    names = ["web_agent_site/utils.py", "web_agent_site/engine/engine.py",
             "web_agent_site/engine/goal.py", "web_agent_site/engine/normalize.py",
             "web_agent_site/envs/web_agent_text_env.py"]
    names += [str(p.relative_to(source)) for p in sorted((source / "web_agent_site/templates").glob("*.html"))]
    return revision, {name: file_record(source / name) for name in names}


def build_index(args, source_files):
    assets = args.assets_dir
    root = assets / "search_engine"
    index = root / "indexes"
    receipt = root / "index_manifest.json"
    inputs = {name: digest for name, digest in backend.ASSET_HASHES.items()}
    identity = {"input_sha256": inputs, "source_files": source_files,
                "pyserini": "0.17.0", "document_format": "official-full-v1"}
    if index.exists():
        if not receipt.is_file():
            raise ValueError("Existing index has no provenance receipt; remove it explicitly to rebuild")
        saved = json.loads(receipt.read_text())
        if any(saved.get(k) != v for k, v in identity.items()):
            raise ValueError("Existing index provenance disagrees with full source/assets")
        for name, metadata in saved["files"].items():
            if file_record(index / name) != metadata:
                raise ValueError(f"Existing index is corrupt: {name}")
        documents = root / "resources/documents.jsonl"
        if documents.is_file() and not args.keep_documents:
            if backend.sha256_file(documents) != saved["documents_sha256"]:
                raise ValueError("Retained indexing documents differ from index provenance")
            documents.unlink()
            if not any(documents.parent.iterdir()):
                documents.parent.rmdir()
        return saved
    official = backend.configure_official(args.source_dir, assets, args.java_home)
    random.seed(backend.SEED)
    products, product_map, prices, attributes = official.load_products(
        str(assets / "items_shuffle.json"), human_goals=True)
    if len(products) < 1_000_000:
        raise ValueError("Refusing to build full index from a small product catalog")
    resources = root / "resources"
    resources.mkdir(parents=True, exist_ok=True)
    documents = resources / "documents.jsonl"
    empty_document_ids = []
    with open(documents.with_suffix(".part"), "w", encoding="utf-8") as stream:
        for product in products:
            options = ", and ".join(f"{key}: {', '.join(values)}" for key, values in product.get("options", {}).items())
            contents = " ".join([product["Title"], product["Description"],
                                 product["BulletPoints"][0], options]).lower()
            if not contents.strip():
                empty_document_ids.append(product["asin"])
            stream.write(json.dumps({"id": product["asin"], "contents": contents, "product": product}) + "\n")
    documents.with_suffix(".part").replace(documents)
    count = len(products)
    del products, product_map, prices, attributes, product
    import gc
    gc.collect()
    temp_index = root / "indexes.part"
    if temp_index.exists():
        shutil.rmtree(temp_index)
    run([sys.executable, "-m", "pyserini.index.lucene", "--collection", "JsonCollection",
         "--input", resources, "--index", temp_index, "--generator", "DefaultLuceneDocumentGenerator",
         "--threads", "1", "--storePositions", "--storeDocvectors", "--storeRaw"])
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(str(temp_index))
    try:
        if searcher.num_docs != count - len(empty_document_ids):
            raise ValueError(f"Index document count {searcher.num_docs} != nonempty catalog {count - len(empty_document_ids)}")
        if not searcher.search("shoes", k=1):
            raise ValueError("New full Lucene index returned no result for sanity query")
    finally:
        searcher.close()
    temp_index.replace(index)
    saved = {**identity, "path": "search_engine/indexes", "documents": count - len(empty_document_ids),
             "catalog_products": count, "empty_document_ids": empty_document_ids,
             "documents_sha256": backend.sha256_file(documents),
             "files": {p.name: file_record(p) for p in sorted(index.iterdir()) if p.is_file()}}
    atomic_json(receipt, saved)
    if not args.keep_documents:
        documents.unlink()
        resources.rmdir()
    return saved


def atomic_json(path, value):
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)




def prepare(args):
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in backend.ASSET_HASHES.items():
        download(f"{MIRROR}/{name}", args.assets_dir / name, expected)
    revision, source_files = source_record(args.source_dir)
    index = build_index(args, source_files)
    instance = backend.WebShopEnv({"assets_dir": str(args.assets_dir), "source_dir": str(args.source_dir),
                                      "java_home": str(args.java_home)}, verify_manifest=False)
    try:
        tasks_path = args.assets_dir / "tasks.jsonl"
        candidate = args.assets_dir / "tasks.candidate.jsonl"
        previous_path = args.assets_dir / "manifest.json"
        previous = json.loads(previous_path.read_text()) if previous_path.is_file() else None
        if previous:
            for key in ("goals_sha256", "product_prices_sha256", "counts"):
                if previous.get(key) != getattr(instance, key):
                    raise ValueError(f"Re-preparation changed benchmark identity: {key}")
        exported = instance.export_tasks(output_path=candidate)
        if previous and previous["tasks"]["sha256"] != exported["sha256"]:
            raise ValueError("Re-preparation changed task initial observations; old export preserved")
        candidate.replace(tasks_path)
        # importlib.metadata works in uv venvs without installing pip itself.
        import importlib.metadata
        installed = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
        manifest = {"format_version": backend.FORMAT_VERSION, "benchmark": "webshop", "variant": "full",
                    "human_goals": True, "seed": backend.SEED, "goal_shuffle_seed": 233,
                    "observation_mode": "text_rich", "source_revision": revision, "source_files": source_files,
                    "construction_order": ["seed(233)", "load_products", "init_search_engine", "get_human_goals", "seed(233)", "shuffle(goals)"],
                    "files": {name: {**file_record(args.assets_dir / name), "url": f"{MIRROR}/{name}",
                                      "official_drive_url": f"https://drive.google.com/uc?id={DRIVE_IDS[name]}"}
                              for name in backend.ASSET_HASHES},
                    "index": index, "counts": instance.counts,
                    "split_ranges": {"test": [0, 500], "dev": [500, 1500], "train": [1500, instance.counts["goals"]]},
                    "goals_sha256": instance.goals_sha256, "product_prices_sha256": instance.product_prices_sha256,
                    "tasks": {"path": "tasks.jsonl", **file_record(tasks_path)},
                    "runtime": {"python": platform.python_version(), "requirements_sha256": backend.sha256_file(REQUIREMENTS),
                                "lockfile_sha256": backend.sha256_file(LOCKFILE),
                                "packages": installed, "java_url": JAVA_URL, "java_archive_sha256": JAVA_SHA256},
                    "verification": {"scope": "asset_integrity", "interaction_tested": False}}
        atomic_json(args.assets_dir / "manifest.json", manifest)
        print(json.dumps({"manifest": str(args.assets_dir / "manifest.json"), "counts": instance.counts,
                          "verification": {"scope": "asset_integrity", "interaction_tested": False}}, indent=2), flush=True)
    finally:
        instance.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Create/update isolated Python 3.10 dependencies and Java 11")
    parser.add_argument("--install-only", action="store_true")
    parser.add_argument("--venv", type=Path, default=ROOT / ".venvs/webshop")
    parser.add_argument("--assets-dir", type=Path, default=backend.DEFAULT_ASSETS)
    parser.add_argument("--source-dir", type=Path, default=backend.DEFAULT_SOURCE)
    parser.add_argument("--java-home", type=Path)
    parser.add_argument("--keep-documents", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Validate asset provenance, deterministic goals and task export without rebuilding")
    args = parser.parse_args()
    args.venv = args.venv.resolve()
    if args.venv == (ROOT / ".venvs/expa-verl").resolve():
        parser.error("The training environment must not be modified")
    args.assets_dir, args.source_dir = args.assets_dir.resolve(), args.source_dir.resolve()
    args.java_home = (args.java_home or args.venv / "java").resolve()
    if args.install or args.install_only:
        install(args.venv)
    if args.install_only:
        return
    python = args.venv / "bin/python"
    if not python.is_file():
        parser.error("WebShop Python missing; pass --install")
    if Path(sys.prefix).resolve() != args.venv:
        command = [python, Path(__file__).resolve(), "--venv", args.venv, "--assets-dir", args.assets_dir,
                   "--source-dir", args.source_dir, "--java-home", args.java_home]
        if args.keep_documents:
            command.append("--keep-documents")
        if args.verify_only:
            command.append("--verify-only")
        run(command)
        return
    os.environ["JAVA_HOME"] = str(args.java_home)
    os.environ["PATH"] = str(args.java_home / "bin") + os.pathsep + os.environ.get("PATH", "")
    if args.verify_only:
        instance = backend.WebShopEnv({"assets_dir": str(args.assets_dir), "source_dir": str(args.source_dir),
                                          "java_home": str(args.java_home)})
        try:
            tasks = args.assets_dir / instance.manifest["tasks"]["path"]
            if backend.sha256_file(tasks) != instance.manifest["tasks"]["sha256"]:
                raise ValueError("Task export checksum mismatch")
            print(json.dumps({"health": instance.health(), "verification": {
                "scope": "asset_integrity", "interaction_tested": False,
            }}, indent=2))
        finally:
            instance.shutdown()
        return
    # Prevent simultaneous prepare processes from replacing the same index/tasks.
    import fcntl
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    with open(args.assets_dir / ".prepare.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another WebShop asset preparation is already running") from exc
        prepare(args)


if __name__ == "__main__":
    main()
