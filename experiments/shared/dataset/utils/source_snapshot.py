"""Offline source snapshots, full validation and atomic dataset-directory publication.

Only immutable assets are hard-linked between generations. External source files
are copied, never linked to a cache. Linux RENAME_EXCHANGE makes all split names
change together; an interrupted publish always leaves a complete generation.
"""
from __future__ import annotations

import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

FORMAT = "source_tasks_v1"
COLUMNS = ["data_source", "index", "prompt", "ability", "reward_model", "extra_info"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path) -> dict:
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def ordered_identity_sha256(identities) -> str:
    digest = hashlib.sha256()
    for identity in identities:
        digest.update((json.dumps(identity, ensure_ascii=False) + "\n").encode("utf-8"))
    return digest.hexdigest()


def relative_file(root: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"Unsafe snapshot path: {name!r}")
    candidate = root.joinpath(*path.parts)
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Snapshot file escapes dataset: {name}")
    return candidate


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def copy_asset(source: Path, root: Path, relative: str) -> Path:
    destination = relative_file(root, relative)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if file_info(source) != file_info(destination):
        raise ValueError(f"Source changed while copying: {source}")
    return destination


def iter_rows(path: Path, columns=None):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
        yield from batch.to_pylist()


def write_rows(path: Path, rows) -> int:
    # Never truncate an immutable hard link inherited from the previous generation.
    path.unlink(missing_ok=True)
    writer = None
    count = 0
    buffer = []
    try:
        for row in rows:
            if list(row) != COLUMNS:
                raise ValueError(f"Expected six transport columns, got {list(row)}")
            buffer.append(row)
            if len(buffer) == 512:
                table = pa.Table.from_pylist(buffer, schema=writer.schema if writer else None)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)
                count += len(buffer)
                buffer.clear()
        if buffer:
            table = pa.Table.from_pylist(buffer, schema=writer.schema if writer else None)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            count += len(buffer)
    finally:
        if writer:
            writer.close()
    if not count:
        raise ValueError(f"Refusing empty split: {path}")
    return count


def source_ref(root: Path, path: str, *, row=None, task_id: str) -> dict:
    return {"path": path, "sha256": sha256(relative_file(root, path)), "row": row, "task_id": task_id}


def mark_row(row: dict, identity: str, reference: dict, source_split: str) -> dict:
    extra = row["extra_info"]
    extra.update(dataset_format=FORMAT, index=row["index"], source_id=identity,
                 source_record=reference, source_split=source_split)
    return row


def preserve_selection(active: Path, stage: Path, splits: dict[str, int]) -> None:
    """Save original transport rows exactly once, including all historical fields.

    These files are explicitly local lineage, NOT original upstream records.
    They fix membership/order, runtime identities and private historical references.
    """
    for split, count in splits.items():
        source = active / f"{split}.parquet"
        if pq.ParquetFile(source).metadata.num_rows != count:
            raise ValueError(f"Unexpected existing {split} size; refusing to change membership")
        if pq.ParquetFile(source).schema_arrow.names != COLUMNS:
            raise ValueError(f"Unexpected existing transport schema: {source}")
        copy_asset(source, stage, f"source/local_selection/{split}.parquet")


def _tracked(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    return (relative.parts[0] == "source" or
            relative.parts[0] == "envs" and "__pycache__" not in relative.parts and path.suffix != ".pyc" or
            len(relative.parts) == 1 and path.suffix == ".parquet")


def make_manifest(root: Path, environment: str, splits: dict[str, int], source: dict) -> dict:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and _tracked(root, path):
            if path.is_symlink():
                raise ValueError(f"Snapshots must not contain symlinks: {path}")
            files[path.relative_to(root).as_posix()] = file_info(path)
    split_info = {}
    for split, count in splits.items():
        ids = [row["extra_info"]["source_id"] for row in iter_rows(root / f"{split}.parquet")]
        if len(ids) != count:
            raise ValueError(f"Wrong row count for {split}")
        split_info[split] = {"path": f"{split}.parquet", "rows": count,
                             "ordered_identity_sha256": ordered_identity_sha256(ids)}
    manifest = {"format": FORMAT, "environment": environment, "files": files,
                "splits": split_info, "source": source,
                "identity_hash": "sha256 of each source_id as ensure_ascii=False JSON string plus LF, in row order",
                "runtime_protocol": "shared_step",
                "local_selection": "source/local_selection stores original local transport rows, not upstream data"}
    write_json(root / "manifest.json", manifest)
    return manifest


def validate_manifest(root: Path, environment: str | None = None) -> dict:
    """Full packaged file verification plus the public transport contract."""
    root = Path(root)
    manifest = verify_transport(root, environment)
    files = manifest["files"]
    for name, expected in files.items():
        path = relative_file(root, name)
        if not path.is_file() or file_info(path) != expected:
            raise ValueError(f"Missing or changed snapshot asset: {name}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and _tracked(root, p)}
    if actual != set(files):
        raise ValueError("Manifest does not cover exactly the source, environments and parquet files")
    return manifest


def verify_transport(root: Path, environment: str | None = None) -> dict:
    """Verify a dataset directory or one manifest-listed parquet for launch.

    A parquet argument checks only that split's bytes and rows; a directory checks
    all splits. Both require every declared source/environment asset to exist with
    its recorded size. This is NOT full source validation: raw asset hashes and
    source-to-derived equivalence need validate_manifest/validate_source_dataset.
    """
    root = Path(root)
    selected = root.name if root.suffix == ".parquet" else None
    if selected is not None:
        root = root.parent
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != FORMAT or (environment and manifest.get("environment") != environment):
        raise ValueError("Wrong dataset manifest format/environment")
    if manifest.get("runtime_protocol") != "shared_step":
        raise ValueError("Source task snapshots require the shared_step runtime protocol")
    if not manifest.get("source", {}).get("limitations"):
        raise ValueError("Source provenance limitations must be explicit")
    files = manifest["files"]
    specs = manifest["splits"]
    if selected is not None and sum(spec.get("path") == selected for spec in specs.values()) != 1:
        raise ValueError(f"Parquet is not uniquely listed in manifest splits: {selected}")
    for name, info in files.items():
        path = relative_file(root, name)
        if PurePosixPath(name).parts[0] in {"source", "envs"}:
            if not path.is_file() or path.stat().st_size != info["bytes"]:
                raise ValueError(f"Missing or changed snapshot asset size: {name}")
    for split, spec in specs.items():
        if selected is not None and spec["path"] != selected:
            continue
        path = relative_file(root, spec["path"])
        if not path.is_file() or file_info(path) != files.get(spec["path"]):
            raise ValueError(f"Missing or changed snapshot transport: {spec['path']}")
        if pq.ParquetFile(path).schema_arrow.names != COLUMNS:
            raise ValueError(f"Invalid transport schema: {split}")
        identities = []
        for row in iter_rows(path):
            extra = row["extra_info"]
            if extra.get("dataset_format") != FORMAT or extra.get("index") != row["index"]:
                raise ValueError(f"Bad format/index in {split}")
            if extra.get("split") != ("test" if split == "test_full" else split):
                raise ValueError(f"Bad local split in {split}")
            identity = extra.get("source_id")
            ref = extra.get("source_record") or {}
            if not isinstance(identity, str) or not identity or not ref.get("task_id"):
                raise ValueError(f"Missing source identity in {split}")
            name = ref.get("path", "")
            relative_file(root, name)
            if not name.startswith("source/") or files.get(name, {}).get("sha256") != ref.get("sha256"):
                raise ValueError(f"Invalid source record reference: {name}")
            if not row["prompt"] or any(m.get("role") not in {"system", "user"}
                                         or not isinstance(m.get("content"), str) for m in row["prompt"]):
                raise ValueError(f"Invalid public messages in {split}")
            identities.append(identity)
        if len(set(identities)) != len(identities):
            raise ValueError(f"Duplicate source identities in {split}")
        if len(identities) != spec["rows"] or ordered_identity_sha256(identities) != spec["ordered_identity_sha256"]:
            raise ValueError(f"Changed ordered identities/count in {split}")
    return manifest


def compare_rows(root: Path, split: str, expected) -> None:
    import itertools
    sentinel = object()
    for index, (actual, want) in enumerate(itertools.zip_longest(
            iter_rows(root / f"{split}.parquet"), expected, fillvalue=sentinel)):
        if actual != want:
            raise ValueError(f"Source/derived row mismatch: {split}[{index}]")


def _exchange(left: Path, right: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        operation = libc.renameat2
    except AttributeError as exc:
        raise RuntimeError("Atomic dataset publication requires Linux renameat2") from exc
    operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    operation.restype = ctypes.c_int
    if operation(-100, os.fsencode(left), -100, os.fsencode(right), 2):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(left))


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_DIRECTORY | os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _durable_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _recover(active: Path, journal: Path) -> None:
    if not journal.exists():
        return
    transaction = json.loads(journal.read_text())
    stage = active.parent / transaction["stage"]
    backup = active.parent / transaction["backup"]
    # All journal-controlled paths are sibling names generated by this module.
    if stage.parent != active.parent or backup.parent != active.parent:
        raise ValueError("Unsafe publication journal")
    new_active = ((active / "manifest.json").is_file()
                  and sha256(active / "manifest.json") == transaction["new_manifest_sha256"])
    if new_active and stage.exists():
        # Exchange completed before interruption. Finish retaining the old generation.
        os.replace(stage, backup / "previous_dataset")
    elif stage.exists():
        shutil.rmtree(stage)
    journal.unlink()
    _sync_directory(active.parent)


@contextlib.contextmanager
def dataset_lock(active: Path):
    active = Path(active).absolute()
    active.parent.mkdir(parents=True, exist_ok=True)
    with (active.parent / f".{active.name}.build.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another dataset build/validation holds the lock: {active}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def rebuild(active: Path, *, overwrite: bool, populate, validate) -> Path | None:
    """Build and validate a same-filesystem stage, then exchange full generations.

    ``populate(stage, active)`` writes only the stage. Existing source/environment
    assets are reused without extraction. A copied old-parquet recovery snapshot
    survives both success and failure. Exceptions after exchange roll back atomically.
    """
    active = Path(active).absolute()
    with dataset_lock(active):
        journal = active.parent / f".{active.name}.publish.json"
        _recover(active, journal)
        if not active.is_dir():
            raise ValueError("Rebuild requires an audited existing dataset to fix membership and order")
        if not overwrite:
            raise FileExistsError(f"Explicit --overwrite required for {active}")
        old = {p.name: file_info(p) for p in sorted(active.glob("*.parquet"))}
        if not old:
            raise ValueError("No existing split parquets to preserve")
        old_manifest = file_info(active / "manifest.json") if (active / "manifest.json").exists() else None
        if old_manifest:
            # Never bless altered packaged sources by regenerating their checksums.
            validate(active)
        stage = Path(tempfile.mkdtemp(prefix=f".{active.name}.stage-", dir=active.parent))
        backup = None
        exchanged = False
        previous = stage
        cleanup_stage = True
        try:
            # Retain ancillary files without touching live envs, lock files or source.
            shutil.copytree(active, stage, dirs_exist_ok=True, copy_function=os.link, symlinks=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for name in [*old, "manifest.json"]:
                (stage / name).unlink(missing_ok=True)
            populate(stage, active)
            validate(stage)
            new_manifest = sha256(stage / "manifest.json")
            if old_manifest and old_manifest["sha256"] == new_manifest:
                print(f"[snapshot] unchanged: {active}", flush=True)
                return None
            backup = active.parent / f"{active.name}.recovery-{uuid.uuid4().hex}"
            backup.mkdir()
            for name, info in old.items():
                shutil.copyfile(active / name, backup / name)
                if file_info(backup / name) != info:
                    raise ValueError(f"Concurrent modification while backing up {name}")
            if old_manifest:
                shutil.copyfile(active / "manifest.json", backup / "manifest.json")
            write_json(backup / "recovery.json", {"format": "dataset_recovery_v1", "files": old,
                                                  "previous_manifest": old_manifest})
            if {p.name: file_info(p) for p in active.glob("*.parquet")} != old:
                raise ValueError("Active parquets changed during build; refusing to overwrite")
            if (file_info(active / "manifest.json") if (active / "manifest.json").exists() else None) != old_manifest:
                raise ValueError("Active manifest changed during build")
            # Flush staged and recovery file data before the durable publication journal.
            for root in (stage, backup):
                for path in root.rglob("*"):
                    if path.is_file() and not path.is_symlink():
                        with path.open("rb") as stream:
                            os.fsync(stream.fileno())
                for directory in sorted((p for p in root.rglob("*") if p.is_dir()),
                                        key=lambda p: len(p.parts), reverse=True):
                    _sync_directory(directory)
                _sync_directory(root)
            _durable_json(journal, {"stage": stage.name, "backup": backup.name,
                                   "new_manifest_sha256": new_manifest})
            _exchange(stage, active)
            exchanged = True
            _sync_directory(active.parent)
            # All expensive validation ran before exchange. Detect a failed/incorrect exchange.
            if sha256(active / "manifest.json") != new_manifest:
                raise ValueError("Published manifest differs from validated stage")
            os.replace(stage, backup / "previous_dataset")
            previous = backup / "previous_dataset"
            journal.unlink()
            _sync_directory(active.parent)
            exchanged = False
            print(f"[snapshot] published {active}; recovery snapshot: {backup}", flush=True)
            return backup
        except BaseException:
            if exchanged:
                try:
                    _exchange(previous, active)
                    _sync_directory(active.parent)
                except BaseException:
                    # Preserve both full generations and journal for explicit recovery.
                    # Never delete the old generation when rollback itself fails.
                    cleanup_stage = False
                    raise
            journal.unlink(missing_ok=True)
            raise
        finally:
            if cleanup_stage and stage.exists():
                shutil.rmtree(stage)


def check_only(active: Path, validate) -> None:
    with dataset_lock(active):
        journal = active.parent / f".{active.name}.publish.json"
        if journal.exists():
            raise ValueError(f"Interrupted publication needs --overwrite recovery: {journal}")
        validate(active)
    print(f"[snapshot] full source and derived validation passed: {active}", flush=True)
