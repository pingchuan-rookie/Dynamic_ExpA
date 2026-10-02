"""Pinned SWE-bench Verified assets. Verification never downloads or builds anything."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.request import urlopen

DATASET_ID = "princeton-nlp/SWE-bench_Verified"
DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
DATASET_FILE = "data/test-00000-of-00001.parquet"
SOURCE_SHA256 = "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
TASKS_SHA256 = "e1b70254514c107a92a37514ee94faae646baed60f97bd535097bb910d042df5"
FULL_INSTANCE_IDS_SHA256 = "33e18be7a9bd9f674790b63ed4d0b3fb17c176994802e3062b7d5a430a4e7d16"
FULL_TASK_COUNT = 500
HARNESS_VERSION = "4.1.0"
HARNESS_COMMIT = "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
HARNESS_SOURCE_MANIFEST = Path(__file__).with_name("harness_source_sha256.json")
HARNESS_SOURCE_MANIFEST_SHA256 = "72b80540eb65637d56a8eebf1ec9c8beaefed85c490b3c1395f232c5d250337f"
PUBLIC_FIELDS = ("instance_id", "problem_statement", "repo", "base_commit")
FORMAT = "swebench-verified-assets-v1"
SOURCE_NAME = "source.parquet"
TASKS_NAME = "evaluator_tasks.json"
MANIFEST_NAME = "manifest.json"
ARCHITECTURES = {"x86_64": "amd64", "arm64": "arm64"}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def value_digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def _select_ids(all_ids, instance_ids):
    selected = list(all_ids if instance_ids is None else instance_ids)
    if (not selected or not all(isinstance(item, str) for item in selected)
            or len(selected) != len(set(selected))):
        raise ValueError("Select nonempty, unique instance IDs")
    unknown = set(selected) - set(all_ids)
    if unknown:
        raise ValueError(f"Unknown instance IDs: {sorted(unknown)}")
    return sorted(selected)


def _validate_tasks(tasks):
    if not isinstance(tasks, list) or len(tasks) != FULL_TASK_COUNT:
        raise ValueError(f"Official Verified source must contain all {FULL_TASK_COUNT} tasks")
    for task in tasks:
        if not isinstance(task, dict) or any(not isinstance(task.get(key), str) or not task[key]
                                             for key in PUBLIC_FIELDS):
            raise ValueError("Invalid official task public fields")
        if not re.fullmatch(r"[0-9a-f]{40}", task["base_commit"]):
            raise ValueError(f"Invalid base_commit for {task['instance_id']}")
        if any(key not in task for key in ("patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS")):
            raise ValueError("Missing evaluator-only task fields")
    ids = sorted(task["instance_id"] for task in tasks)
    if len(set(ids)) != FULL_TASK_COUNT or value_digest(ids) != FULL_INSTANCE_IDS_SHA256:
        raise ValueError("Official 500-task membership mismatch")
    if value_digest(tasks) != TASKS_SHA256:
        raise ValueError("Official raw task content mismatch")
    return ids


def official_image_reference(instance_id, architecture="x86_64"):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+", instance_id):
        raise ValueError(f"Invalid instance ID: {instance_id!r}")
    # This mutable registry tag is used only during explicit local registration.
    # Runtime containers must use the recorded immutable image_id instead.
    key = f"swebench/sweb.eval.{architecture}.{instance_id.lower()}:latest"
    return key.replace("__", "_1776_")


def _image_identity(docker_client, reference, architecture):
    if docker_client is None:
        raise ValueError("A local Docker client is required; images are never pulled automatically")
    try:
        image = docker_client.images.get(reference)
    except Exception as exc:
        raise RuntimeError(f"Required local Docker image unavailable: {reference}; no pull attempted") from exc
    image_id = image.id
    if not isinstance(image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError(f"Invalid immutable Docker image ID: {reference}")
    attrs = image.attrs
    if attrs.get("Architecture") != ARCHITECTURES[architecture] or attrs.get("Os") != "linux":
        raise ValueError(f"Docker image platform mismatch: {reference}")
    digests = attrs.get("RepoDigests") or []
    if not isinstance(digests, list) or any(not isinstance(item, str)
            or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", item) for item in digests):
        raise ValueError(f"Invalid Docker repository digests: {reference}")
    return {"image_id": image_id, "repo_digests": sorted(set(digests)),
            "architecture": attrs["Architecture"], "os": attrs["Os"]}


def _valid_digests(values, *, nonempty=False):
    return (isinstance(values, list) and (bool(values) or not nonempty)
            and all(isinstance(item, str) and re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", item)
                    for item in values)
            and len(values) == len(set(values)))


def _validate_image_record(task_id, image, architecture):
    official = official_image_reference(task_id, architecture)
    if (not isinstance(image, dict)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(image.get("image_id", "")))
            or image.get("architecture") != ARCHITECTURES[architecture]
            or image.get("os") != "linux" or not _valid_digests(image.get("repo_digests"))):
        raise ValueError(f"Invalid registered image identity: {task_id}")
    kind = image.get("kind", "official")
    if kind == "official":
        if image.get("reference") != official:
            raise ValueError(f"Nonofficial image reference requires prepared provenance: {task_id}")
    elif kind == "prepared":
        reference = image.get("reference")
        if (not isinstance(reference, str) or not reference or re.search(r"\s", reference)
                or reference == official or image.get("upstream_reference") != official
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(image.get("upstream_image_id", "")))
                or image["image_id"] == image["upstream_image_id"]
                or not _valid_digests(image.get("upstream_repo_digests"), nonempty=True)
                or not isinstance(image.get("preparation"), dict) or not image["preparation"]
                or image.get("preparation_sha256") != value_digest(image["preparation"])):
            raise ValueError(f"Invalid prepared image provenance: {task_id}")
    else:
        raise ValueError(f"Unknown image kind: {kind}")


def _prepared_image(task_id, override, docker_client, architecture):
    required = {"reference", "upstream_image_id", "upstream_repo_digests", "preparation"}
    if not isinstance(override, dict) or set(override) != required:
        raise ValueError(f"Image override for {task_id} requires exactly {sorted(required)}")
    official = official_image_reference(task_id, architecture)
    # The registry tag is not retagged or rewritten to impersonate an upstream image.
    image = {**override, "kind": "prepared", "upstream_reference": official,
             "preparation_sha256": value_digest(override["preparation"]),
             **_image_identity(docker_client, override["reference"], architecture)}
    _validate_image_record(task_id, image, architecture)
    upstream = _image_identity(docker_client, override["upstream_image_id"], architecture)
    if (upstream["image_id"] != override["upstream_image_id"]
            or not set(override["upstream_repo_digests"]).issubset(upstream["repo_digests"])):
        raise ValueError(f"Prepared image upstream identity mismatch: {task_id}")
    # Docker diff IDs authenticate that the prepared root filesystem extends
    # the declared base, not merely an unrelated image named in a JSON record.
    base_layers = docker_client.images.get(upstream["image_id"]).attrs.get("RootFS", {}).get("Layers", [])
    layers = docker_client.images.get(image["image_id"]).attrs.get("RootFS", {}).get("Layers", [])
    if not base_layers or layers[:len(base_layers)] != base_layers:
        raise ValueError(f"Prepared image does not extend declared upstream layers: {task_id}")
    return image


def verify_harness_identity():
    """Authenticate installed source bytes without importing unverified harness code.

    The static roster comes from the pinned Git tree and its namespaces=false
    package discovery, not from an installed wheel's mutable RECORD metadata.
    No upstream checkout, Git executable or network is needed at verification.
    """
    distribution = metadata.distribution("swebench")
    if distribution.version != HARNESS_VERSION:
        raise ValueError(f"SWE-bench harness must be {HARNESS_VERSION}, found {distribution.version}")
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        source = json.loads(direct_url)
        vcs = source.get("vcs_info")
        if vcs and vcs.get("commit_id") != HARNESS_COMMIT:
            raise ValueError("Installed SWE-bench harness VCS commit differs from pinned release")
        if source.get("dir_info", {}).get("editable"):
            raise ValueError("Editable SWE-bench harness installs are not immutable; install the pinned release")
    if file_digest(HARNESS_SOURCE_MANIFEST) != HARNESS_SOURCE_MANIFEST_SHA256:
        raise ValueError("Pinned harness source checksum manifest SHA256 mismatch")
    manifest = read_json(HARNESS_SOURCE_MANIFEST)
    if (manifest.get("format") != "swebench-installed-source-sha256-v1"
            or manifest.get("harness_version") != HARNESS_VERSION
            or manifest.get("harness_commit") != HARNESS_COMMIT):
        raise ValueError("Pinned harness source checksum manifest identity mismatch")
    package_root = Path(distribution.locate_file("swebench"))
    if package_root.is_symlink() or not package_root.is_dir():
        raise ValueError("Installed SWE-bench package root is missing or symlinked")
    expected = manifest["files"]
    for relative, digest in expected.items():
        path = Path(distribution.locate_file(relative))
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Installed SWE-bench source missing or symlinked: {relative}")
        if file_digest(path) != digest:
            raise ValueError(f"Installed SWE-bench source SHA256 mismatch: {relative}")
    # Normal __pycache__ bytecode is generated by Python on installation/import.
    # Sourceless bytecode and extra modules/extensions can change import behavior.
    for path in package_root.rglob("*"):
        relative = "swebench/" + path.relative_to(package_root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Installed SWE-bench contains a symlink: {relative}")
        if path.is_file() and relative not in expected:
            if path.suffix == ".pyc" and path.parent.name == "__pycache__":
                source_name = path.name.split(".", 1)[0] + ".py"
                source = "swebench/" + (path.parent.parent / source_name).relative_to(package_root).as_posix()
                if source in expected:
                    continue
            raise ValueError(f"Installed SWE-bench contains unpinned package file: {relative}")
    return {"version": HARNESS_VERSION, "commit": HARNESS_COMMIT,
            "source_manifest_sha256": HARNESS_SOURCE_MANIFEST_SHA256,
            "verified_source_files": len(expected)}


def _write_manifest(directory, manifest, *, replace=False):
    directory = Path(directory)
    text = canonical_json(manifest) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    # Registration is the only explicit manifest update operation. Existing
    # source data and already registered image identities are never replaced.
    for name, content in ((MANIFEST_NAME, text), ("manifest.sha256", digest + "\n")):
        if replace:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            try:
                os.replace(temporary, directory / name)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            with (directory / name).open("x", encoding="utf-8") as stream:
                stream.write(content)


def prepare_assets(output_dir, *, instance_ids=None, docker_client=None, download=False,
                   source_file=None, architecture="x86_64"):
    """Prepare full gold source with an explicit full/debug selection.

    Networking requires download=True. A pinned, predownloaded source_file can
    instead be imported entirely offline. Existing valid output is idempotent;
    changed source, selection or platform is rejected rather than overwritten.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    if download and source_file is not None:
        raise ValueError("Choose download or a local source_file, not both")
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        manifest = verify_assets(output_dir, require_images=False)
        selected = _select_ids(manifest["full_instance_ids"], instance_ids)
        if selected != manifest["instance_ids"] or architecture != manifest["architecture"]:
            raise ValueError("Existing assets use another selection/platform; choose a new directory")
        if source_file is not None and file_digest(source_file) != SOURCE_SHA256:
            raise ValueError("Local source file differs from the pinned official source")
        if docker_client is not None:
            return record_images(output_dir, docker_client)
        return manifest
    if not download and source_file is None:
        raise FileNotFoundError("No prepared assets: explicitly use --download or --source-file")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".swebench-assets-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        source = staging / SOURCE_NAME
        if download:
            url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{DATASET_FILE}"
            with urlopen(url, timeout=120) as response, source.open("xb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        else:
            source.write_bytes(Path(source_file).read_bytes())
        if file_digest(source) != SOURCE_SHA256:
            raise ValueError("Downloaded/local source SHA256 differs from pinned official source")
        import pyarrow.parquet as pq
        tasks = pq.read_table(source).to_pylist()
        all_ids = _validate_tasks(tasks)
        selected = _select_ids(all_ids, instance_ids)
        (staging / TASKS_NAME).write_text(canonical_json(tasks) + "\n", encoding="utf-8")
        manifest = {
            "format": FORMAT, "benchmark": "swebench_verified", "split": "test",
            "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
            "harness_version": HARNESS_VERSION, "harness_commit": HARNESS_COMMIT,
            "architecture": architecture, "full_task_count": FULL_TASK_COUNT,
            "full_instance_ids": all_ids, "full_instance_ids_sha256": FULL_INSTANCE_IDS_SHA256,
            "instance_ids": selected, "num_records": len(selected),
            "is_full_verified": selected == all_ids,
            "scope": "full" if selected == all_ids else "debug",
            "ready_for_training": False,
            "source": {"path": SOURCE_NAME, "sha256": SOURCE_SHA256, "bytes": source.stat().st_size},
            "evaluator_source": {"path": TASKS_NAME, "sha256": file_digest(staging / TASKS_NAME),
                                 "visibility": "evaluator_only"},
            "images": {},
        }
        if docker_client is not None:
            for task_id in selected:
                reference = official_image_reference(task_id, architecture)
                manifest["images"][task_id] = {"reference": reference,
                    **_image_identity(docker_client, reference, architecture)}
        _write_manifest(staging, manifest)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents a concurrent preparer overwriting files.
        for name in (SOURCE_NAME, TASKS_NAME, MANIFEST_NAME, "manifest.sha256"):
            with (output_dir / name).open("xb") as stream:
                stream.write((staging / name).read_bytes())
    return manifest


def verify_assets(asset_dir, *, docker_client=None, require_images=True, require_full=False, instance_ids=None):
    """Verify local bytes, pinned identities, task roster and optionally images."""
    directory = Path(asset_dir)
    manifest_path = directory / MANIFEST_NAME
    if file_digest(manifest_path) != (directory / "manifest.sha256").read_text().strip():
        raise ValueError("Asset manifest SHA256 mismatch")
    manifest = read_json(manifest_path)
    expected = {"format": FORMAT, "benchmark": "swebench_verified", "split": "test",
                "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
                "harness_version": HARNESS_VERSION, "harness_commit": HARNESS_COMMIT,
                "full_task_count": FULL_TASK_COUNT, "full_instance_ids_sha256": FULL_INSTANCE_IDS_SHA256,
                "ready_for_training": False}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Asset identity mismatch: {key}")
    if manifest.get("architecture") not in ARCHITECTURES:
        raise ValueError("Unsupported asset architecture")
    source = manifest.get("source", {})
    if (source.get("path") != SOURCE_NAME or source.get("sha256") != SOURCE_SHA256
            or file_digest(directory / SOURCE_NAME) != SOURCE_SHA256
            or (directory / SOURCE_NAME).stat().st_size != source.get("bytes")):
        raise ValueError("Pinned source file identity mismatch")
    evaluator = manifest.get("evaluator_source", {})
    if (evaluator.get("path") != TASKS_NAME or evaluator.get("visibility") != "evaluator_only"
            or file_digest(directory / TASKS_NAME) != evaluator.get("sha256")):
        raise ValueError("Evaluator source file identity mismatch")
    all_ids = _validate_tasks(read_json(directory / TASKS_NAME))
    if manifest.get("full_instance_ids") != all_ids:
        raise ValueError("Full task roster mismatch")
    selected = _select_ids(all_ids, manifest.get("instance_ids", []))
    is_full = selected == all_ids
    if (selected != manifest["instance_ids"] or manifest.get("num_records") != len(selected)
            or manifest.get("is_full_verified") is not is_full
            or manifest.get("scope") != ("full" if is_full else "debug")):
        raise ValueError("Asset selection/full-scope identity mismatch")
    if require_full and not is_full:
        raise ValueError("Debug subset is not the full 500-task SWE-bench Verified evaluation")
    images = manifest.get("images")
    if not isinstance(images, dict) or set(images) - set(selected):
        raise ValueError("Invalid registered image task IDs")
    for task_id, image in images.items():
        _validate_image_record(task_id, image, manifest["architecture"])
    required_ids = _select_ids(selected, instance_ids)
    if require_images:
        missing = set(required_ids) - set(images)
        if missing:
            raise FileNotFoundError(f"Unregistered local task images: {sorted(missing)}; prepare explicitly")
        for task_id in required_ids:
            expected_image = images[task_id]
            actual = _image_identity(docker_client, expected_image["image_id"], manifest["architecture"])
            if any(actual[key] != expected_image[key] for key in ("image_id", "architecture", "os")):
                raise ValueError(f"Local Docker image identity mismatch: {task_id}")
            if not set(expected_image["repo_digests"]).issubset(actual["repo_digests"]):
                raise ValueError(f"Local Docker image digest mismatch: {task_id}")
    return manifest


def record_images(asset_dir, docker_client, *, instance_ids=None, image_map=None):
    """Register local images or explicit prepared derivatives, preserving pins.

    image_map maps task IDs to reference, upstream_image_id,
    upstream_repo_digests and a nonempty preparation identity object.
    Overrides never replace an existing registration; use a new asset directory.
    Preparation should record recipe/wheel hashes and fixed package versions.
    """
    manifest = verify_assets(asset_dir, require_images=False)
    selected = _select_ids(manifest["instance_ids"], instance_ids)
    if image_map is None:
        image_map = {}
    if not isinstance(image_map, dict) or set(image_map) - set(selected):
        raise ValueError("Image map keys must belong to the selected registration batch")
    updated = dict(manifest["images"])
    for task_id in selected:
        old = updated.get(task_id)
        override = image_map.get(task_id)
        if task_id in image_map:
            image = _prepared_image(task_id, override, docker_client, manifest["architecture"])
            if old is not None and image != old:
                raise ValueError(f"Refusing to replace registered image identity: {task_id}; use a new asset directory")
        else:
            reference = old["reference"] if old else official_image_reference(task_id, manifest["architecture"])
            lookup = old["image_id"] if old else reference
            image = {"reference": reference, **_image_identity(docker_client, lookup, manifest["architecture"])}
            if old is not None:
                if any(image[key] != old[key] for key in ("image_id", "architecture", "os")) or not set(old["repo_digests"]).issubset(image["repo_digests"]):
                    raise ValueError(f"Refusing to replace registered image identity: {task_id}")
        if old is None:
            updated[task_id] = image
    if updated != manifest["images"]:
        manifest["images"] = updated
        _write_manifest(asset_dir, manifest, replace=True)
    return manifest


def load_evaluator_tasks(asset_dir, *, instance_ids=None):
    """Trusted scorer only. Never pass these objects to policy/actor code."""
    manifest = verify_assets(asset_dir, require_images=False)
    selected = set(_select_ids(manifest["instance_ids"], instance_ids))
    return [task for task in read_json(Path(asset_dir) / TASKS_NAME) if task["instance_id"] in selected]


def load_public_tasks(asset_dir, *, instance_ids=None):
    """Trusted preparation only: project source into an agent-facing allowlist.

    This process reads evaluator data to authenticate the source. Policy/actor
    processes must instead load the separately exported public test.parquet.
    New upstream columns cannot leak into that public projection implicitly.
    """
    return [{key: task[key] for key in PUBLIC_FIELDS}
            for task in load_evaluator_tasks(asset_dir, instance_ids=instance_ids)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Explicit preparation; never pulls task images")
    prepare_parser.add_argument("--asset-dir", type=Path, required=True)
    prepare_parser.add_argument("--download", action="store_true")
    prepare_parser.add_argument("--source-file", type=Path)
    prepare_parser.add_argument("--instance-ids", nargs="+")
    prepare_parser.add_argument("--architecture", choices=ARCHITECTURES, default="x86_64")
    for command in ("register-images", "verify"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--asset-dir", type=Path, required=True)
        if command == "register-images":
            command_parser.add_argument("--instance-ids", nargs="+")
            command_parser.add_argument("--image-map", type=Path,
                                        help="Explicit prepared image references and upstream/preparation identities")
        else:
            command_parser.add_argument("--data-only", action="store_true", help="Does not certify execution readiness")
            command_parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_assets(args.asset_dir, instance_ids=args.instance_ids, download=args.download,
                                  source_file=args.source_file, architecture=args.architecture)
    else:
        client = None
        if args.command == "register-images" or not args.data_only:
            import docker
            client = docker.from_env()
        try:
            if args.command == "register-images":
                manifest = record_images(args.asset_dir, client, instance_ids=args.instance_ids,
                                         image_map=read_json(args.image_map) if args.image_map else None)
            else:
                if not args.data_only:
                    verify_harness_identity()
                manifest = verify_assets(args.asset_dir, docker_client=client,
                                         require_images=not args.data_only, require_full=args.require_full)
        finally:
            if client is not None:
                client.close()
    print(json.dumps({"scope": manifest["scope"], "num_records": manifest["num_records"],
                      "registered_images": len(manifest["images"]),
                      "manifest_sha256": file_digest(args.asset_dir / MANIFEST_NAME),
                      "local_asset_bytes": sum(path.stat().st_size for path in args.asset_dir.iterdir() if path.is_file())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
