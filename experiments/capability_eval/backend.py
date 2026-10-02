"""Pinned EvalScope adapters for single-turn capability_eval evaluation.

No model loading, agent loop, generated-code execution, or bespoke scorer lives here.
LiveCodeBench uses the pinned checker through an explicitly selected execution backend.
The scorer is identified explicitly; official checker parity is not assumed.
"""
from __future__ import annotations

from contextlib import contextmanager
import base64
import copy
import hashlib
import importlib.metadata
import io
import json
import pickle
import pickletools
import re
import zlib
from pathlib import Path

EVALSCOPE_COMMIT = "09b1be41d05f5996ad58296fb6c59aff28ee3ee9"
# release_v6 contains legitimate decoded private-test strings up to 198,722,655
# bytes. Keep a finite per-record cap above that pinned-data maximum.
MAX_PRIVATE_TEST_BYTES = 256 * 1024 * 1024
_TASKS = {
    "mmlu_pro": {
        "task": "mmlu_pro", "repo": "TIGER-Lab/MMLU-Pro",
        "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "split": "test", "fewshot_split": "validation", "fewshot": 5,
        "expected_count": 12032, "metric": "accuracy",
        "scorer": "EvalScope MMLUProAdapter/MultiChoiceAdapter",
        "files": ["data/test-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet"],
    },
    "hmmt26": {
        "task": "hmmt26", "repo": "MathArena/hmmt_feb_2026",
        "revision": "02fba4f74d8e68e73e66a02d540fd979c05c274c",
        "split": "train", "fewshot": 0, "expected_count": 33,
        "metric": "avg@4", "scorer": "EvalScope HMMT26Adapter numeric accuracy",
        "repeats": 4, "temperature": 0.6, "seed_policy": "base_seed_plus_repeat_index",
        "files": ["data/train-00000-of-00001.parquet"],
    },
    "livecodebench_v6": {
        "task": "live_code_bench", "repo": "livecodebench/code_generation_lite",
        "revision": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        "split": "test", "release": "release_v6", "fewshot": 0,
        "expected_count": 1055, "metric": "pass@1",
        "scorer": "Pinned EvalScope testing_util.run_test",
        "official_checker_parity_verified": False,
        "score_authenticity": "upstream checker shares candidate interpreter; not tamper-resistant",
        "execution_enabled": True,
        "files": ["test.jsonl", *[f"test{i}.jsonl" for i in range(2, 7)]],
    },
}


def protocol(benchmark: str | None = None) -> dict:
    """Return serializable protocol metadata, without importing evaluation packages."""
    tasks = copy.deepcopy(_TASKS if benchmark is None else _TASKS[benchmark])
    return {
        "framework": "evalscope", "framework_commit": EVALSCOPE_COMMIT,
        "single_turn": True,
        "model_tools": False, "repeats": _TASKS[benchmark].get("repeats", 1) if benchmark else {
            name: spec.get("repeats", 1) for name, spec in _TASKS.items()}, "tasks": tasks,
        "limit_semantics": "EvalScope limit is per subject/subset; not a full benchmark result",
    }


def _verify_install(name: str, commit: str) -> None:
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Missing {name}; install backend_requirements.txt in a separate evaluation environment") from exc
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != commit:
        raise RuntimeError(f"{name} must be installed from the pinned git commit {commit}")


class _DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("LiveCodeBench test payload cannot import or construct classes")

    def persistent_load(self, pid):
        raise ValueError("LiveCodeBench test payload cannot use persistent references")


def _private_tests(value: str) -> str:
    """Decode official compressed JSON strings without executable pickle globals."""
    try:
        tests = json.loads(value)
    except json.JSONDecodeError:
        compressed = base64.b64decode(value, validate=True)
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, MAX_PRIVATE_TEST_BYTES)
        if decoder.unconsumed_tail or not decoder.eof or decoder.unused_data:
            raise ValueError("Malformed or oversized LiveCodeBench private tests")
        # Only string serialization opcodes are necessary in the official dataset.
        # This also excludes extension-cache lookups and all object construction.
        allowed = {"PROTO", "FRAME", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8",
                   "UNICODE", "BINSTRING", "SHORT_BINSTRING", "STRING", "MEMOIZE",
                   "BINPUT", "LONG_BINPUT", "PUT", "STOP"}
        if any(op.name not in allowed for op, _, _ in pickletools.genops(raw)):
            raise ValueError("Executable or non-string opcode in LiveCodeBench test payload")
        payload = _DataOnlyUnpickler(io.BytesIO(raw)).load()
        if not isinstance(payload, str):
            raise ValueError("Expected a JSON string in LiveCodeBench private test payload")
        tests = json.loads(payload)
    if not isinstance(tests, list) or not tests:
        raise ValueError("LiveCodeBench private tests must be a nonempty list")
    if any(not isinstance(t, dict) or not {"input", "output", "testtype"} <= t.keys() for t in tests):
        raise ValueError("Malformed LiveCodeBench test case")
    return json.dumps(tests, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _prepare_dataset(benchmark: str, output_dir: Path) -> tuple[Path, dict]:
    """Download only pinned official data files; never execute HF dataset scripts."""
    from huggingface_hub import snapshot_download

    spec = _TASKS[benchmark]
    snapshot = Path(snapshot_download(
        repo_id=spec["repo"], repo_type="dataset", revision=spec["revision"],
        allow_patterns=spec["files"],
    ))
    destination = output_dir / "dataset"
    destination.mkdir(exist_ok=False)
    if benchmark == "livecodebench_v6":
        return _prepare_livecodebench(snapshot, destination, spec)
    hashes, splits = {}, {}
    for filename in spec["files"]:
        path = snapshot / filename
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            split = "validation" if "validation-" in filename else spec["split"]
            splits.setdefault(split, []).extend(pq.read_table(path).to_pylist())
        else:
            with path.open(encoding="utf-8") as stream:
                splits.setdefault("test", []).extend(json.loads(line) for line in stream if line.strip())
    rows = splits[spec["split"]]
    if len(rows) != spec["expected_count"]:
        raise ValueError(f"Pinned {spec['repo']} expected {spec['expected_count']} rows, got {len(rows)}")
    id_key = "problem_idx" if benchmark == "hmmt26" else "question_id"
    ids = [str(row[id_key]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate benchmark question IDs")
    if benchmark == "mmlu_pro" and len(splits["validation"]) != 70:
        raise ValueError("Pinned MMLU-Pro validation must contain 70 demonstrations")
    subset, extension = "default", "jsonl"
    for split, data in splits.items():
        path = destination / f"{subset}_{split}.{extension}"
        _write_jsonl(path, data)
    # DefaultDataAdapter uses HF's local builder, not LocalDataLoader. Explicit
    # data-only HF metadata keeps config and split discovery exact.
    card = ["---", "configs:", f"- config_name: {subset}", "  data_files:"]
    for split in splits:
        card.extend([f"  - split: {split}", f"    path: {subset}_{split}.{extension}"])
    (destination / "README.md").write_text("\n".join([*card, "---", ""]), encoding="utf-8")
    metadata = {
        "repo": spec["repo"], "revision": spec["revision"], "source_sha256": hashes,
        "count": len(rows), "ordered_ids": ids,
        "ordered_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
    }
    return destination, metadata


def _prepare_livecodebench(snapshot, destination, spec):
    """Bound memory by one problem instead of retaining the decoded release."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    hashes, ids, writer = {}, [], None
    output = destination / "release_v6_test.parquet"
    try:
        for filename in spec["files"]:
            path = snapshot / filename
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[filename] = digest.hexdigest()
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    ids.append(str(row["question_id"]))
                    row["private_test_cases"] = _private_tests(row["private_test_cases"])
                    table = pa.Table.from_pylist([row])
                    if writer is None:
                        writer = pq.ParquetWriter(output, table.schema)
                    writer.write_table(table)
        if len(ids) != spec["expected_count"] or len(ids) != len(set(ids)):
            raise ValueError("Pinned LiveCodeBench has unexpected count or duplicate question IDs")
    finally:
        if writer is not None:
            writer.close()
    (destination / "README.md").write_text(
        "---\nconfigs:\n- config_name: release_v6\n  data_files:\n"
        "  - split: test\n    path: release_v6_test.parquet\n---\n", encoding="utf-8")
    return destination, {
        "repo": spec["repo"], "revision": spec["revision"], "source_sha256": hashes,
        "count": len(ids), "ordered_ids": ids,
        "ordered_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
    }


def validate_sandbox(value: dict | None) -> dict:
    from checker import validate_config
    return validate_config(value)


def _register_livecodebench(config):
    from evalscope.api.registry import BENCHMARK_REGISTRY, register_benchmark
    from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter
    from evalscope.api.metric import Score
    from checker import score, execution_method
    import importlib.util

    name = "capability_livecodebench_v6"
    metadata = copy.deepcopy(BENCHMARK_REGISTRY["live_code_bench"])
    metadata.name = name
    metadata.pretty_name = "LiveCodeBench v6 (pinned checker)"
    checker = Path(importlib.util.find_spec(
        "evalscope.benchmarks.live_code_bench.testing_util").origin)
    sandbox = validate_sandbox(config.get("sandbox_config"))

    class IsolatedLiveCodeBenchAdapter(LiveCodeBenchAdapter):
        def match_score(self, original_prediction, filtered_prediction, reference, task_state):
            passed, details = score(sandbox, filtered_prediction,
                                    task_state.metadata["evaluation_sample"], checker,
                                    test_timeout=self.review_timeout)
            return Score(value={"acc": passed}, main_score_name="acc",
                         extracted_prediction=filtered_prediction, prediction=original_prediction,
                         explanation=f"Pinned checker via {execution_method(sandbox)}", metadata=details)

    # One in-process backend invocation owns this registration. Updating the
    # custom metadata on subsequent runs does not patch the upstream adapter.
    if name in BENCHMARK_REGISTRY:
        metadata.data_adapter = IsolatedLiveCodeBenchAdapter
        BENCHMARK_REGISTRY[name] = metadata
    else:
        register_benchmark(metadata)(IsolatedLiveCodeBenchAdapter)
    return name


def _register_hmmt():
    """Keep the upstream prompt/scorer, with a distinct deterministic seed per trial."""
    from evalscope.api.registry import BENCHMARK_REGISTRY
    from evalscope.benchmarks.hmmt.hmmt26_adapter import HMMT26Adapter

    class RepeatedHMMT26Adapter(HMMT26Adapter):
        def _on_inference(self, model, sample):
            if type(sample.id) is not int or type(model.config.seed) is not int:
                raise ValueError("HMMT avg@4 requires an indexed sample and integer base seed")
            generation = model.config.model_copy(deep=True)
            generation.seed += sample.id % self.repeats
            return model.generate(input=sample.input, tools=sample.tools, config=generation)

    metadata = copy.deepcopy(BENCHMARK_REGISTRY["hmmt26"])
    metadata.data_adapter = RepeatedHMMT26Adapter
    BENCHMARK_REGISTRY["hmmt26"] = metadata


def preflight(config: dict) -> dict:
    """Check local dependencies and enforce execution gates without downloading data."""
    benchmark = config["benchmark"]
    metadata = protocol(benchmark)
    if config.get("generate_only"):
        raise RuntimeError("Pinned EvalScope run_task has no generation-only phase; refusing to score under a generation-only label")
    if benchmark == "livecodebench_v6":
        from checker import preflight as checker_preflight, execution_method
        sandbox = validate_sandbox(config.get("sandbox_config"))
        metadata["sandbox_identity"] = checker_preflight(sandbox)
        metadata["execution_backend"] = execution_method(sandbox)
    _verify_install("evalscope", EVALSCOPE_COMMIT)
    from evalscope.config import TaskConfig
    from evalscope.api.registry import get_benchmark
    if benchmark == "hmmt26":
        _register_hmmt()
    task_name = _register_livecodebench(config) if benchmark == "livecodebench_v6" else _TASKS[benchmark]["task"]
    if benchmark == "livecodebench_v6":
        from checker import score, CheckerInfrastructureError
        import importlib.util
        checker = Path(importlib.util.find_spec("evalscope.benchmarks.live_code_bench.testing_util").origin)
        passed, _ = score(sandbox, "print(2)",
                          json.dumps({"inputs": [""], "outputs": ["2\n"], "fn_name": None}), checker)
        if not passed:
            raise CheckerInfrastructureError("Pinned checker readiness fixture failed")
    task = TaskConfig(datasets=[task_name], model=config["model"],
                      eval_type="openai_api", api_url=config["api_url"],
                      api_key=config.get("api_key") or "EMPTY", judge={"strategy": "rule"})
    adapter = get_benchmark(task_name, task)
    if adapter.benchmark_meta.name != task_name:
        raise RuntimeError("Pinned EvalScope task registry mismatch")
    return metadata


@contextmanager
def isolated_dataset_cache(output_dir: Path):
    """Keep derived local Arrow caches scoped to one evaluation run.

    HF local builders can share a cache key for identically named dataset
    directories with identical cards, even when their Parquet rows differ.
    Changing only HF_DATASETS_CACHE in the environment is insufficient after
    datasets has been imported by evaluator preflight.
    """
    import datasets

    previous = datasets.config.HF_DATASETS_CACHE
    cache = Path(output_dir).resolve() / "hf_datasets_cache"
    cache.mkdir(exist_ok=False)
    datasets.config.HF_DATASETS_CACHE = cache
    try:
        yield cache
    finally:
        datasets.config.HF_DATASETS_CACHE = previous


def run(config: dict, output_dir: Path):
    """Run one pinned benchmark against a pre-existing OpenAI-compatible server.

    sandbox_config explicitly selects Docker or the image-integrated process checker.
    MMLU-Pro's limit applies per subject, as in EvalScope, not globally.
    Return the upstream result; predictions, reviews and reports remain under output_dir.
    """
    benchmark = config["benchmark"]
    spec = _TASKS[benchmark]
    print(f"[capability_eval][{benchmark}] phase=preflight", flush=True)
    checked_protocol = preflight(config)
    _verify_install("evalscope", EVALSCOPE_COMMIT)
    generation = copy.deepcopy(config["generation_config"])
    if generation.get("n", 1) != 1:
        raise ValueError("capability_eval backend requires one completion per request (n=1); repetitions use TaskConfig.repeats")
    if any(key in generation for key in ("tools", "tool_choice", "functions", "function_call")):
        raise ValueError("capability_eval generation must not provide model tools")
    extra = generation.get("extra_body") or {}
    if any(key in extra for key in ("tools", "tool_choice", "functions", "function_call")):
        raise ValueError("capability_eval generation must not provide model tools")
    generation["n"] = 1
    if config.get("limit") is not None and (type(config["limit"]) is not int or config["limit"] < 1):
        raise ValueError("limit must be None or a positive integer")
    from evalscope import run_task
    from evalscope.config import TaskConfig

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[capability_eval][{benchmark}] phase=preparing_dataset", flush=True)
    dataset, dataset_metadata = _prepare_dataset(benchmark, output_dir)
    task_args = {"dataset_id": str(dataset), "dataset_revision": spec["revision"],
                 "eval_split": spec["split"], "few_shot_num": spec["fewshot"], "shuffle": False}
    if benchmark == "livecodebench_v6":
        task_args.update(subset_list=["release_v6"], extra_params={"start_date": None, "end_date": None, "debug": False})
    task_name = _register_livecodebench(config) if benchmark == "livecodebench_v6" else spec["task"]
    task = TaskConfig(
        datasets=[task_name], dataset_args={task_name: task_args}, dataset_hub="local",
        model=config["model"], api_url=config["api_url"], api_key=config.get("api_key") or "EMPTY",
        eval_type="openai_api", generation_config=generation,
        limit=config.get("limit"), eval_batch_size=config["eval_batch_size"], repeats=checked_protocol["repeats"],
        work_dir=str(output_dir / "evalscope"),
        dataset_dir=str(output_dir / "evalscope_dataset_cache"),
        no_timestamp=True, ignore_errors=False, enable_progress_tracker=True, collect_perf=False,
        judge={"strategy": "rule"}, sandbox=None,
    )
    record = checked_protocol
    record["dataset"] = dataset_metadata
    record["status"] = "running"
    record["full_benchmark"] = config.get("limit") is None
    manifest = output_dir / "backend_protocol.json"
    manifest.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    try:
        from progress import BenchmarkProgress
        with BenchmarkProgress(benchmark, output_dir / "evalscope"), isolated_dataset_cache(output_dir):
            result = run_task(task_cfg=task)
        report_dir = output_dir / "evalscope" / "reports"
        reports = [p for p in report_dir.rglob("*.json") if p.stat().st_size > 0]
        if not result or not reports:
            raise RuntimeError("EvalScope produced no evaluation result/report; run is invalid")
        benchmark_reports = []
        for path in reports:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("dataset_name") == task_name:
                benchmark_reports.append(payload)
        if not benchmark_reports:
            raise RuntimeError("No report for the requested benchmark")
        for payload in benchmark_reports:
            execution = payload.get("execution_summary") or {}
            if (not payload.get("metrics") or not payload.get("primary_metric_identity")
                    or payload.get("primary_metric_unavailable_reason")
                    or execution.get("incomplete") or execution.get("errored", 0)
                    or execution.get("requested", 0) < 1
                    or execution.get("requested") != execution.get("succeeded")):
                raise RuntimeError("EvalScope report has incomplete execution or no primary metric")
        record["status"] = "completed"
        print(f"[capability_eval][{benchmark}] phase=completed reports_validated=true", flush=True)
        return result
    except BaseException:
        record["status"] = "failed"
        print(f"[capability_eval][{benchmark}] phase=failed", flush=True)
        raise
    finally:
        manifest.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
