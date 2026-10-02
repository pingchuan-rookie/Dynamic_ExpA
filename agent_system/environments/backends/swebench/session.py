"""Isolated SWE-bench session adapter; private tasks never cross the worker wire."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import sys


def memory_bytes(value):
    match = re.fullmatch(r"([1-9][0-9]*)([kmg]?)", str(value).lower())
    if match is None:
        raise ValueError("memory-limit must be a positive integer with optional k/m/g suffix")
    return int(match[1]) * (1024 ** {"": 0, "k": 1, "m": 2, "g": 3}[match[2]])


def network_evidence(config):
    evidence = {"worker_network": "not_checked", "python_executable": sys.executable}
    if config.get("require_network_isolation"):
        from agent_system.environments.env_package.swebench.offline import verify_network_isolation
        evidence["worker_network"] = verify_network_isolation()["driver_network"]
    return evidence


def public_snapshot(config, *, inspect_docker=False):
    from agent_system.environments.env_package.swebench.assets import file_digest, load_public_tasks, verify_assets, verify_harness_identity
    evidence = network_evidence(config)
    if config.get("mini_swe_agent") is not None:
        from agent_system.environments.env_package.swebench.mini_agent import MiniConfig, verify_installation
        mini = MiniConfig(**config["mini_swe_agent"])
        if mini.api_key_env:
            import os
            if not os.environ.get(mini.api_key_env):
                raise ValueError("mini-swe-agent API key environment variable is unset")
        evidence["mini_swe_agent_version"] = verify_installation()
    if config.get("expected_ray_runtime") is not None:
        from agent_system.environments.backends.swebench.worker import verify_ray_runtime
        evidence["ray_runtime"] = verify_ray_runtime(config["expected_ray_runtime"])
        heavy = [name for name in ("torch", "verl") if name in sys.modules]
        if heavy:
            raise RuntimeError(f"SWE-bench preflight imported policy-only libraries: {heavy}")
    client = None
    try:
        if inspect_docker:
            from agent_system.environments.env_package.swebench.envs import local_client
            client = local_client()
        manifest = verify_assets(config["assets"], require_images=False)
        selected_ids = config.get("task_ids")
        evaluation_selection = None
        if not config.get("debug") and selected_ids is None and config.get("limit", -1) == -1:
            from agent_system.utils.evaluation_protocol import selection_identity
            evaluation_selection = selection_identity("swebench_verified", manifest["full_instance_ids"])
            selected_ids = evaluation_selection["selected_task_ids"]
        verify_harness_identity()
        tasks = load_public_tasks(config["assets"], instance_ids=selected_ids)
        # Exact public fields; image IDs and harness evidence travel separately.
        fields = ("instance_id", "repo", "base_commit", "problem_statement")
        tasks = [{key: row[key] for key in fields} for row in tasks]
        limit = config.get("limit", -1)
        if limit != -1:
            tasks = tasks[:limit]
        if not tasks:
            raise ValueError("Empty SWE-bench task selection")
        manifest = verify_assets(config["assets"], docker_client=client, require_images=inspect_docker,
                                 instance_ids=[row["instance_id"] for row in tasks])
        from agent_system.environments.env_package.swebench.assets import MANIFEST_NAME
        source = {"manifest_sha256": file_digest(Path(config["assets"]) / MANIFEST_NAME),
                  "dataset_revision": manifest["dataset_revision"], "harness_commit": manifest["harness_commit"],
                  "architecture": manifest["architecture"], "is_full_verified": manifest["is_full_verified"],
                  "images": {row["instance_id"]: manifest["images"].get(row["instance_id"]) for row in tasks}}
        return {"format": "swebench_public_v1", "benchmark": "swebench_verified", "tasks": tasks,
                "source_identity": source, "images_verified": inspect_docker, "offline_evidence": evidence,
                "evaluation_selection": evaluation_selection}
    finally:
        if client is not None:
            client.close()


class BoundGrader:
    """Bind hidden tests to the trusted scorer, not the interactive session."""
    def __init__(self, grader, task):
        self.grader, self.task = grader, deepcopy(task)

    def score(self, public_task, patch):
        if public_task["instance_id"] != self.task["instance_id"]:
            raise ValueError("Grading task mismatch")
        return self.grader.score(self.task, patch)


class SwebenchSession:
    def __init__(self, **config):
        self.config = config
        self.session = self.client = self.result = None
        self.context = None

    def reset(self, **spec):
        from agent_system.environments.env_package.swebench.assets import file_digest, MANIFEST_NAME, load_evaluator_tasks, load_public_tasks, verify_assets
        from agent_system.environments.env_package.swebench.envs import RuntimeConfig, SweBenchEnv, local_client
        from agent_system.environments.env_package.swebench.grading import OfficialGrader
        if self.session is not None:
            raise RuntimeError("Close the previous SWE-bench session")
        self.offline_evidence = network_evidence(self.config)
        source = spec["source_identity"]
        if file_digest(Path(self.config["assets"]) / MANIFEST_NAME) != source["manifest_sha256"]:
            raise ValueError("Asset manifest changed since task planning")
        self.spec = {key: deepcopy(spec[key]) for key in ("benchmark", "instance_id", "trial", "seed", "episode_id", "source_identity")}
        if self.spec["benchmark"] != "swebench_verified" or not re.fullmatch(r"[a-f0-9]{24}", self.spec["episode_id"]):
            raise ValueError("Invalid SWE-bench episode identity")
        task_id = spec["instance_id"]
        public = load_public_tasks(self.config["assets"], instance_ids=[task_id])[0]
        public = {key: public[key] for key in ("instance_id", "repo", "base_commit", "problem_statement")}
        if public != spec["public_task"]:
            raise ValueError("Runtime public task differs from selected snapshot")
        self.client = local_client()
        manifest = verify_assets(self.config["assets"], docker_client=self.client, require_images=True,
                                 instance_ids=[task_id])
        image = manifest["images"][task_id]["image_id"]
        runtime_config = RuntimeConfig(max_steps=self.config["max_steps"], tool_timeout_s=self.config["command_timeout"],
            episode_timeout_s=self.config["episode_timeout"], test_timeout_s=self.config["grading_timeout"],
            cpus=self.config["environment_cpus"], memory_bytes=memory_bytes(self.config["memory_limit"]),
            pids_limit=self.config["pids_limit"], arch=manifest["architecture"])
        private = load_evaluator_tasks(self.config["assets"], instance_ids=[task_id])[0]
        # Separate from the episode directory created only after finalization.
        grading_dir = Path(self.config["output_dir"]).parent / "grading" / self.spec["episode_id"]
        grader = BoundGrader(OfficialGrader(self.client, image, runtime_config, artifact_dir=grading_dir), private)
        self.session = SweBenchEnv(public, image, config=runtime_config, docker_client=self.client, grader=grader,
            mini_agent=self.config.get("mini_swe_agent"),
            mini_artifact_dir=Path(self.config["output_dir"]).parent / "mini_swe_agent" / self.spec["episode_id"])
        self.context = self.session.start()
        return self._response(self.context)

    def _response(self, value):
        response = deepcopy(value)
        response["protocol"] = "native_tools"
        response["generation_seed"] = self.spec["seed"]
        if self.result is not None:
            response.update(result=deepcopy(self.result), episode_result=deepcopy(self.result), done=True)
        return response

    def step(self, action):
        self.context = self.session.step(action)
        return self._response(self.context)

    def finalize(self, reason):
        if self.result is None:
            self.result = {**self.session.finalize(reason), **deepcopy(self.spec),
                           "offline_evidence": deepcopy(self.offline_evidence)}
            # Keep the actual patch and official result even for failed scoring.
            directory = Path(self.config["output_dir"]) / self.spec["episode_id"]
            directory.mkdir(parents=True, exist_ok=False)
            patch = self.result.pop("patch", None)
            if patch is not None:
                patch_path = directory / "model.patch"
                patch_path.write_text(patch)
                import hashlib
                self.result.update(patch_path=str(patch_path), patch_sha256=hashlib.sha256(patch.encode()).hexdigest())
            from agent_system.environments.prompts.protocol import task_prompt_protocol
            self.result.update(protocol="native_tools", task_prompt_protocol=task_prompt_protocol("swebench_verified"),
                               history_length=self.config.get("history_length", 2))
            (directory / "result.json").write_text(json.dumps(self.result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        return self._response(self.context)

    def close(self):
        try:
            if self.session is not None:
                self.session.close()
        finally:
            if self.client is not None:
                self.client.close()
            self.session = self.client = None
        return {"ok": True}


def main():
    if sys.argv[1:] not in (["--check"], ["--preflight"]):
        raise ValueError("Expected --check or --preflight")
    config = json.load(sys.stdin)
    snapshot = public_snapshot(config, inspect_docker=sys.argv[1] == "--preflight")
    print(json.dumps(snapshot, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
