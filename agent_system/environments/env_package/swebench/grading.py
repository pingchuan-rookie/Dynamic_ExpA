"""Trusted, offline use of the pinned SWE-bench v4.1.0 official evaluator.

Do not call run_instance/build_container: those paths can pull or build images.
Only the candidate patch crosses from the agent container into this clean one.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import traceback

from .envs import DockerSandbox, RuntimeConfig, SweBenchInfrastructureError, validate_repository_base


def make_offline_test_spec(instance, *, arch="x86_64"):
    """Construct the official evaluation-only TestSpec without build-time I/O.

    v4.1.0 make_test_spec also generates environment installation scripts, which
    fetch requirements from GitHub even when all Docker images already exist.
    Those unused build scripts must not be generated in an offline evaluator.
    """
    from swebench.harness.constants import MAP_REPO_TO_EXT, MAP_REPO_VERSION_TO_SPECS
    from swebench.harness.test_spec.create_scripts import make_eval_script_list
    from swebench.harness.test_spec.test_spec import TestSpec

    repo, version = instance["repo"], instance["version"]
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    def tests(key):
        value = instance[key]
        value = json.loads(value) if isinstance(value, str) else deepcopy(value)
        if not isinstance(value, list) or any(not isinstance(test, str) for test in value):
            raise SweBenchInfrastructureError("Invalid official test reference list")
        return value
    return TestSpec(instance_id=instance["instance_id"], repo=repo, version=version,
                    repo_script_list=[], env_script_list=[],
                    eval_script_list=make_eval_script_list(instance, specs, "testbed", "/testbed",
                                                           instance["base_commit"], instance["test_patch"]),
                    arch=arch, FAIL_TO_PASS=tests("FAIL_TO_PASS"), PASS_TO_PASS=tests("PASS_TO_PASS"),
                    language=MAP_REPO_TO_EXT[repo], docker_specs=deepcopy(specs.get("docker_specs", {})),
                    namespace=None)


def guarded_eval_script(script):
    """Keep official commands verbatim, but fail closed outside the test phase.

    The upstream script deliberately omits errexit. Without a guard, failed
    installation/test-patch setup can still print both test markers and produce
    a misleading valid unresolved report. Test-command failures remain normal.
    """
    start = ": '>>>>> Start Test Output'"
    end = ": '>>>>> End Test Output'"
    lines = script.splitlines()
    if lines.count(start) != 1 or lines.count(end) != 1 or lines.index(start) >= lines.index(end):
        raise SweBenchInfrastructureError("Official eval script has unexpected test-phase markers")
    guarded = []
    for index, line in enumerate(lines):
        if line == start:
            guarded.append("set +e")
        guarded.append(line)
        if index == 1 or line == end:
            guarded.append("set -e")
    return "\n".join(guarded) + "\n"


class OfficialGrader:
    def __init__(self, docker_client, image, config=None, *, harness=None, artifact_dir=None):
        self.client, self.image = docker_client, image
        self.config = config if isinstance(config, RuntimeConfig) else RuntimeConfig(**dict(config or {}))
        self.harness = harness
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None

    def _harness(self):
        if self.harness is not None:
            return self.harness
        from .assets import verify_harness_identity
        verify_harness_identity()
        from swebench.harness.grading import get_eval_report
        return make_offline_test_spec, get_eval_report

    def score(self, instance, patch):
        """Return only a score summary. Raw hidden-test logs stay controller-side.

        Ordinary official test failures are valid unresolved attempts. Docker,
        timeout, output truncation and malformed/unavailable reports are not.
        """
        sandbox = None
        artifacts = None
        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            artifacts = Path(tempfile.mkdtemp(prefix="grade-", dir=self.artifact_dir))
        try:
            make_test_spec, get_eval_report = self._harness()
            spec = make_test_spec(deepcopy(instance), arch=self.config.arch)
            prediction = {"instance_id": instance["instance_id"],
                          "model_name_or_path": "dyad-offline", "model_patch": patch}
            if not isinstance(patch, str) or len(patch.encode()) > self.config.max_patch_bytes:
                raise SweBenchInfrastructureError("Invalid submission patch")
            sandbox = DockerSandbox(self.client, self.image, self.config, purpose="grade")
            validate_repository_base(sandbox, instance["base_commit"])
            if artifacts is not None:
                (artifacts / "prediction.json").write_text(json.dumps(prediction, indent=2))
                (artifacts / "eval.sh").write_text(spec.eval_script)
            sandbox.put_file("/tmp/model.patch", patch)
            if patch.strip():
                # Ordered v4.1.0 run_evaluation.GIT_APPLY_CMDS, without shell interpolation.
                commands = [["git", "apply", "--verbose", "/tmp/model.patch"],
                            ["git", "apply", "--verbose", "--reject", "/tmp/model.patch"],
                            ["patch", "--batch", "--fuzz=5", "-p1", "-i", "/tmp/model.patch"]]
                apply_results = []
                for index, command in enumerate(commands):
                    if index:
                        # --reject/patch may modify files before failing. Each
                        # upstream fallback receives a genuinely pristine image.
                        sandbox.close()
                        sandbox = DockerSandbox(self.client, self.image, self.config, purpose="grade")
                        validate_repository_base(sandbox, instance["base_commit"])
                        sandbox.put_file("/tmp/model.patch", patch)
                    applied = sandbox.exec(command)
                    apply_results.append({"command": command, "container_id": sandbox.inspection["Id"], **applied})
                    if applied["exit_code"] == 0:
                        break
                if artifacts is not None:
                    (artifacts / "patch_apply.json").write_text(json.dumps(apply_results, indent=2))
                if apply_results[-1]["exit_code"] != 0:
                    raise SweBenchInfrastructureError("Candidate patch could not be applied in a clean image")
            script = guarded_eval_script(spec.eval_script)
            sandbox.put_file("/eval.sh", script)
            inspection = sandbox.inspection
            if artifacts is not None:
                (artifacts / "eval_guarded.sh").write_text(script)
                (artifacts / "grading_container_inspect.json").write_text(json.dumps(inspection, indent=2))
            output = sandbox.exec(["/bin/bash", "/eval.sh"], timeout=self.config.test_timeout_s,
                                  max_bytes=self.config.max_test_output_bytes)
            if artifacts is not None:
                (artifacts / "test_output.txt").write_text(output["output"], encoding="utf-8")
                (artifacts / "execution.json").write_text(json.dumps({"exit_code": output["exit_code"], "truncated": output["truncated"]}))
            if output["truncated"]:
                raise SweBenchInfrastructureError("Official test output exceeded its byte budget")
            if output["exit_code"] != 0:
                raise SweBenchInfrastructureError("Official evaluation setup or cleanup failed")
            # Test-command failures are allowed inside the guarded test phase.
            # Official parsing, not that command's status, determines resolution.
            with tempfile.TemporaryDirectory(prefix="dyad-swe-grade-") as directory:
                log = Path(directory) / "test_output.txt"
                log.write_text(output["output"], encoding="utf-8")
                report = get_eval_report(test_spec=spec, prediction=prediction,
                                         test_log_path=str(log), include_tests_status=True)
            if artifacts is not None:
                (artifacts / "report.json").write_text(json.dumps(report, indent=2))
            item = report.get(instance["instance_id"])
            if not isinstance(item, dict) or not isinstance(item.get("resolved"), bool):
                raise SweBenchInfrastructureError("Official harness returned an invalid report")
            if item.get("patch_successfully_applied") is not True or not isinstance(item.get("tests_status"), dict):
                raise SweBenchInfrastructureError("Official harness could not complete test evaluation")
            return {"status": "completed", "metric_valid": True, "official_scored": True,
                    "official_reward": float(item["resolved"]), "resolved": item["resolved"],
                    "official_report_sha256": hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    "test_output_sha256": hashlib.sha256(output["output"].encode()).hexdigest(),
                    "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(), "image_id": self.image,
                    "grading_artifact_dir": str(artifacts) if artifacts is not None else None,
                    "grading_container_inspect": inspection,
                    "grading_container_inspect_sha256": hashlib.sha256(json.dumps(inspection, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    "eval_guard_version": "setup-errexit-v1",
                    "harness_version": "4.1.0", "harness_commit": "726c5461e2ef52d83cf1ea2107870a8bb3328d57"}
        except Exception as exc:
            # Trusted controller diagnostics never enter the agent context.
            if artifacts is not None:
                (artifacts / "error.json").write_text(json.dumps({"error_type": type(exc).__name__,
                    "error": str(exc), "traceback": traceback.format_exc()}, indent=2))
            if isinstance(exc, SweBenchInfrastructureError):
                raise
            raise SweBenchInfrastructureError("Official SWE-bench grading failed") from exc
        finally:
            if sandbox is not None:
                sandbox.close()
