"""Where Alignment talks to Weights & Biases, and what a Alignment run looks like on the dashboard.

Two processes publish to the **same** run.
`agent_system/policies/dyad/training/action_encoder_alignment/actenc_alignment_train.py` opens it and streams the
curve
one evaluation point at a time, so the dashboard follows the run while it trains rather than after
it. `experiments/dyad_training/action_encoder_alignment/analysis/actenc_alignment_upload_wandb.py` reopens it
afterwards and finishes it:
the figure, the breakdown tables, and the headline summary -- none of which exist until training is
over. `analysis/actenc_alignment_verify_wandb.py` then reads the whole thing back from the API and compares it with
what is on disk.

Everything those callers have to agree on lives here: the project, the run id, which
`metrics.jsonl` field becomes which series, the `wandb.init` arguments. Split across the callers,
a renamed series gets renamed on one side only, and the dashboard grows two half-length curves
with nothing saying which one is current.

Two asymmetries are deliberate and are the reason this module is shaped the way it is.

**Online is forced, never inherited.** `offline` is a silent failure: it does not raise, it just
uploads nothing, and you find out when you go looking for the curve and the GPU hours are already
spent. An unwanted online run is merely noise in the project. The costs are not comparable, so
`init_kwargs` hard-codes the mode. AGENTS.md and `lucia_job/RULES.md` C1.4 record the same rule for
Agentic RL, where `scripts/run_layout.sh` *overwrites* `WANDB_MODE` rather than defaulting it.

**Opening the run may be fatal; writing to it is not.** `LiveWandb.start` runs before the first
training step, where dying costs no GPU time and catches a credential that C1.4's preflight
somehow missed. `LiveWandb.log_row` runs in the middle of a three-hour job, where killing
everything over one 5xx is a bad trade -- the gap it leaves is what `actenc_alignment_verify_wandb.py` W2/W3 exist
to catch, point by point, against `metrics.jsonl`.
"""

from __future__ import annotations

import netrc
import os
from pathlib import Path
from typing import Any

from agent_system.policies.dyad.data.actenc_alignment_parquet import SPLIT_POLICY

DEFAULT_PROJECT = "dynamic-dyad-alignment"

#: How many `log_row` failures get their own warning before the rest are counted silently. A
#: network that is down stays down, and one warning per evaluation point for three hours buries
#: the training log in the same message.
_MAX_LOG_WARNINGS = 3

#: Which fields of a `metrics.jsonl` row become which wandb series. Written out rather than
#: derived, so a renamed field fails loudly here instead of silently publishing a chart with a
#: missing line.
HISTORY_KEYS = {
    "train/loss": lambda row: row.get("train_loss"),
    "gate": lambda row: row.get("gate"),
    "projector_l2_displacement": lambda row: row.get("projector_l2_displacement"),
}


LEGACY_TEST = "legacy-test-as-validation"
LEGACY_VAL = "legacy-val-unseen"


def metric_protocol(record: dict[str, Any]) -> str:
    """Disambiguate historical test-as-val from explicit new validation and final test."""
    policy, role = record.get("split_policy"), record.get("evaluation_role")
    splits = {"test", "val", "unseen"} & record.keys()
    if policy == SPLIT_POLICY:
        expected = {"train-validation": {"val"}, "final-test": {"test"}}.get(role)
        related = {key.split("_", 1)[0] for key in record
                   if key.startswith(("test_", "val_", "unseen_"))}
        if expected is not None and splits == expected and related <= expected:
            return role
    elif policy is None and role is None:
        if splits == {"test"}:
            return LEGACY_TEST
        if splits in ({"val"}, {"val", "unseen"}):
            return LEGACY_VAL
    raise ValueError("Alignment split semantics require an explicit train-validation/final-test protocol; "
                     "legacy test-as-val and historical val/unseen cannot be mixed")


def selection_split(record: dict[str, Any]) -> str:
    """On-disk scored split (final-test is evaluation only, never checkpoint selection)."""
    return "test" if metric_protocol(record) in (LEGACY_TEST, "final-test") else "val"


def published_name(local_name: str, record: dict[str, Any]) -> str:
    """Only legacy test is renamed. Explicit final-test always remains test."""
    if metric_protocol(record) == LEGACY_TEST:
        return {"test": "val", "test_by_form": "val_by_form",
                "test_by_size": "val_by_size"}.get(local_name, local_name)
    return local_name


def breakdown_groups(record: dict[str, Any]) -> list[str]:
    metric_protocol(record)
    return [name for name in record if "_by_" in name and name.startswith(("test_", "val_", "unseen_"))]


EVALUATION_METRICS = (
    "cross_entropy", "top1_accuracy", "demonstrated_action_probability",
    "chance_accuracy", "chance_cross_entropy",
)


NOTES = (
    "Alignment of dynamic_dyad_training_strategy_mcp_updated.md: policy LM and encoder LM frozen, "
    "projector trained with cross-entropy over C_t. The curve is streamed by actenc_alignment_train.py as it "
    "trains; the figure, the breakdown tables and the summary are added afterwards by "
    "experiments/dyad_training/action_encoder_alignment/analysis/actenc_alignment_upload_wandb.py."
)


def _print(message: str) -> None:
    print(f"[alignment-wandb] {message}", flush=True)


#: Replaced by `set_logger`. Indirection rather than a plain `print` because the trainer's log has
#: two properties this module cannot reproduce: it is chief-only, and it goes through `tqdm.write`
#: while the progress bar is alive. A bare `print` from here during training shreds the bar.
_log_to = _print


def set_logger(sink) -> None:
    """Send this module's output through the caller's logger."""
    global _log_to
    _log_to = sink


def log(message: str) -> None:
    _log_to(message)


def run_id(run_name: str) -> str:
    """The wandb run id for the run directory named `run_name`.

    Deriving it from the directory name is what lets the trainer and the uploader find the same
    run without passing an id between two processes that are minutes apart. It is also why the id
    is not disposable: it names the checkpoint on disk, so it has to stay usable. See the note
    above `find_run`.
    """
    return f"alignment-{run_name}"


def history_payload(row: dict[str, Any]) -> dict[str, Any]:
    """One `metrics.jsonl` row -> one `wandb.log` payload.

    NaN is dropped rather than sent: step 0's `train_loss` is NaN by construction (nothing has
    been trained yet) and a NaN in the history makes the whole series unplottable on the
    dashboard, not just that point.
    """
    split = selection_split(row)
    payload = {name: fn(row) for name, fn in HISTORY_KEYS.items()}
    payload.update({f"{published_name(split, row)}/{metric}": row[split][metric]
                    for metric in EVALUATION_METRICS})
    payload = {k: v for k, v in payload.items() if v is not None and v == v}
    payload["step"] = row["step"]
    payload["epoch"] = row["epoch"]
    return payload


def run_config(config_yaml: dict[str, Any]) -> dict[str, Any]:
    """`config.yaml`'s contents -> the run's wandb config.

    Takes the same dictionary `actenc_alignment_train.py` writes to `config.yaml`, so the panel's config and the
    file cannot drift: there is one dictionary, written to two places.
    """
    config = dict(config_yaml.get("args") or {})
    config.update({
        "trainable_parameters": config_yaml.get("trainable_parameters"),
        "trainable_names": config_yaml.get("trainable_names"),
        "policy_hidden": config_yaml.get("policy_hidden"),
        "encoder_hidden": config_yaml.get("encoder_hidden"),
        "stage": "alignment",
    })
    for key in ("split_policy", "evaluation_role", "dataset_identity", "checkpoint_identity"):
        if key in config_yaml:
            config[key] = config_yaml[key]
    return config


def init_kwargs(run_name: str, config: dict[str, Any], project: str = DEFAULT_PROJECT,
                *, resume: str) -> dict[str, Any]:
    """The `wandb.init` arguments, identical for the trainer and the uploader but for `resume`.

    `resume` is the one thing the two callers must NOT share, and getting it wrong is silent:

        "never"   the trainer, and `--rebuild`. Means "this id is starting from nothing".
        "allow"   the uploader's normal path: land on the run the trainer opened, or create it
                  when there was no trainer to open it (`--wandb off`, or an older run directory).

    `"allow"` in the trainer's place is the dangerous one. A second run under the same `RUN_NAME`
    would resume the previous one and append to its history, and since wandb requires `step` to be
    non-decreasing the new curve either lands past the old one or is dropped -- leaving a single
    run whose history came from two different trainings, with nothing saying so.

    A note on `"never"`, because the observed behaviour and the documented one differ in wandb
    0.28.2. Documented: "If a run with the same ID already exists, it will result in failure."
    Observed: the existing run is reset -- history cleared, `lastHistoryStep` back to -1 -- and
    init succeeds. **Both outcomes are fine here**, which is why this depends on neither. Under
    the documented behaviour a `FORCE=1` rerun of an existing `RUN_NAME` stops and asks for a new
    name, which is correct; under the observed one it starts the run over, which is also correct.
    Only `"allow"` produces a wrong answer, and that is the one thing this rules out.
    """
    if resume not in ("never", "allow", "must"):
        raise ValueError(f"resume must be never, allow or must; got {resume!r}")
    return {
        "project": project,
        "name": run_name,
        "id": run_id(run_name),
        "resume": resume,
        "mode": "online",
        "config": config,
        "job_type": "alignment-test" if config.get("evaluation_role") == "final-test" else "alignment-training",
        "tags": ["alignment", "action-encoder", "frozen-lms"],
        "notes": NOTES,
    }


def define_metrics(handle: Any, role: str = "train-validation") -> None:
    """Plot the curves against `step` rather than wandb's own `_step` counter.

    Without this the figure, the breakdown tables and every future `log` that carries no `step`
    field would still advance the x axis, and the curve would end some way past its last real
    evaluation point.
    """
    handle.define_metric("test/*" if role == "final-test" else "val/*", step_metric="step")
    if role != "final-test":
        handle.define_metric("train/*", step_metric="step")


def resolve_api_key() -> tuple[str, str] | None:
    """`(key, where it came from)`, or None when there is no credential anywhere.

    Two sources, matching the two ways Alignment runs: cluster jobs get `WANDB_API_KEY` from the
    launch yaml, local docker gets the host's `~/.netrc` mounted in. Both are resolved to an
    explicit key here and handed to `wandb.login`; wandb is never left to find one itself,
    because the path it takes when it cannot is to ask standard input, and an unattended job that
    reaches that prompt hangs until the cluster kills it.

    Returning None means "no credential", which is a decision for the caller: `actenc_alignment_train.py --wandb
    auto` treats it as "do not stream", the entrypoint treats it as fatal.
    """
    key = os.environ.get("WANDB_API_KEY", "").strip()
    if key:
        return key, "WANDB_API_KEY"

    path = Path("~/.netrc").expanduser()
    if not path.exists():
        return None
    host = os.environ.get("WANDB_HOST_MACHINE", "api.wandb.ai")
    try:
        auth = netrc.netrc(str(path)).authenticators(host)
    except Exception as exc:                                        # noqa: BLE001
        # A malformed ~/.netrc is not the same as an absent one: it means someone meant to
        # provide a credential. Raising keeps that distinguishable from "no credential".
        raise RuntimeError(f"{path} cannot be parsed: {type(exc).__name__}: {exc}") from exc
    if not auth or not auth[2]:
        return None
    return auth[2], str(path)


def default_entity() -> str:
    """The account's default entity, which every run path is built from."""
    import wandb

    entity = wandb.Api().default_entity
    if not entity:
        raise RuntimeError("this wandb account has no default_entity; run paths cannot be built")
    return entity


def find_run(run_name: str, project: str = DEFAULT_PROJECT) -> Any | None:
    """The server's copy of this run, or None if it does not have one."""
    import wandb

    path = f"{default_entity()}/{project}/{run_id(run_name)}"
    try:
        return wandb.Api().run(path)
    except Exception:                                               # noqa: BLE001
        # The public API raises for "no such run" and for "cannot reach the server" alike. The
        # callers treat both the same way -- there is nothing to resume onto -- and the one that
        # cannot afford to guess (`actenc_alignment_upload_wandb.py`) fails on the subsequent init.
        return None


def history_steps(run: Any, namespace: str = "val") -> list[int]:
    """The `step` values the server holds for this run, in order.

    Fetched whole and filtered here rather than with `scan_history(keys=[...])`: asking the server
    to project the columns fails on this project with "Step column '_step' not found in schema".
    Only rows carrying val CE count; tables are not curve points.
    Old test-prefixed curves must not be mistaken for a completed val upload.
    """
    return [h["step"] for h in run.scan_history()
            if h.get(f"{namespace}/cross_entropy") is not None and h.get("step") is not None]


# **Nothing here deletes a run, and nothing should.** Deleting one and recreating it under the
# same id is the obvious-looking way to republish, and it is not available: wandb retires the id
# along with the run, so a later init gets
#
#     CommError: run alignment-<name> was previously created and deleted; try a new run id
#
# A delete is therefore not "start over", it is "burn this name forever" -- and the name comes
# from the run directory, which is what the checkpoint on disk is called. Republishing uses
# `resume="never"` instead (see `init_kwargs`), which costs nothing when it does not work.


class LiveWandb:
    """The training process's end of the run: history rows, as they are computed.

    Every method is a no-op on a disabled instance, so `actenc_alignment_train.py` has one code path whether or not
    wandb is on. Disabled is the normal state on every replica but the chief -- several ranks
    initialising the same run id fight over it -- and on any local run without a credential.

    Failures after `start` are warnings, never exceptions. See the module docstring for why the
    asymmetry with `start` is deliberate rather than an oversight.
    """

    def __init__(self, handle: Any | None) -> None:
        self._handle = handle
        self._logged = 0
        self._failures = 0

    @property
    def enabled(self) -> bool:
        return self._handle is not None

    @property
    def url(self) -> str | None:
        return self._handle.url if self._handle is not None else None

    @classmethod
    def start(cls, *, run_name: str, config: dict[str, Any], mode: str = "auto",
              enabled: bool = True, project: str = DEFAULT_PROJECT) -> "LiveWandb":
        """Open the run, or return a disabled instance.

        `mode` is the `--wandb` flag:

            off     never open one
            auto    open one when there is a credential, otherwise say so and carry on. This is
                    the default because a local `train.sh` on a machine that has never
                    logged in should train, not stop.
            on      open one or raise. Cluster jobs pass this: the entrypoint has already verified
                    the login (C1.4), so a failure here is a real problem and the cheapest place
                    to find out is before the first step.
        """
        if not enabled or mode == "off":
            return cls(None)
        if mode not in ("auto", "on"):
            raise ValueError(f"--wandb must be auto, on or off; got {mode!r}")

        def give_up(reason: str) -> "LiveWandb":
            if mode == "on":
                raise RuntimeError(f"--wandb on, but {reason}")
            log(f"not streaming to wandb: {reason}")
            log("  the run is still recorded -- experiments/dyad_training/action_encoder_alignment/analysis/"
                "actenc_alignment_upload_wandb.py replays metrics.jsonl after training")
            return cls(None)

        offline = os.environ.get("WANDB_MODE") == "offline"
        source = "offline"
        if not offline:
            try:
                credential = resolve_api_key()
            except RuntimeError as exc:
                return give_up(str(exc))
            if credential is None:
                return give_up("there is no WANDB_API_KEY and no ~/.netrc entry for this host")
            key, source = credential
        try:
            import wandb

            if not offline:
                wandb.login(key=key, relogin=True, timeout=30)
            # `never`, not `allow`: this is a run starting from nothing, exactly as the caller has
            # just truncated its metrics.jsonl. See `init_kwargs` for what each outcome means when
            # the id is already taken -- both of them are acceptable, `allow` is the one that is
            # not.
            kwargs = init_kwargs(run_name, config, project, resume="never")
            if offline:
                kwargs.update(mode="offline", resume=None)
            handle = wandb.init(**kwargs)
            define_metrics(handle)
        except Exception as exc:                                    # noqa: BLE001
            return give_up(f"{type(exc).__name__}: {exc} (credential from {source})")

        log(f"streaming to {handle.url} (credential from {source})")
        return cls(handle)

    def log_row(self, row: dict[str, Any]) -> None:
        """Publish one `metrics.jsonl` row. Called after the row is on disk, never before."""
        if self._handle is None:
            return
        try:
            self._handle.log(history_payload(row), step=row["step"])
            self._logged += 1
        except Exception as exc:                                    # noqa: BLE001
            self._failures += 1
            if self._failures <= _MAX_LOG_WARNINGS:
                log(f"WARN step {row['step']} did not reach wandb: {type(exc).__name__}: {exc}")
                if self._failures == _MAX_LOG_WARNINGS:
                    log("WARN further upload failures will be counted, not printed")

    def finish(self) -> None:
        """Close the run. Reports what did not make it, so the log says so before the gate does."""
        if self._handle is None:
            return
        if self._failures:
            log(f"WARN {self._failures} of {self._logged + self._failures} points did not reach "
                f"wandb. actenc_alignment_verify_wandb.py W2/W3 will fail on this run; re-publish it with "
                f"actenc_alignment_upload_wandb.py --rebuild")
        try:
            self._handle.finish()
            log(f"{self._logged} points published")
        except Exception as exc:                                    # noqa: BLE001
            log(f"WARN wandb.finish failed: {type(exc).__name__}: {exc}")
        finally:
            self._handle = None
