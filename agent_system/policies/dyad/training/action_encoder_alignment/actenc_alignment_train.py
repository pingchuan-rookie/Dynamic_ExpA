"""Alignment training: fit the projector to demonstrated action choices.

    python -m agent_system.policies.dyad.training.action_encoder_alignment.actenc_alignment_train --out
    /absolute/artifacts/outputs/local/alignment/<model>_<time>

Section 2.1 of the strategy document says which parameters move; this file is where that becomes
enforceable rather than intended. Two assertions run before the first step:

  1. the set of parameters with `requires_grad` is exactly the projector's;
  2. the optimizer's parameter groups cover exactly that set.

Both, because either one alone passes in a configuration that trains the wrong thing. Freezing the
policy LLM backbone and then handing `model.parameters()` to the optimizer trains nothing extra but reports a
parameter count two orders of magnitude too high; conversely, an optimizer built from the head
while something else was left unfrozen accumulates gradients forever on a tensor nobody steps. Both
run to completion, and every metric looks ordinary.

The direct head projects pooled encoder representations into action rows.
Its trainable parameters share a single optimizer group at --lr.

Every row written to `metrics.jsonl` is also streamed to wandb as it is computed, so the dashboard
follows the run instead of appearing at the end of it. This half of the publishing is the curve
only; the figure, the breakdown tables and the summary are added afterwards by
`experiments/dyad_training/action_encoder_alignment/analysis/actenc_alignment_upload_wandb.py`. Where the line between
them falls, and why
opening the run may be fatal while writing to it is not, is in
`agent_system/policies/dyad/training/action_encoder_alignment/actenc_alignment_wandb_run.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml

from agent_system.policies.dyad.algorithms.action_selection import alignment_loss
from agent_system.policies.dyad.data.actenc_alignment_dataset import batches, load_split
from agent_system.policies.dyad.data.actenc_alignment_parquet import SPLIT_POLICY
from agent_system.policies.dyad.models.actenc_alignment_model import AlignmentActionSelector, AlignmentConfig
from agent_system.policies.dyad.training.action_encoder_alignment import actenc_alignment_config as cfg
from agent_system.policies.dyad.training.action_encoder_alignment import actenc_alignment_wandb_run as wandb_run
from agent_system.policies.dyad.training.action_encoder_alignment.actenc_alignment_evaluation import (
    by_group,
    evaluate,
    file_identity,
)
from agent_system.policies.dyad.training.action_encoder_alignment.actenc_alignment_wandb_run import LiveWandb

DEFAULT_POLICY_MODEL = "Qwen/Qwen3.5-2B"


def setup_replicas() -> tuple[int, int]:
    """`(rank, world_size)`. Data-parallel over replicas, each replica owning two GPUs.

    Alignment holds two frozen models and trains a projector between them, so one replica is
    (policy LLM backbone, encoder LLM backbone) on a card each -- see
    `agent_system/policies/dyad/models/actenc_alignment_model.py`. A four-GPU box therefore
    fits **two** replicas, and running one replica there leaves half the machine idle.

    Started by `torchrun --nproc_per_node=N`; without it this returns (0, 1) and every path below
    degrades to exactly the single-process behaviour it had before.

    NCCL rather than gloo: the only thing that crosses the wire is the projector's gradient
    (12.6M parameters on 3B), and it is already on a CUDA device.
    """
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) == 1:
        return 0, 1
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    # Each replica's *first* card is its NCCL device. The second one carries the encoder and never
    # takes part in a collective.
    torch.cuda.set_device(rank * 2)
    return rank, world


def replica_devices(rank: int, world: int, policy: str, encoder: str) -> tuple[str, str]:
    """Where this replica's two models go.

    With one replica the configured values are used verbatim, so `POLICY_DEVICE=cuda:0` still means
    cuda:0. With more, the pair is derived from the rank -- (cuda:0, cuda:1), (cuda:2, cuda:3), ...
    -- because a configuration file cannot name a device per rank and having each replica read the
    same two names would put every one of them on the same two cards.
    """
    if world == 1:
        return policy, encoder
    return f"cuda:{rank * 2}", f"cuda:{rank * 2 + 1}"


def average_gradients(head, world: int) -> None:
    """Mean the projector's gradient across replicas, in place.

    Hand-written rather than `DistributedDataParallel`: DDP would have to wrap the head, and the
    head is called from inside `AlignmentActionSelector.score_batch` with a three-argument signature
    that is not a forward over the batch. The projector is 12.6M parameters at 3B, so one
    all-reduce per step costs less than the wrapping would.

    Averaged, not summed: with the global batch split evenly across replicas, the mean of the
    per-replica means is the mean over the global batch -- which is what `alignment_loss` computes on
    one process.
    """
    if world == 1:
        return
    for param in head.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world


def log(message: str) -> None:
    """Log from the chief only, avoiding duplicate writes from synchronized replicas."""
    if not _IS_CHIEF:
        return
    line = f"[alignment-train] {message}"
    # Use tqdm.write while the bar is active to avoid corrupting terminal refreshes.
    if _BAR is not None:
        _BAR.write(line)
    else:
        print(line, flush=True)


def log_every_rank(message: str, rank: int) -> None:
    """Log rank-specific information such as device placement."""
    print(f"[alignment-train][r{rank}] {message}", flush=True)


def hms(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class _OneLinePerRefresh:
    """Adapt tqdm refreshes to separate lines for non-TTY output.

    Remove carriage returns and trailing padding so progress and event logs remain
    readable when stdout is redirected.
    """

    def __init__(self, stream) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        cleaned = text.replace("\r", "").rstrip()
        if not cleaned:
            return 0
        return self._stream.write(cleaned + "\n")

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return False


class Progress:
    """Report training progress with an ETA that includes validation time.

    Use tqdm for both terminals and redirected logs, throttling non-TTY refreshes.
    The custom ETA includes pending evaluations; tqdm's step-only ETA does not.
    Fall back to periodic log lines if tqdm is unavailable.
    """

    # Use the validation-aware ETA instead of tqdm's step-only remaining time.
    BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"

    def __init__(self, total_steps: int, eval_every: int, *, enabled: bool,
                 every_seconds: float = 30.0) -> None:
        global _BAR
        self.total = max(1, total_steps)
        self.eval_every = max(1, eval_every)
        self.every_seconds = every_seconds
        self.milestone = max(1, self.total // 20)
        self.enabled = enabled
        self.bar = None
        now = time.time()
        self.started = now
        self.last_line = now
        self.last_tick = now
        # An exponential average lets the initial CUDA warmup decay out of the ETA.
        self.step_seconds = 0.0
        self.eval_seconds = 0.0
        if not enabled:
            return
        try:
            from tqdm import tqdm
        except ImportError:
            return
        tty = sys.stdout.isatty()
        self.bar = tqdm(
            total=self.total,
            unit="step",
            bar_format=self.BAR_FORMAT,
            # Throttle redirected output independently of interactive terminal refreshes.
            mininterval=0.1 if tty else every_seconds,
            # Terminal-width padding is useful only for TTY output.
            dynamic_ncols=tty,
            # Share the event-log stream to preserve progress/event ordering.
            file=sys.stdout if tty else _OneLinePerRefresh(sys.stdout),
        )
        _BAR = self.bar

    def advance(self, step: int, epoch: int, loss: float) -> None:
        if not self.enabled:
            return
        now = time.time()
        sample = now - self.last_tick
        self.last_tick = now
        self.step_seconds = sample if self.step_seconds == 0.0 else (
            0.9 * self.step_seconds + 0.1 * sample)
        if self.bar is not None:
            self.bar.update(1)
            self.bar.set_postfix(epoch=epoch, loss=f"{loss:.4f}",
                                 eta=hms(self.eta(step)), refresh=False)
            return
        # The fallback reports by time or step count and always emits the final step.
        due_by_time = now - self.last_line >= self.every_seconds
        due_by_step = step % self.milestone == 0
        if not (due_by_time or due_by_step or step >= self.total):
            return
        self.last_line = now
        log(f"progress {step}/{self.total} ({100.0 * step / self.total:.0f}%) epoch {epoch} "
            f"loss={loss:.4f} {self.step_seconds:.1f}s/step "
            f"elapsed {hms(now - self.started)} eta {hms(self.eta(step))}")

    def note_eval(self, seconds: float) -> None:
        """Record validation duration and restart the training-step timer.

        This keeps validation time out of the moving average of training-step time.
        """
        if not self.enabled:
            return
        self.eval_seconds = seconds if self.eval_seconds == 0.0 else (
            0.5 * self.eval_seconds + 0.5 * seconds)
        self.last_tick = time.time()

    def eta(self, step: int) -> float:
        """Estimate remaining training time plus pending validation time.

        Count evaluation boundaries still ahead, including a final off-boundary
        evaluation, rather than dividing the remaining steps by the interval.
        """
        left = max(0, self.total - step)
        evals_left = (self.total // self.eval_every - step // self.eval_every
                      + (1 if self.total % self.eval_every else 0))
        return left * self.step_seconds + max(0, evals_left) * self.eval_seconds

    def close(self) -> None:
        global _BAR
        if self.bar is not None:
            self.bar.close()
        _BAR = None


_IS_CHIEF = True
_BAR = None


def steps_in_epoch(n_rows: int, global_batch: int, world: int, *, drop_last: bool = False) -> int:
    """Actual optimizer updates per epoch, including the selected tail policy.

    The public training launcher sets drop_last so every update has the same global
    sample count. Legacy direct callers may retain their partial final update.
    """
    # A batch smaller than the replica count cannot supply even one sample per replica.
    if world > 1 and global_batch < world:
        return 0
    full, tail = divmod(n_rows, global_batch)
    if drop_last or tail == 0 or (world > 1 and tail < world):
        return full
    return full + 1


def projector_displacement(head, initial: dict[str, torch.Tensor]) -> float:
    """L2 distance the projector has travelled from its initialisation.

    Every head parameter is included in the distance.
    """
    total = 0.0
    for name, param in head.named_parameters():
        if name not in initial:
            continue
        total += float((param.detach().float().cpu() - initial[name]).pow(2).sum().item())
    return total ** 0.5


def assert_parameter_surface(model: AlignmentActionSelector, optimizer: torch.optim.Optimizer) -> int:
    trainable = sorted(model.trainable_parameter_names())
    expected = sorted(model.expected_trainable_names())
    if trainable != expected:
        raise RuntimeError(
            "the trainable parameter set is not the head's.\n"
            f"  unexpectedly trainable: {sorted(set(trainable) - set(expected))[:8]}\n"
            f"  unexpectedly frozen:    {sorted(set(expected) - set(trainable))[:8]}\n"
            "Section 2.1 freezes both language models; a run that trains one of them anyway "
            "produces perfectly normal metrics."
        )
    in_optimizer = {id(p) for group in optimizer.param_groups for p in group["params"]}
    should_be = {id(p) for name, p in model.named_parameters() if name in set(expected)}
    if in_optimizer != should_be:
        raise RuntimeError(
            f"the optimizer covers {len(in_optimizer)} tensors but the trainable set has "
            f"{len(should_be)}. Freezing and stepping are two different surfaces, and agreeing on "
            "one of them is not agreeing on both."
        )
    return sum(p.numel() for name, p in model.named_parameters() if name in set(expected))


def accumulate_batch(model, batch, micro_batch_size: int) -> float:
    """Backpropagate the sample mean once per micro-batch; caller owns zero/step."""
    if not batch or micro_batch_size < 1:
        raise ValueError("A nonempty batch and positive micro-batch size are required")
    total_loss = 0.0
    for micro in batches(batch, micro_batch_size):
        logits = model.score_batch(micro)
        labels = [s["action_set"].index(s["label"]) for s in micro]
        loss, _ = alignment_loss(logits, labels)
        weighted = loss * (len(micro) / len(batch))
        weighted.backward()
        total_loss += float(weighted.detach().item())
    return total_loss


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=str(cfg.dataset_path()),
                        help="Alignment Parquet dataset containing train/val/test splits; only train and val are used here")
    parser.add_argument("--out", default=str(cfg.records_dir() / "base"))
    parser.add_argument("--checkpoint-dir", help="weights directory; defaults to the Alignment checkpoint root with the output run name")
    parser.add_argument("--policy-model", default=DEFAULT_POLICY_MODEL)
    parser.add_argument("--encoder-model", default="",
                        help="empty = the same path as the policy LM, loaded as separate weights")
    parser.add_argument("--projector", default="attention", choices=["attention", "mlp", "mean"])
    parser.add_argument("--representation", default="final_layer_hidden_states",
                        choices=["final_layer_hidden_states", "input_embeddings"])
    parser.add_argument("--scale", default="unit", choices=["unit", "uniform", "none"],
                        help="unit = every action row has norm 1 (default). uniform "
                             "matches mean(||lm_head row||) and only exists for "
                             "comparison against runs made before actions and "
                             "vocabulary were confirmed to be separate softmaxes")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr-schedule", default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="samples per replica per optimizer update (not micro-batch size)")
    parser.add_argument("--micro-batch-size", type=int, default=None,
                        help="maximum samples per replica per forward/backward; defaults to --batch-size")
    parser.add_argument("--drop-last", action="store_true",
                        help="drop incomplete epoch tails to keep samples/update constant")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--encoder-batch-size", type=int, default=8,
                        help="maximum action prompts per encoder forward; independent of samples/update")
    parser.add_argument("--encoder-max-length", type=int, default=1536)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--encoder-device", default="cuda:1")
    parser.add_argument("--dtype", default="bfloat16", choices=list(("bfloat16", "float16", "float32")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="cap train and val only; for smoke runs")
    parser.add_argument("--wandb", default="auto", choices=["auto", "on", "off"],
                        help="stream the curve to wandb while training. auto = when a credential "
                             "is available, on = required (cluster jobs), off = never")
    args = parser.parse_args(argv)
    if args.micro_batch_size is None:
        args.micro_batch_size = args.batch_size
    if args.batch_size < 1 or args.micro_batch_size < 1:
        parser.error("--batch-size and --micro-batch-size must be positive")

    if args.eval_every < 1 or args.eval_batch_size < 1 or args.epochs < 1 or args.limit < 0:
        parser.error("eval cadence, batch size and epochs must be positive; limit must be nonnegative")

    if args.encoder_batch_size < 1:
        parser.error("--encoder-batch-size must be positive")

    rank, world = setup_replicas()
    is_chief = rank == 0
    global _IS_CHIEF
    _IS_CHIEF = is_chief
    policy_device, encoder_device = replica_devices(
        rank, world, args.policy_device, args.encoder_device)
    # Same seed on every replica: the projector must start identical, otherwise averaging the
    # gradients averages the updates of two different models. The data split is what differs, and
    # it is derived from the rank rather than from the seed.
    torch.manual_seed(args.seed)
    out = Path(args.out).expanduser().resolve()
    checkpoint_dir = (Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir
                      else cfg.runs_dir() / out.name)
    args.out, args.checkpoint_dir = str(out), str(checkpoint_dir)
    for key, suffix in (('WANDB_DIR', 'wandb'), ('WANDB_CACHE_DIR', 'wandb/cache'),
                        ('WANDB_CONFIG_DIR', 'wandb/config'), ('WANDB_DATA_DIR', 'wandb/data')):
        os.environ[key] = str(out / suffix)
        if is_chief:
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    if is_chief:
        out.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_split(args.dataset, split="train", limit=args.limit)
    if args.drop_last and len(train_rows) < args.batch_size * world:
        raise ValueError("Training split is smaller than the global batch; no complete update is possible")
    val_rows = load_split(args.dataset, split="val", limit=args.limit)
    if world > 1:
        # Report placement on every rank so each pair of devices can be checked.
        log_every_rank(f"replica {rank}/{world} on {policy_device} + {encoder_device}", rank)
    log(f"train={len(train_rows)} val={len(val_rows)}")

    model = AlignmentActionSelector(AlignmentConfig(
        policy_model=args.policy_model,
        encoder_model=args.encoder_model,
        projector=args.projector,
        representation=args.representation,
        scale=args.scale,
        max_length=args.max_length,
        encoder_max_length=args.encoder_max_length,
        encoder_batch_size=args.encoder_batch_size,
        policy_device=policy_device,
        encoder_device=encoder_device,
        dtype=args.dtype,
    ))
    optimizer = torch.optim.AdamW(
        [p for p in model.head.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    trainable = assert_parameter_surface(model, optimizer)
    log(f"trainable parameters: {trainable:,} ({', '.join(model.expected_trainable_names())})")

    initial = {name: p.detach().float().cpu().clone() for name, p in model.head.named_parameters()}
    # Use one configuration for both disk metadata and W&B.
    protocol = {"split_policy": SPLIT_POLICY, "evaluation_role": "train-validation",
                "dataset_identity": file_identity(args.dataset)}
    run_record = {
        **protocol,
        "args": vars(args),
        "model_config": asdict(model.config),
        "replicas": world,
        "global_batch_size": args.batch_size * world,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": math.ceil(args.batch_size / args.micro_batch_size),
        "dropped_samples_per_epoch": len(train_rows) % (args.batch_size * world) if args.drop_last else 0,
        "trainable_parameters": trainable,
        "trainable_names": model.expected_trainable_names(),
        "policy_hidden": model.policy_hidden,
        "encoder_hidden": model.encoder.hidden_size,
    }
    if is_chief:
        (out / "config.yaml").write_text(
            yaml.safe_dump(run_record, sort_keys=False), encoding="utf-8")

    metrics_path = out / "metrics.jsonl"
    metric_lines: list[str] = []
    if is_chief:
        metrics_path.write_text("", encoding="utf-8")

    # Only the chief opens the W&B run; start before recording the step-zero evaluation.
    wandb_run.set_logger(log)
    live = LiveWandb.start(
        run_name=out.name,
        config=wandb_run.run_config(run_record),
        mode=args.wandb,
        enabled=is_chief,
    )

    def require_validation(metrics: dict[str, Any]) -> dict[str, Any]:
        ce = metrics.get("cross_entropy")
        if not isinstance(ce, (int, float)) or not math.isfinite(ce):
            raise ValueError("Checkpoint selection requires finite validation cross_entropy; "
                             "refusing to publish or save an invalid checkpoint")
        return metrics

    def record(step: int, epoch: int, train_loss: float) -> dict[str, Any]:
        # `evaluate` is collective when there is more than one replica -- every rank has to call it
        # or the all-reduce inside hangs. Only the write is chief-only.
        row = {
            **protocol,
            "step": step,
            "epoch": epoch,
            "train_loss": train_loss,
            "projector_l2_displacement": projector_displacement(model.head, initial),
            "val": require_validation(evaluate(model, val_rows, args.eval_batch_size)),
        }
        if is_chief:
            # Rewrite the short evaluation history: repeated append opens fail on BlobFuse.
            metric_lines.append(json.dumps(row) + "\n")
            metrics_path.write_text("".join(metric_lines), encoding="utf-8")
            # Persist the authoritative metric row before attempting the network upload.
            live.log_row(row)
            log(f"step {step} loss={train_loss:.4f} val_ce={row['val']['cross_entropy']:.4f} "
                f"val_top1={row['val']['top1_accuracy']:.4f} "
                f"(chance {row['val']['chance_accuracy']:.4f}) "
                f"proj_l2={row['projector_l2_displacement']:.4f}")
        return row

    # Record step zero with checkpoint selection below to reuse its evaluation.

    # The global batch is what the loss is a mean over, so it is what the step count is derived
    # from. `batches` yields global batches and each replica takes a stride of one; a global batch
    # shorter than `world` is skipped, because it would leave some replica with nothing to reduce.
    global_batch = args.batch_size * world
    steps_per_epoch = steps_in_epoch(len(train_rows), global_batch, world, drop_last=args.drop_last)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup = max(1, int(total_steps * args.warmup_ratio))
    scheduler = None
    if args.lr_schedule == "cosine":
        def factor(current: int) -> float:
            if current < warmup:
                return (current + 1) / warmup
            progress = (current - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    # Only val selects the checkpoint. The independent test split is never scored here.
    # Strict improvement keeps the earlier checkpoint on a tie, including step zero.
    best = {"step": 0, "cross_entropy": float("inf"), "state": None}

    def consider(step: int, metrics: dict[str, Any]) -> None:
        require_validation(metrics)
        if metrics["cross_entropy"] < best["cross_entropy"]:
            best.update(
                step=step,
                cross_entropy=metrics["cross_entropy"],
                state={k: v.detach().cpu().clone()
                       for k, v in model.head.state_dict().items() if v is not None},
            )

    # Reuse the recorded evaluation for checkpoint selection.
    zero_started = time.time()
    consider(0, record(0, 0, float("nan"))["val"])
    zero_seconds = time.time() - zero_started

    progress = Progress(total_steps, args.eval_every, enabled=is_chief)
    # Seed the ETA with the already measured step-zero evaluation duration.
    progress.note_eval(zero_seconds)
    log(f"{total_steps} steps ({steps_per_epoch}/epoch × {args.epochs} epochs), "
        f"eval every {args.eval_every} (~{zero_seconds:.0f}s each)")

    step = 0
    started = time.time()
    running: list[float] = []
    last_loss = float("nan")
    last_row: dict[str, Any] | None = None
    for epoch in range(args.epochs):
        # Every replica walks the same shuffled sequence of *global* batches -- same seed, same
        # order -- and takes a disjoint stride out of each. Sharding the row list instead would
        # give each replica its own shuffle, and then "one step" would mean a different set of
        # rows on each of them.
        for whole in batches(train_rows, global_batch, shuffle=True, seed=args.seed + epoch):
            if args.drop_last and len(whole) != global_batch:
                continue
            if world > 1:
                if len(whole) < world:
                    continue                    # tail too short to give every replica a row
                batch = whole[rank::world]
            else:
                batch = whole
            optimizer.zero_grad(set_to_none=True)
            loss_value = accumulate_batch(model, batch, args.micro_batch_size)
            average_gradients(model.head, world)
            if args.grad_clip > 0:
                # After the all-reduce, so every replica clips the same averaged gradient and their
                # parameters stay bit-identical. Clipping first would let each replica scale by its
                # own local norm and the averaged update would no longer be a clipped update.
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.head.parameters() if p.requires_grad], args.grad_clip
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running.append(loss_value)
            step += 1
            progress.advance(step, epoch, running[-1])
            if step % args.eval_every == 0:
                last_loss = sum(running) / len(running)
                eval_started = time.time()
                last_row = record(step, epoch, last_loss)
                consider(step, last_row["val"])
                progress.note_eval(time.time() - eval_started)
                running = []
        log(f"epoch {epoch} done at step {step} ({time.time() - started:.0f}s)")

    # `running` is empty whenever the last step happened to land on an eval boundary. Reporting the
    # NaN that `sum([])/0` would give makes the final row look like a numerical failure.
    progress.close()
    final_train_loss = sum(running) / len(running) if running else last_loss
    if last_row is not None and last_row["step"] == step:
        # The final step was already an eval boundary (`total_steps % eval_every == 0`), so it has
        # been evaluated and written. Doing it again costs a full validation pass -- 90 seconds on
        # 3B -- and puts two rows with the same `step` in metrics.jsonl, which every curve reader
        # downstream then has to decide what to do about. h200 hits this exactly: 168 steps at
        # EVAL_EVERY=8. Reuse instead, and do not pick a non-dividing cadence to dodge it.
        last = last_row
    else:
        last = record(step, args.epochs - 1, final_train_loss)
        consider(step, last["val"])

    # metrics.jsonl stays the honest curve; only what gets shipped is the selected checkpoint.
    if best["state"] is None or not math.isfinite(best["cross_entropy"]):
        raise ValueError("No checkpoint selected with finite validation cross_entropy; refusing to save")
    if best["step"] != step:
        log(f"restoring the step {best['step']} checkpoint "
            f"(val CE {best['cross_entropy']:.4f} against {last['val']['cross_entropy']:.4f} at the end)")
        model.head.load_state_dict(best["state"], strict=True)

    summary = {
        **protocol,
        "steps": step,
        "selected_step": best["step"],
        "wall_seconds": time.time() - started,
        "trainable_parameters": trainable,
        "projector_l2_displacement": projector_displacement(model.head, initial),
        "train_loss": final_train_loss,
        "val": require_validation(evaluate(model, val_rows, args.eval_batch_size)),
        "val_at_last_step": last["val"],
        "val_by_form": by_group(model, val_rows, "action_set_form", args.eval_batch_size),
        "val_by_size": by_group(model, val_rows, "mcp_size", args.eval_batch_size),
    }
    if is_chief:
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Everything Agentic RL has to check before it may load this file. The first three fields were
    # here already because they are the ones that fail loudly; the rest are the ones that do not.
    #
    # `scale` is the reason this list grew. `unit` and `uniform` differ by an 8.9% temperature on
    # Qwen2.5-3B and by nothing else, so loading a `unit` head into a `uniform` run raises nothing,
    # trains to completion, and reports every metric one notch off. It cannot be recovered at load
    # time either -- the rows carry their norm, not the rule that produced it.
    #
    # `policy_model` and `encoder_model` pin the space. `w_a` is only meaningful against the `x_t`
    # of the model it was fitted to, and hidden size is a weak proxy: several models share 2048.
    # `encoder_model` is stored **resolved**, not as the flag was given: the flag's empty string
    # means "same path as the policy LLM backbone", so writing it verbatim would store a value that compares
    # equal to any other empty value, which is exactly the check being asked for.
    if not is_chief:
        # Only the chief writes synchronized weights; barrier before peers tear down the process group.
        dist.barrier()
        dist.destroy_process_group()
        return 0

    # Package the completed run with its weights so a checkpoint-only Blob download is
    # self-contained. Copy real files: symlinks into outputs do not survive that download.
    for name in ("config.yaml", "metrics.jsonl", "summary.json"):
        source, destination = out / name, checkpoint_dir / name
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)

    torch.save(
        {
            **protocol,
            "representation": args.representation,
            "state_dict": {k: v.detach().cpu() for k, v in model.head.state_dict().items()
                           if v is not None},
            "initial_state_dict": initial,
            "projector": args.projector,
            "encoder_hidden": model.encoder.hidden_size,
            "policy_hidden": model.policy_hidden,
            "scale": args.scale,
            "policy_model": args.policy_model,
            "encoder_model": model.config.resolved_encoder_model(),
            # Which step's weights these are. `state_dict` is the best-CE checkpoint, not the last
            # one, and `initial_state_dict` is step 0 -- three different things that a reader who
            # only sees the file would have to guess between.
            "selected_step": best["step"],
            "steps": step,
        },
        checkpoint_dir / "projector.pt",
    )
    log(f"val top1={summary['val']['top1_accuracy']:.4f} "
        f"(chance {summary['val']['chance_accuracy']:.4f})")
    log(f"wrote {out}")
    # Save weights before waiting for W&B finish.
    # The post-run uploader sets the final summary from the selected checkpoint, which may precede the last step.
    live.finish()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
