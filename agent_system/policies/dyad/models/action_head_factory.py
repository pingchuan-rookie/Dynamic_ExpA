# Copyright 2025 ExpA_sys
"""Build direct action heads and apply the encoder training schedule.

The encoder supplies one hidden-state sequence per action in action_ids order.
The projector and optional width projection produce each action row.
The vocabulary head supplies the output device/dtype.
Uniform scaling retains description-token reference row norms for checkpoint behavior compatibility.
The same head parameters and encoder cache must be used for sampling and training replay.
"""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Construct and initialize the action encoder at the official model-setup boundary.
# Extension point: install_dyad_encoder / rollout head materialization

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from agent_system.policies.dyad.models.encoder_config import LlmEncoderConfig


def _reference_rows(action_config: dict, vocab_head: torch.Tensor, tokenizer) -> torch.Tensor:
    """Preserve the description-token norm reference used by uniform scaling.

    These detached rows supply norms only; the encoder projector still supplies every head direction.
    Tokenization, out-of-vocabulary filtering, row order, and averaging retain the existing arithmetic.
    """
    if vocab_head.ndim != 2:
        raise ValueError(f"expected vocab_head [vocab, hidden], got {tuple(vocab_head.shape)}")
    descriptions = action_config["id_to_description"]
    rows = []
    for index, action_id in enumerate(action_config["action_ids"]):
        description = (descriptions[action_id] if action_id in descriptions
                       else descriptions.get(str(action_id), "")) or ""
        token_ids = tokenizer.encode(description, add_special_tokens=False) if description else []
        token_ids = [int(t) for t in token_ids if 0 <= int(t) < vocab_head.shape[0]]
        if not token_ids:
            raise ValueError(f"action row {index} has no encodable description (got {description!r})")
        ids = torch.tensor(token_ids, device=vocab_head.device, dtype=torch.long)
        rows.append(vocab_head[ids].mean(dim=0).detach())
    return torch.stack(rows, dim=0)


def build_action_head(
    action_config: dict[str, Any],
    vocab_head: torch.Tensor,
    tokenizer,
    *,
    llm_cfg: Optional[LlmEncoderConfig] = None,
    encoder=None,
    residual_head=None,
    prompts: Optional[list[str]] = None,
) -> torch.Tensor:
    """Direct head rows in action_ids order, with an enabled encoder required.

    MeanProjector is a parameter-free pooler over encoder representations, not vocabulary rows.
    Only uniform scaling tokenizes descriptions to preserve its existing norm reference.
    """
    if llm_cfg is None or not llm_cfg.enabled:
        raise ValueError(
            "build_action_head needs an enabled LlmEncoderConfig; set DYAD_ENCODER_ENABLED=1. "
            "For a head with no "
            "trainable parameters use DYAD_ENCODER_PROJECTOR=mean, which still runs the encoder."
        )
    if encoder is None or residual_head is None:
        raise ValueError(
            "LlmEncoderConfig.enabled is true but no encoder/residual_head was passed."
        )
    num_actions = len(action_config["action_ids"])

    if prompts is None:
        # The text form comes from the setting when one is attached to the config; `mcp` otherwise,
        # which is what every existing run used.
        prompts = encoder_prompts(action_config, description=getattr(llm_cfg, "description", "mcp"))
    if len(prompts) != num_actions:
        raise ValueError(
            f"{len(prompts)} encoder prompts for {num_actions} head rows -- the prompt list and "
            "the head rows are the same ordered list, so a length mismatch means they are not aligned"
        )

    cached = encoder.encode_hidden(prompts)
    if cached.hidden.shape[0] != num_actions:
        raise ValueError(
            f"encoder produced {cached.hidden.shape[0]} rows for {num_actions} action ids; "
            "the encoder output and action rows must be aligned"
        )
    reference = (
        _reference_rows(action_config, vocab_head, tokenizer)
        if residual_head.scale == "uniform" else vocab_head
    )
    return residual_head(cached.hidden, cached.mask, reference)


def encoder_prompts(
    action_config: dict[str, Any],
    description: str,
    *,
    with_catalogue: bool = True,
) -> list[str]:
    """One prompt per head row, in `action_ids` order.

    Each prompt is the two-part text of settings.md section 5:

        [the whole environment's action descriptions]  +  [now understand *this* action]

    The catalogue is identical for every row, so it is built once here rather than per action.
    `with_catalogue=False` drops it, which is what the encoder read before the preface existed --
    kept so the two can be compared rather than swapped blindly.

    `description` picks the module under `models/description_forms/`. Required, deliberately:
    it used to default to `"mcp"`, and the trainer called this without it -- so a run configured
    for plain English produced an MCP run with the right shapes, the right metrics, and an ablation
    comparing MCP against itself. A caller with no opinion here does not exist; one that forgets
    should not compile.

    `action_config["env_name"]` is deliberately **not** read here. It used to be, and it reached the
    encoder as `... in the "alfworld_base" environment` -- see `descriptions.py`'s docstring for why
    that breaks PRINCIPLES P2. The environment reaches the encoder only as its catalogue of actions.
    """
    from agent_system.policies.dyad.models.action_descriptions import build_catalogue
    from agent_system.policies.dyad.models.action_descriptions import describe_action_config

    docs = describe_action_config(action_config)
    catalogue = build_catalogue(docs, form=description) if with_catalogue else ""
    return [doc.prompt(form=description, catalogue=catalogue) for doc in docs]



def assert_encoder_lm_training_supported(setting) -> None:
    """Require an independent backbone for full encoder training.

    The engine's EncoderTrainingMixin owns optimizer registration, gradient
    reduction before clipping, placement, checkpoints and sampling-replica sync.
    Static heads use fresh encoder features; dynamic heads replay saved prompts.
    """
    # Check the raw schedule: trains_encoder_lm already assumes an owned backbone.
    from agent_system.policies.dyad.models.encoder_config import EncoderTraining

    if getattr(setting, "encoder_training", None) is not EncoderTraining.ADAPTER_AND_ENCODER_LM:
        return
    if not getattr(setting, "needs_separate_encoder_lm", False):
        raise ValueError(
            "encoder_training=projector_and_encoder_lm 需要 backbone=encoder_lm（encoder 拥有自己的模型），"
            "而这个 setting 读的是 policy LM —— 训那个 backbone 就是训 policy LM，由 training_schedule 决定，"
            "不是这个开关的事。"
        )


def _encoder_side_params(encoder):
    """Yield trainable encoder-backbone parameters for freezing and counting.

    LoRA selects lora_ parameters; full fine-tuning selects the whole backbone.
    Both callers use this selection to keep reported and actual training scope aligned.
    """
    backbone = getattr(encoder, "backbone", None) if encoder is not None else None
    if backbone is None:
        return []
    if getattr(encoder, "lora_enabled", False):
        return [(n, p) for n, p in backbone.named_parameters() if "lora_" in n]
    if getattr(encoder, "full_finetune", False):
        return list(backbone.named_parameters())
    return []


def apply_training_schedule(
    setting,
    *,
    actor_module=None,
    residual_head=None,
    encoder=None,
) -> dict:
    """Freeze whichever side this training_schedule does not train. Returns what is trainable, and why.

    All three schedules are single phase:

        joint            both sides learn                    (the main experiment)
        frozen_llm_adaptation     policy LLM backbone frozen, encoder trains    (ablation: what did the encoder add)
        policy_lm_only   action head frozen, policy LLM backbone trains (ablation: what did learning the
                                                              interface add, given a fixed one)

    `step` / `total_steps` used to be parameters here, for `encoder_then_policy_lm`'s midpoint
    switch. That schedule was dropped on 2026-09-01; the arguments went with it rather than being
    kept "in case" -- an unused switch step is a thing a future reader has to prove is unused.

    Returning the counts instead of nothing is the point of the function as much as the freezing is:
    whether a freeze took effect is otherwise invisible until a run has been spent, and getting it
    wrong raises nothing.
    """
    from agent_system.policies.dyad.models.encoder_config import TrainingSchedule

    if setting.training_schedule is TrainingSchedule.JOINT_OPTIMIZATION:
        phase, train_actor, train_encoder = "joint_optimization", True, True
    elif setting.training_schedule is TrainingSchedule.FROZEN_LLM_ADAPTATION:
        phase, train_actor, train_encoder = "frozen_llm_adaptation", False, True
    elif setting.training_schedule is TrainingSchedule.POLICY_LM_ONLY:
        phase, train_actor, train_encoder = "policy_lm_only", True, False
    else:
        # Reject unknown schedules instead of silently choosing a training scope.
        raise ValueError(
            f"training_schedule {setting.training_schedule!r} has no branch here. Adding one means "
            "adding a branch -- falling through to another schedule's freezing would train the "
            "wrong side and raise nothing."
        )

    projector_ids = {id(p) for p in residual_head.parameters()} if residual_head is not None else set()
    if actor_module is not None:
        for param in actor_module.parameters():
            if id(param) not in projector_ids:
                param.requires_grad_(train_actor)
    if residual_head is not None:
        for param in residual_head.parameters():
            param.requires_grad_(train_encoder and setting.trains_projector)
    # Use the same encoder-parameter selection for LoRA and full-backbone training.
    for _name, param in _encoder_side_params(encoder):
        param.requires_grad_(train_encoder and setting.trains_encoder_lm)

    counts = {"policy_lm": 0, "projector": 0, "encoder_lm": 0}
    if actor_module is not None:
        counts["policy_lm"] = sum(
            p.numel() for p in actor_module.parameters()
            if p.requires_grad and id(p) not in projector_ids
        )
    if residual_head is not None:
        counts["projector"] = sum(p.numel() for p in residual_head.parameters() if p.requires_grad)
    counts["encoder_lm"] = sum(p.numel() for _n, p in _encoder_side_params(encoder)
                                if p.requires_grad)

    if sum(counts.values()) == 0:
        raise RuntimeError(
            f"training_schedule {setting.training_schedule.value} left nothing trainable "
            f"({setting.describe()}). The run would complete with every metric looking normal and "
            "no parameter updated."
        )

    # Which gradient lines the forward hook should emit. Derived from `counts` rather than from
    # `train_actor` / `train_encoder`, because those two say what the schedule intends and `counts`
    # says what the freezing actually produced -- `trains_projector` can be false under a schedule
    # that nominally trains the encoder, and then there is no encoder line to compute.
    #
    # See `agent_system.policies.dyad.models.policy_forward.attach_dyad_forward_hook` for what the values mean. Setting it on the
    # module is how it reaches a hook that runs inside FSDP's forward, where no config object is in
    # scope.
    policy_line = counts["policy_lm"] > 0
    encoder_line = counts["projector"] > 0 or counts["encoder_lm"] > 0
    if actor_module is not None:
        actor_module.dyad_gradient_lines = (
            "both" if policy_line and encoder_line
            else "encoder" if encoder_line
            else "policy" if policy_line
            else None
        )

    return {"phase": phase, "counts": counts, "setting": setting.describe(),
            "gradient_lines": getattr(actor_module, "dyad_gradient_lines", None)}


def llm_encoder_config_from_env(env: Optional[dict] = None) -> LlmEncoderConfig:
    """Read the encoder's configuration from the environment.

    Env vars rather than the Hydra config because the encoder has to be built inside Ray actors and
    inside the vLLM engine fork, neither of which is handed the trainer's config object. `main_dyad`
    already injects the whole `DYAD_` prefix into `runtime_env.env_vars`, so these arrive intact.
    """
    import os

    source = env if env is not None else os.environ
    gpu_ids = tuple(
        int(x) for x in str(source.get("DYAD_ENCODER_GPU_IDS", "")).replace(",", " ").split() if x.strip()
    )
    # Empty means "share the policy LLM backbone's rate"; see LlmEncoderConfig.projector_lr for the measurement
    # that says sharing it leaves the projector effectively frozen.
    _projector_lr = str(source.get("DYAD_PROJECTOR_LR", "") or "").strip()
    return LlmEncoderConfig(
        enabled=str(source.get("DYAD_ENCODER_ENABLED", "0")).strip().lower() in {"1", "true", "yes"},
        model_path=str(source.get("DYAD_ENCODER_MODEL_PATH", "") or ""),
        # are two switches that can disagree, and the disagreement would show up as a head built by
        # one rule and described by the other.
        projector=str(source.get("DYAD_ENCODER_PROJECTOR", "attention")).strip().lower(),
        scale=str(source.get("DYAD_ENCODER_SCALE", "unit")).strip().lower(),
        max_length=int(source.get("DYAD_ENCODER_MAX_LENGTH", "1024")),
        dtype=str(source.get("DYAD_ENCODER_DTYPE", "bfloat16")),
        device=str(source.get("DYAD_ENCODER_DEVICE", "cuda")),
        gpu_ids=gpu_ids,
        backbone=str(source.get("DYAD_ENCODER_BACKBONE", "encoder_lm")).strip().lower(),
        representation=str(source.get("DYAD_ENCODER_REPRESENTATION", "final_layer_hidden_states")).strip().lower(),
        description=str(source.get("DYAD_ENCODER_DESCRIPTION", "mcp")).strip().lower(),
        remote=str(source.get("DYAD_ENCODER_REMOTE", "0")).strip().lower() in {"1", "true", "yes"},
        num_gpus=int(source.get("DYAD_ENCODER_NUM_GPUS", "0")),
        actor_name=str(source.get("DYAD_ENCODER_ACTOR_NAME", "dyad_action_encoder")),
        projector_lr=float(_projector_lr) if _projector_lr else None,
        projector_init=str(source.get("DYAD_ENCODER_PROJECTOR_INIT", "") or "").strip(),
    )


def _model_identity(path: str) -> str:
    """`Qwen2.5-0.5B-Instruct` from either spelling of the same model.

    Alignment is given a repo id (`Qwen/Qwen2.5-0.5B-Instruct`) and lets transformers resolve it.
    Agentic RL's engines resolve the snapshot themselves and hand the trainer an absolute path
    (`.../models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<sha>/`). Comparing the two as strings
    rejects a checkpoint that matches perfectly, which would make the check worse than useless:
    the first person to hit it learns that the check lies, and the fix is to stop checking.

    What this deliberately does *not* distinguish is two revisions of one repo. It could -- the sha
    is right there in the path -- but Alignment never records which revision it read, so the
    comparison would be against a value that does not exist. Same-name-different-revision is a real
    hazard and this does not catch it; what it catches is same-hidden-size-different-model, which is
    the one that produces a run rather than a shape error.
    """
    text = str(path).strip().rstrip("/")
    marker = "models--"
    if marker in text:
        tail = text.split(marker, 1)[1]
        repo = tail.split("/snapshots", 1)[0]
        return repo.split("--")[-1]
    return text.rsplit("/", 1)[-1]


# Keep site namespaces separate: equal run names can describe different experiments.
PROJECTOR_INIT_SITES = ("lucia", "local")


def _resolve_projector_init(raw: str, *, project: "Path | None" = None) -> "Path":
    """Turn whatever `DYAD_ENCODER_PROJECTOR_INIT` was given into the `projector.pt` it means.

    Historical runs remain under `ckpt/<site>/stage1/<model>_<timestamp>/` in the repository.
    New runs use `ARTIFACT_ROOT/ckpt/`; the public launcher resolves those to absolute paths before
    this compatibility resolver is called. The file Agentic RL needs is `projector.pt` inside a run.
    These five legacy spellings are accepted:

        ckpt/local/alignment/qwen2.5-0.5b-instruct_20260901_052225   relative to the repository root
        local/qwen2.5-0.5b-instruct_20260901_052225               site and run name
        qwen2.5-0.5b-instruct_20260901_052225                     just the run name
        /abs/.../ckpt/local/alignment/<run>                          the run directory
        /abs/.../ckpt/local/alignment/<run>/projector.pt             the file itself

    Accepting only the last one is what the code did before, and it made the common case -- copying
    a run name out of `ls` -- fail with "is not a file" on a path the person had every reason to
    think was right. The failure below lists what does exist instead.

    **A bare run name that exists under both sites is an error, not a choice.** The two subtrees
    were introduced precisely because the same `<model>_<timestamp>` can appear in both, and
    picking one -- either one, by any rule -- gives a run that trains normally against whichever
    checkpoint the rule happened to prefer. There is no metric that would report the difference,
    so the ambiguity has to fail here or it never surfaces at all.

    `project` overrides the repository root. It exists for the tests: the two-site behaviour is
    about what sits next to what, and asserting it against the real `ckpt/` would mean writing
    fixtures into a tree that holds the only copy of every local run.
    """
    from pathlib import Path

    project = project or Path(__file__).resolve().parents[4]          # Dynamic_ExpA
    site_roots = {site: project / "ckpt" / site / "alignment" for site in PROJECTOR_INIT_SITES}
    legacy_site_roots = {site: project / "ckpt" / site / "stage1" for site in PROJECTOR_INIT_SITES}
    text = str(raw).strip().rstrip("/")

    candidates = [Path(text), project / text]
    candidates += [root / text for root in site_roots.values()]
    candidates += [root / text for root in legacy_site_roots.values()]
    # `<site>/<run>`: what someone types after this function has told them a bare name is
    # ambiguous, so it has to be one of the accepted spellings rather than a near miss.
    head, _, rest = text.partition("/")
    if rest and head in site_roots:
        candidates.append(site_roots[head] / rest)
        candidates.append(legacy_site_roots[head] / rest)

    # Collect every hit rather than returning the first. Two candidates resolving to the same file
    # is the ordinary case (a relative path given from the project directory matches twice); two
    # resolving to *different* files is the ambiguity above, and a first-match loop cannot tell
    # those apart because it stops before it has seen the second one.
    hits: list[Path] = []
    for cand in candidates:
        found = None
        if cand.is_file():
            found = cand.resolve()
        elif cand.is_dir() and (cand / "projector.pt").is_file():
            found = (cand / "projector.pt").resolve()
        # Always hand back an absolute path: a Ray worker does not inherit the launcher's cwd, so
        # a relative hit here would resolve somewhere else -- or nowhere -- inside the actor.
        if found is not None and found not in hits:
            hits.append(found)

    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ValueError(
            f"DYAD_ENCODER_PROJECTOR_INIT={raw!r} names a run in more than one site:\n"
            + "".join(f"  {h}\n" for h in hits)
            + "  Spell the site to say which one, e.g. "
            f"{PROJECTOR_INIT_SITES[0]}/{hits[0].parent.name}. A cluster run and a local run of "
            "the same model share the timestamp format, and loading the wrong one changes no "
            "metric that would tell you."
        )

    listing = []
    for site, root in site_roots.items():
        runs = sorted(d.name for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
        listing.append(f"  {site}: {', '.join(runs) if runs else '(none)'}")
    raise FileNotFoundError(
        f"DYAD_ENCODER_PROJECTOR_INIT={raw!r} does not name a Alignment projector.\n"
        f"  tried: {', '.join(str(c) for c in candidates)}\n"
        f"  runs under {project / 'ckpt'}:\n" + "\n".join(listing) + "\n"
        "  Give the run directory or its name; leave the variable unset to train the projector "
        "from scratch."
    )



def load_projector_init(head, cfg: LlmEncoderConfig, actor_model_path: str) -> Optional[dict]:
    """Load a Alignment `projector.pt` into `head`, or raise. Returns what was loaded, or None.

    Every check here fails the run rather than falling back, for one reason: the thing being
    guarded against is a run that completes. A projector loaded into the wrong configuration
    produces a normal loss curve, a normal gradient norm and a normal L2 displacement, and the only
    way to find out is to spend the run and compare it against one that was set up correctly.

    Four checks, and they are not equally urgent:

      `projector`     `mean` has no parameters and `attention` has four, so the key sets differ and
                      a mismatch is caught by the state dict anyway. Checked here to say why.
      `scale`         **the one that is silent.** `unit` and `uniform` produce head rows that
                      differ by a constant factor -- 8.9% on Qwen2.5-3B -- and nothing downstream
                      can tell which rule made them. Only the stored field can.
      `policy_model`  `w_a` is fitted against one model's `x_t` and means nothing against another's.
                      Hidden size is a weak proxy: several models are 2048 wide.
      `encoder_model` same, for the rows' own side.

    `policy_hidden` / `encoder_hidden` are checked too, but they are a consequence of the model
    identities rather than an independent fact, and a mismatch there would raise on the shapes.

    Older checkpoints predate the last three fields. They are rejected rather than loaded
    partially: "this file cannot tell you whether it matches" and "this file matches" are different
    answers, and only one of them is a reason to proceed.
    """
    import os
    native_directory = os.environ.get("DYAD_NATIVE_RESTORE_DIR", "").strip()
    if native_directory:
        from agent_system.policies.dyad.inference.checkpoint import load_native_projector
        model_config = os.environ.get("DYAD_NATIVE_MODEL_CONFIG", "").strip()
        if not model_config:
            raise ValueError("DYAD_NATIVE_MODEL_CONFIG is required with DYAD_NATIVE_RESTORE_DIR")
        return load_native_projector(head, cfg, actor_model_path, native_directory, model_config)
    if not cfg.projector_init:
        return None

    path = _resolve_projector_init(cfg.projector_init)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    required = ("state_dict", "projector", "scale", "policy_model", "encoder_model",
                "policy_hidden", "encoder_hidden")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(
            f"{path} is missing {missing}. Checkpoints written before 2026-08-31 carry only "
            "projector / encoder_hidden / policy_hidden, which cannot answer whether the scale rule "
            "and the two models match this run -- and a scale mismatch changes nothing that any "
            "metric would show. Re-run Alignment with the current "
            "agent_system/policies/dyad/training/action_encoder_alignment/actenc_alignment_train.py."
        )

    encoder_model = cfg.resolved_model_path(actor_model_path)
    for field, want, got, why in (
        ("projector", cfg.projector, payload["projector"],
         "a mean projector has no parameters and an attention projector has four"),
        ("scale", cfg.scale, payload["scale"],
         "unit and uniform differ by a constant factor on every row and by nothing else, so this "
         "is the one mismatch that no metric would reveal"),
        ("policy_model", _model_identity(actor_model_path), _model_identity(payload["policy_model"]),
         "w_a was fitted against that model's x_t; the inner product means nothing across models"),
        ("encoder_model", _model_identity(encoder_model), _model_identity(payload["encoder_model"]),
         "the action rows come from that model's hidden states"),
    ):
        if str(want) != str(got):
            raise ValueError(
                f"{path} was written with {field}={got!r} but this run has {field}={want!r}. {why}."
            )
    for field, want in (("policy_hidden", head.proj_out_features(payload["encoder_hidden"])),
                        ("encoder_hidden", payload["encoder_hidden"])):
        if int(payload[field]) != int(want):
            raise ValueError(
                f"{path} has {field}={payload[field]} against this run's {want}. The model "
                "identities matched, so this is a bug rather than a misconfiguration."
            )

    # None-valued persistent encoder buffers are omitted from state_dict, so strict loading
    # accepts projector-only Alignment checkpoints while rejecting missing projector weights
    # and cached encoder buffers from a different action schema.
    try:
        head.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"{path} does not match this head key for key: {exc}\n"
            "A Alignment checkpoint holds the projector's tensors and nothing else. Missing keys mean "
            "a partial load, which leaves part of the projector at its own initialisation; the two "
            "encoder buffers turning up as unexpected keys means the checkpoint was written after "
            "set_encoder_cache and carries Alignment's action set."
        ) from exc
    return payload


def build_llm_encoder(
    cfg: LlmEncoderConfig,
    actor_model_path: str,
    actor_hidden: int,
    *,
    actor_gpu_ids: tuple[int, ...] = (),
    actor_module=None,
    actor_tokenizer=None,
):
    """`_build_llm_encoder` plus the Alignment checkpoint, if one was configured.

    Wrapped rather than inlined because the builder has three return points (policy_lm backbone,
    remote actor, local encoder LLM backbone) and the load has to happen on all three. A per-branch call is
    three chances for a new branch to be added without one.
    """
    encoder, head = _build_llm_encoder(
        cfg,
        actor_model_path,
        actor_hidden,
        actor_gpu_ids=actor_gpu_ids,
        actor_module=actor_module,
        actor_tokenizer=actor_tokenizer,
    )
    if head is not None:
        payload = load_projector_init(head, cfg, actor_model_path)
        if payload is not None:
            if payload.get("source_kind") == "native_stage2":
                print(f"[dyad] native AgenticRL projector restored from {payload['source']} "
                      f"(manifest {payload['manifest_sha256']})", flush=True)
            else:
                print(
                    f"[dyad] projector initialised from {cfg.projector_init} "
                    f"(Alignment step {payload.get('selected_step', '?')} of {payload.get('steps', '?')})",
                    flush=True,
                )
    return encoder, head


def _build_llm_encoder(
    cfg: LlmEncoderConfig,
    actor_model_path: str,
    actor_hidden: int,
    *,
    actor_gpu_ids: tuple[int, ...] = (),
    actor_module=None,
    actor_tokenizer=None,
):
    """Instantiate `(encoder, residual_head)`, or `(None, None)` when the encoder is off.

    Two hosting modes, and the difference is which GPUs the backbone lands on:

      `remote=False` in-process. The encoder shares the worker's device. Fine for a single-GPU
                     smoke test, but it is not task 2's 2+2 split -- both halves compete for the
                     same memory.
      `remote=True`  its own Ray actor with its own `num_gpus`. This is the split task 2 asks for;
                     see models/action_encoder.py.

    The GPU check is not decoration. `LlmEncoderConfig.gpu_ids` and the policy LLM backbone's must not overlap:
    an overlap does not raise anywhere, it just makes the encoder and the policy LLM backbone compete for the same
    memory, and it surfaces as an OOM in whichever happens to allocate last -- which is usually not
    the one that was misconfigured.
    """
    if not cfg.enabled:
        return None, None

    from agent_system.policies.dyad.models.encoder_config import Backbone

    backbone = Backbone(cfg.backbone) if not isinstance(cfg.backbone, Backbone) else cfg.backbone
    if backbone.is_actor:
        # The encoder reads the **policy LLM backbone**. One model in the whole run, so there is nothing to
        # place on separate GPUs and nothing to host in a Ray actor -- the backbone is already
        # wherever the policy LLM backbone is.
        if cfg.remote or cfg.gpu_ids:
            raise ValueError(
                "backbone=policy_lm reads the policy itself, so it has no backbone of its own to "
                "place. DYAD_ENCODER_REMOTE / DYAD_ENCODER_GPU_IDS only apply to "
                "backbone=encoder_lm, which loads a second model. Drop them, or switch backbone."
            )
        if actor_module is None or actor_tokenizer is None:
            raise ValueError(
                "backbone=policy_lm needs the policy LM's module and tokenizer passed in -- that is the "
                "whole point of it. Loading a second copy here would give the run two LLMs while "
                "the configuration says one, and the only symptom would be the memory."
            )
        from agent_system.policies.dyad.models.action_head import DirectActionHead
        from agent_system.policies.dyad.models.action_encoder import LlmActionEncoder

        # The device follows the policy LLM backbone, not cfg.device. The encoder *is* the policy LLM backbone here, and at
        # construction time the policy LLM backbone is still on CPU -- FSDP moves it later. Sending the input ids
        # to cfg.device ("cuda" by default) while the shared embedding table is on CPU gives
        #   RuntimeError: Expected all tensors to be on the same device, but got index is on cuda:0,
        #   different from other tensors on cpu (... wrapper_CUDA__index_select)
        # from inside the embedding lookup, which says nothing about whose device was wrong.
        shared_device = str(next(actor_module.parameters()).device)
        encoder = LlmActionEncoder(
            cfg.resolved_model_path(actor_model_path),
            device=shared_device,
            max_length=cfg.max_length,
            representation=cfg.representation,
            backbone=actor_module,
            tokenizer=actor_tokenizer,
            # Never freeze here: this backbone is the policy LLM backbone, and the training_schedule decides whether it
            # trains. Freezing it as a side effect of building the encoder would silently turn an
            # training_schedule=joint_optimization run into one that trains nothing.
            freeze=False,
        )
        residual_head = DirectActionHead(
            cfg.projector,
            encoder.hidden_size,
            actor_hidden,
            scale=cfg.scale,
            projector_kwargs=cfg.projector_kwargs,
        ).to(shared_device)
        return encoder, residual_head

    if cfg.remote:
        if cfg.gpu_ids:
            # Ray owns the assignment in this mode. Honouring an explicit pin here is impossible
            # (the ordinals are not addressable inside the policy LLM backbone) and ignoring it silently would
            # leave a run configured for GPUs 2,3 quietly running somewhere else.
            raise ValueError(
                f"DYAD_ENCODER_GPU_IDS={sorted(cfg.gpu_ids)} is set together with "
                "DYAD_ENCODER_REMOTE=1, but Ray assigns the policy LM's GPUs itself and remaps "
                "CUDA_VISIBLE_DEVICES inside it, so an absolute ordinal cannot be honoured. "
                "Use DYAD_ENCODER_NUM_GPUS to say how many, and drop DYAD_ENCODER_GPU_IDS."
            )
        from agent_system.policies.dyad.rollout.encoder_worker import build_remote_encoder

        return build_remote_encoder(
            cfg,
            actor_model_path,
            actor_hidden,
            # The projector trains with the policy LLM backbone, so it belongs on the policy LLM backbone's device, not the
            # backbone's. cfg.device is the local device here.
            device=cfg.device,
        )

    overlap = set(cfg.gpu_ids) & set(actor_gpu_ids)
    if overlap:
        raise ValueError(
            f"encoder GPUs {sorted(cfg.gpu_ids)} overlap the policy LM's {sorted(actor_gpu_ids)} on "
            f"{sorted(overlap)}. Task 2 asks for a disjoint 2+2 split; sharing does not fail here, "
            "it fails later as an OOM that points at the wrong component."
        )

    import torch

    from agent_system.policies.dyad.models.action_head import DirectActionHead
    from agent_system.policies.dyad.models.action_encoder import LlmActionEncoder

    device = cfg.device
    if cfg.gpu_ids and device.startswith("cuda"):
        ordinal = cfg.gpu_ids[0]
        visible = torch.cuda.device_count()
        if ordinal >= visible:
            # Ray remaps CUDA_VISIBLE_DEVICES per worker, so inside a policy LLM backbone the only visible
            # device is usually ordinal 0 regardless of which physical GPU it is. An absolute index
            # from the launch environment is meaningless here, and passing it through produces
            # `CUDA error: invalid device ordinal` from deep inside .to() -- a message that says
            # nothing about where the number came from.
            raise ValueError(
                f"DYAD_ENCODER_GPU_IDS asks for cuda:{ordinal} but this process can only see "
                f"{visible} device(s). Ray gives each worker its own remapped view, so absolute "
                "GPU indices do not survive into a policy LM. Either drop DYAD_ENCODER_GPU_IDS (the "
                "encoder then shares the worker's device) or host the encoder as its own Ray actor "
                "with its own GPU allocation."
            )
        device = f"cuda:{ordinal}"
    dtype = getattr(torch, cfg.dtype, torch.bfloat16)

    encoder = LlmActionEncoder(
        cfg.resolved_model_path(actor_model_path),
        device=device,
        dtype=dtype,
        max_length=cfg.max_length,
        representation=cfg.representation,
    )
    residual_head = DirectActionHead(
        cfg.projector,
        encoder.hidden_size,
        actor_hidden,
        scale=cfg.scale,
        projector_kwargs=cfg.projector_kwargs,
    ).to(device)
    return encoder, residual_head
