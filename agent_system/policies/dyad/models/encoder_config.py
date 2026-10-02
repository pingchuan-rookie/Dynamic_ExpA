# Copyright 2025 ExpA_sys
"""The setting space of Dynamic Dyad's action encoder, and what makes a combination legal.

Prose version with the reasoning:
`../dynamic-expa-design/experiments/dynamic_dyad_experiments.md`. Six dimensions:

    backbone     encoder_lm | policy_lm          what the projector pools over
    projector      attention | mean             how a variable-length sequence becomes one vector
    pooling      final_layer_hidden_states | input_embeddings
    description  mcp | natural_language       how the action set is described to the encoder
    training_schedule    joint_optimization | frozen_llm_adaptation | policy_lm_only   which side this run updates
    encoder      projector_only | projector_and_encoder_llm | not_applicable

`backbone` and `projector` are deliberately two switches rather than one four-valued one. They are
two independent questions -- "does the encoder load its own model" and "does the projector have
parameters" -- and an ablation that changes one of them should read as changing one of them.

The dimensions are **not** orthogonal, and the illegal corners are the point of this module:

  - Mean pooling has no parameters. Combined with "train the projector only" there is nothing to
    train, and if the training_schedule also freezes the policy LLM backbone, the whole run updates nothing.
  - Reading the policy LLM backbone as the backbone means the run holds one model, so "train the encoder LLM backbone"
    does not exist -- training that backbone is training the policy LLM backbone, which the training_schedule decides.

Refusing those matters more than it looks. Every one of them *runs*: a frozen policy LLM backbone plus a
parameterless projector produces a perfectly ordinary training loop where nothing is updated,
pg_loss still moves, and the metrics look normal for as long as you care to watch. This module is
where that becomes an error at configuration time instead of a wasted run.
"""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Validate method-specific encoder settings before model construction.
# Extension point: install_dyad_encoder -> EncoderSetting / LlmEncoderConfig

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Backbone(str, Enum):
    """Whose hidden states the projector pools over."""

    ENCODER_LM = "encoder_lm"     # a second model, loaded just for this
    POLICY_LM = "policy_lm"       # the policy itself; the run then holds one model, not two

    @property
    def is_policy_lm(self) -> bool:
        return self is Backbone.POLICY_LM

    @property
    def is_actor(self) -> bool:
        """Compatibility alias retained for external callers."""
        return self.is_policy_lm


class Projector(str, Enum):
    """How a variable-length token sequence becomes one vector."""

    ATTENTION = "attention"   # a learnable query attends over the sequence
    MEAN = "mean"             # a plain average -- **zero parameters**

    @property
    def has_parameters(self) -> bool:
        return self is Projector.ATTENTION


class Representation(str, Enum):
    """What the projector pools over."""

    FINAL_LAYER_HIDDEN_STATES = "final_layer_hidden_states"   # the final layer, the whole backbone runs
    INPUT_EMBEDDINGS = "input_embeddings"       # the embedding layer only, no transformer block


class Description(str, Enum):
    """How the action set is described to the encoder."""

    MCP = "mcp"
    NATURAL_LANGUAGE = "natural_language"


class TrainingSchedule(str, Enum):
    """Which side of the system this run updates. Single-phase, all three.

    `JOINT_OPTIMIZATION` is the main experiment: the action encoder and the policy LLM backbone learn together, which is
    the setting the method is actually claimed in. The other two are its ablations -- each freezes
    one side to ask what that side contributed.

    `POLICY_LM_ONLY` freezes the action head, so it is only meaningful with a head worth freezing:
    point `DYAD_ENCODER_PROJECTOR_INIT` at an alignment run. Without one the ablation asks "how far do
    you get with a randomly initialised action interface", which is a different question and rarely
    the intended one.

    There used to be a fourth, `encoder_then_policy_lm`: phase 1 encoder, phase 2 policy LLM backbone, with a
    switch at the midpoint. It was dropped on 2026-09-01 when `joint_optimization` became the main experiment --
    a two-phase schedule is a third answer to "what trains" that neither of the ablations isolates,
    and its switch step was the only thing in this module that depended on `total_training_steps`.
    """

    JOINT_OPTIMIZATION = "joint_optimization"                      # the main experiment: both sides learn together
    FROZEN_LLM_ADAPTATION = "frozen_llm_adaptation"        # policy LLM backbone frozen, action encoder trains
    POLICY_LM_ONLY = "policy_lm_only"    # action head frozen, policy LLM backbone trains

    @classmethod
    def _missing_(cls, value):
        # Saved configurations predate the paper's names for these two settings.
        return {"joint": cls.JOINT_OPTIMIZATION, "encoder_only": cls.FROZEN_LLM_ADAPTATION}.get(value)


class EncoderTraining(str, Enum):
    """What trains **inside** the action encoder.

    `NOT_APPLICABLE` is not a third scheme -- it is how a parameterless projector over the policy LLM backbone
    says the dimension does not apply. Forcing it to pick one of the other two would make every
    such configuration illegal, which reads as "unsupported" when the truth is "it trains through
    the policy LLM backbone".
    """

    ADAPTER_ONLY = "projector_only"                          # encoder LLM backbone frozen
    ADAPTER_AND_ENCODER_LM = "projector_and_encoder_lm"    # the encoder LLM backbone trains too
    NOT_APPLICABLE = "not_applicable"                      # no encoder-side parameter exists


def _parse(enum_cls, value, field: str):
    """Parse an enum value or name case-insensitively.

    Reject retired spellings explicitly so stale configurations cannot silently
    select a different training setup.
    """
    if isinstance(value, enum_cls):
        return value
    raw = str(value).strip()
    for member in enum_cls:
        if raw.lower() in (member.value.lower(), member.name.lower()):
            return member
    known = ", ".join(m.value for m in enum_cls)
    raise ValueError(f"{field}={value!r} is not one of: {known}")


@dataclass(frozen=True)
class EncoderSetting:
    """One fully specified action-encoder configuration.

    Constructed through `build` (or `from_env`), never by hand, so the legality check cannot be
    skipped.
    """

    backbone: Backbone
    projector: Projector
    representation: Representation
    description: Description
    training_schedule: TrainingSchedule
    encoder_training: EncoderTraining

    # ---- derived, so callers never re-derive them differently -------------------------
    @property
    def needs_separate_encoder_lm(self) -> bool:
        return self.backbone is Backbone.ENCODER_LM

    @property
    def trains_projector(self) -> bool:
        """Does this *setting* give the projector something to train?

        Setting-level, not schedule-level: it answers "is there a trainable projector at all",
        which `build_head.apply_training_schedule` then ANDs with "does this schedule train the
        encoder side". Two questions, composed at the point of use -- see build_head.py:405.
        """
        return (
            self.projector.has_parameters
            and self.encoder_training is not EncoderTraining.NOT_APPLICABLE
        )

    @property
    def trains_actor(self) -> bool:
        return self.training_schedule in (TrainingSchedule.JOINT_OPTIMIZATION, TrainingSchedule.POLICY_LM_ONLY)

    @property
    def trains_action_head(self) -> bool:
        """Does this run update the action head?

        This is what decides whether the head is recomputed from the projector with a live graph or
        stays a frozen `nn.Linear` (`agent_system/policies/dyad/actions/schema_config.py`, `dyad_workers._install_weight_fn`).
        It used to be a separate `TRAIN_TARGET` env var, which could contradict the schedule: with
        `joint_optimization` + `TRAIN_TARGET=policy_lm` the projector was unfrozen and received no gradient --
        a run that trains nothing while every metric looks normal. One dimension cannot disagree
        with itself.
        """
        return self.training_schedule in (TrainingSchedule.JOINT_OPTIMIZATION, TrainingSchedule.FROZEN_LLM_ADAPTATION)

    @property
    def trains_encoder_lm(self) -> bool:
        return (
            self.needs_separate_encoder_lm
            and self.encoder_training is EncoderTraining.ADAPTER_AND_ENCODER_LM
        )

    def describe(self) -> str:
        where = "the policy LM" if self.backbone.is_policy_lm else "its own encoder LM"
        return (
            f"{self.projector.value} pooling over {where} ({self.representation.value}) "
            f"description={self.description.value} training_schedule={self.training_schedule.value} "
            f"encoder_training={self.encoder_training.value}"
        )

    # ---- the part that matters --------------------------------------------------------
    def _validate(self) -> None:
        # Reading the policy LLM backbone: there is no second model, so "train the encoder LLM backbone" has no object.
        if self.backbone.is_actor and self.encoder_training is EncoderTraining.ADAPTER_AND_ENCODER_LM:
            raise ValueError(
                "backbone=policy_lm cannot train an encoder LLM: there is none. The projector pools over "
                "the policy itself, so training that backbone means training the policy LM, which the "
                "training_schedule (joint_optimization / policy_lm_only) decides -- not this switch. Use "
                "projector_only, or backbone=encoder_lm for a separate model."
            )
        # A parameterless projector over the policy LLM backbone has no encoder-side parameter under any scheme,
        # so the dimension must be declared not_applicable rather than answered. projector_only
        # would claim "the projector trains" about an empty projector.
        parameterless_over_actor = self.backbone.is_actor and not self.projector.has_parameters
        if parameterless_over_actor and self.encoder_training is not EncoderTraining.NOT_APPLICABLE:
            raise ValueError(
                "mean pooling over the policy LM has no encoder-side parameter at all, so "
                f"encoder_training must be {EncoderTraining.NOT_APPLICABLE.value}; "
                f"got {self.encoder_training.value}. It learns through the policy LM (training_schedule=joint_optimization)."
            )
        if not parameterless_over_actor and self.encoder_training is EncoderTraining.NOT_APPLICABLE:
            raise ValueError(
                f"encoder_training={EncoderTraining.NOT_APPLICABLE.value} is only for mean pooling "
                "over the policy LM. This setting has an encoder-side parameter to decide about, so "
                "pick projector_only or projector_and_encoder_llm."
            )
        # Mean pooling over the policy LLM backbone can only learn through the policy LLM backbone, so the schedule has
        # to train the policy LLM backbone. `frozen_llm_adaptation` freezes it and leaves nothing else trainable.
        if parameterless_over_actor and not self.trains_actor:
            raise ValueError(
                f"mean pooling over the policy LM + training_schedule={self.training_schedule.value} is not a training "
                "configuration: the projector has zero parameters and the backbone is the policy LM, so "
                "this schedule would freeze the only thing that could learn. It would still run "
                "-- pg_loss moves, the metrics look normal, and nothing is updated. Use "
                "training_schedule=joint_optimization or policy_lm_only, or projector=attention."
            )
        # Claiming to train the encoder LLM backbone while the schedule freezes the whole encoder side is the
        # same shape of contradiction this module exists for: it runs, the policy LLM backbone learns, and the
        # encoder LLM backbone everyone believes is being fine-tuned never receives a gradient.
        # `projector_only` is fine here -- it is the neutral value, and under a schedule that does
        # not train the encoder the dimension is simply moot rather than contradicted.
        if not self.trains_action_head and self.encoder_training is EncoderTraining.ADAPTER_AND_ENCODER_LM:
            raise ValueError(
                f"training_schedule={self.training_schedule.value} freezes the action encoder, so "
                f"encoder_training={EncoderTraining.ADAPTER_AND_ENCODER_LM.value} claims to train a "
                "model this run never updates. Use projector_only, or a schedule that trains the "
                "encoder (joint_optimization / frozen_llm_adaptation)."
            )
        # A schedule that trains the action encoder **and nothing else** needs the action encoder to
        # be trainable. `frozen_llm_adaptation` is the only one: under `joint_optimization` the policy LLM backbone carries the
        # learning even when the encoder side has no parameters of its own (mean pooling over the
        # policy LLM backbone -- §2.4b), which is a legal and deliberate configuration.
        if self.trains_action_head and not self.trains_actor and not (
                self.trains_projector or self.trains_encoder_lm):
            raise ValueError(
                f"training_schedule={self.training_schedule.value} trains only the action encoder, but this "
                f"setting makes it untrainable (projector={self.projector.value} has "
                f"{'parameters' if self.projector.has_parameters else 'no parameters'}, "
                f"encoder_training={self.encoder_training.value}). Nothing would be updated at all."
            )

    @classmethod
    def build(
        cls,
        backbone="encoder_lm",
        projector="attention",
        representation="final_layer_hidden_states",
        description="mcp",
        training_schedule="joint_optimization",
        encoder_training="projector_only",
    ) -> "EncoderSetting":
        setting = cls(
            backbone=_parse(Backbone, backbone, "backbone"),
            projector=_parse(Projector, projector, "projector"),
            representation=_parse(Representation, representation, "representation"),
            description=_parse(Description, description, "description"),
            training_schedule=_parse(TrainingSchedule, training_schedule, "training_schedule"),
            encoder_training=_parse(EncoderTraining, encoder_training, "encoder_training"),
        )
        setting._validate()
        return setting

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "EncoderSetting":
        """Read the setting from the environment.

        Env vars rather than the Hydra config for the same reason the rest of action_encoder does
        it: the encoder is built inside Ray actors and inside the vLLM engine fork, neither of
        which is handed the trainer's config object. `main_dyad` injects the whole `DYAD_` prefix
        into `runtime_env.env_vars`, so these arrive intact.
        """
        source = env if env is not None else os.environ
        return cls.build(
            backbone=source.get("DYAD_ENCODER_BACKBONE", "encoder_lm"),
            projector=source.get("DYAD_ENCODER_PROJECTOR", "attention"),
            representation=source.get("DYAD_ENCODER_REPRESENTATION", "final_layer_hidden_states"),
            description=source.get("DYAD_ENCODER_DESCRIPTION", "mcp"),
            training_schedule=source.get("DYAD_TRAINING_SCHEDULE", "frozen_llm_adaptation"),
            encoder_training=source.get("DYAD_ENCODER_TRAINING", "projector_only"),
        )


def legal_settings() -> list[EncoderSetting]:
    """Every combination that passes `_validate`. Used by the tests and by the matrix docs."""
    out = []
    for backbone in Backbone:
        for projector in Projector:
            for representation in Representation:
                for description in Description:
                    for training_schedule in TrainingSchedule:
                        for training in EncoderTraining:
                            try:
                                out.append(EncoderSetting.build(
                                    backbone, projector, representation, description, training_schedule, training))
                            except ValueError:
                                continue
    return out


# Keep runtime configuration dependency-free for validation and lightweight environment workers.

@dataclass(frozen=True)
class LlmEncoderConfig:
    """Encoder hosting, representation, direct-head projection, and training configuration.

    An empty model_path resolves to the policy model path.
    The backbone setting determines whether weights are shared or separately loaded.
    """

    enabled: bool = False
    model_path: str = ""               # "" = same path as the policy LLM backbone
    scale: str = "unit"                # every action row has norm 1; see llm_encoder._apply_scale
    max_length: int = 1024
    dtype: str = "bfloat16"
    device: str = "cuda"
    # Which GPUs the encoder gets, as absolute ordinals. Must not overlap the policy LLM backbone's -- an overlap
    # does not raise, it just makes both fight for memory and shows up as an OOM in whichever happens
    # to allocate last.
    #
    # Only meaningful when `remote` is False, i.e. when the encoder shares the worker's process.
    # Inside a Ray worker an absolute ordinal is not addressable at all (Ray remaps
    # CUDA_VISIBLE_DEVICES per worker), which is what `remote` exists to solve.
    gpu_ids: tuple[int, ...] = ()
    # The structural dimensions of the experiment design document, carried on this object rather
    # than on a second one. The encoder is constructed in two places -- the FSDP trainer worker and
    # the vLLM engine fork -- and every extra object that has to reach both is another place for one
    # of them to be built with a different setting than the other, which produces two different
    # `action_head`s and no error.
    backbone: str = "encoder_lm"        # encoder_lm | policy_lm
    projector: str = "attention"          # attention | mean
    representation: str = "final_layer_hidden_states"  # final_layer_hidden_states | input_embeddings
    description: str = "mcp"            # mcp | natural_language
    # Host the frozen backbone in its own Ray actor with its own GPU allocation -- the real form of
    # task 2's "2 actor + 2 encoder" split. See models/action_encoder.py for why only the frozen half
    # moves and why that leaves AGENTS.md section 1 untouched.
    remote: bool = False
    num_gpus: int = 0                  # GPUs for that actor; must be >= 1 when remote is True
    actor_name: str = "dyad_action_encoder"   # named so every FSDP worker shares one backbone
    projector_kwargs: Optional[dict] = None
    # Independent projector learning rate; None shares the policy LLM backbone rate.
    projector_lr: Optional[float] = None
    # Path to a Alignment `projector.pt`. Empty means "start from this module's own initialisation",
    # which is what every run did before Agentic RL existed and is still what the ablations want.
    #
    # Loading it is the only thing that makes Agentic RL a *second* stage rather than a fresh fit. The
    # failure mode it guards against is silent by construction: a run with no path set trains a
    # projector from scratch, reports a falling loss and a growing L2 displacement, and looks
    # exactly like a run that inherited one. See `build_head.load_projector_init` for the four
    # checks that make loading it either correct or loud.
    projector_init: str = ""

    def resolved_model_path(self, actor_model_path: str) -> str:
        return self.model_path or actor_model_path
