"""Dyad's worker: verl 0.9's `ActorRolloutRefWorker` plus what the action head needs.

Before the 0.9 port this file was a ~2700-line fork of verl 0.7's `fsdp_workers.py`, because 0.7
had no way to change the policy LLM backbone's forward without replacing the worker. 0.9 provides three hooks
that make the fork unnecessary:

    model_config.model_type   selects the engine class through `EngineRegistry`
    TrainingWorker.set_loss_fn injects the objective
    engine.prepare_model_*    overrides the forward

So Dyad is now a subclass with four additions, and everything else -- FSDP, offload, checkpointing,
rollout, weight sync -- is upstream's.

  1. `model_type = dyad_language_model`, which resolves to `DyadFSDPEngineWithLMHead`. That engine
     installs the action head and the encoder before the FSDP wrap (`dyad_model_setup`).
  2. `dyad_ppo_loss` instead of `ppo_loss`.
  3. The direct action head is materialized at the start of every `compute_log_prob`.
     Sampling and replay must use matching projector parameters, encoder representations,
     and scale references; otherwise their distributions and the PPO ratio disagree.
  4. The training_schedule's phase is applied at the start of every `update_actor`.
"""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Specialize the official worker at model setup, replay, and actor-update boundaries.
# Extension point: PPOTrainer._init_resource_pool_mgr -> ActorRolloutRefWorker subclass

from agent_system.policies.dyad.actions.task_context import dynamic_actions_enabled
import os
from contextlib import contextmanager
from functools import partial
from typing import Any, Optional

import torch
from omegaconf import DictConfig, open_dict
from tensordict import TensorDict

from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.profiler import DistProfiler
from verl.workers.config import ActorConfig
from verl.workers.engine.base import BaseEngineCtx
from verl.workers.engine_workers import ActorRolloutRefWorker, _with_routing_replay_flag

from agent_system.utils.logging import get_dyad_logger
from agent_system.policies.dyad.training.ppo_loss import dyad_ppo_loss

dyad_logger = get_dyad_logger()

# Importing the engine module is what registers `dyad_language_model` with `EngineRegistry`.
# Without this import the registry lookup in `TrainingWorker.__init__` raises
# "Unknown model_type: dyad_language_model" from inside a Ray actor, several frames away from
# anything naming Dyad.
import agent_system.policies.dyad.training.dyad_engine  # noqa: F401


class DyadActorRolloutRefWorker(ActorRolloutRefWorker):
    """0.9's hybrid worker with Dyad's engine and loss."""

    # actor.strategy=dyad selects this worker; model_type selects its engine on the FSDP backend.
    DYAD_STRATEGY = "dyad"
    # Use the same distributed backend as the baseline.
    ENGINE_BACKEND = "fsdp"

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # Decided here rather than in the training scripts: which engine the policy LLM backbone needs is a
        # consequence of running Dyad, not a per-run choice. The reference actor inherits the same
        # model_type, which is safe -- DyadFSDPEngineWithLMHead keys the Dyad path on
        # `engine_config.forward_only`, and the ref engine is forward_only by config.
        with open_dict(self.config.model):
            self.config.model.model_type = "dyad_language_model"

        # Replace the worker selector with a registered backend for both actor and reference engines.
        for section in ("actor", "ref"):
            node = self.config.get(section)
            if node is None or node.get("strategy") != self.DYAD_STRATEGY:
                continue
            with open_dict(node):
                node.strategy = self.ENGINE_BACKEND

        super().init_model()

        if not self._is_actor:
            return

        # Replace the objective upstream just installed. A second ActorConfig instance is fine:
        # `ppo_loss` is no longer called, and each loss writes its own `global_batch_info`.
        actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
        actor_config.model_config = omega_conf_to_dataclass(self.config.model)
        # >>> DYAD-REPLACE(init_model.loss_binding)
        # Override the objective after the official model/optimizer lifecycle has completed.
        # Original binding installed by super().init_model() (now superseded):
        # verl v0.9.0, workers/engine_workers.py L637:
        # self.loss_fn = partial(ppo_loss, config=actor_config)
        self.loss_fn = partial(dyad_ppo_loss, config=actor_config)
        self.actor.set_loss_fn(self.loss_fn)
        # <<< DYAD-REPLACE(init_model.loss_binding)

        # The engine installed these while building the module; the Dyad helpers below read them
        # under the names they have always used.
        state = getattr(self.actor.engine, "dyad_state", None)
        if state is None:
            raise RuntimeError(
                "The Dyad actor engine did not install an action head. This means the engine "
                "resolved to something other than DyadFSDPEngineWithLMHead, or it was built with "
                "forward_only=True. Training would silently optimize the base-token policy."
            )
        self.dyad_action_config = state.action_config
        self.dyad_llm_encoder_config = state.encoder_config
        self.dyad_setting = state.setting
        self.dyad_llm_encoder = state.llm_encoder
        self.dyad_llm_backbone = state.llm_backbone
        self.dyad_residual_head = state.residual_head
        self.dyad_encoder_prompts = state.encoder_prompts
        self._dyad_param_counts = state.param_counts

        if (self.dyad_setting.trains_encoder_lm
                and self.config.rollout.checkpoint_engine.backend == "delta_sharded"):
            raise ValueError("Trainable encoder requires full weight synchronization, not delta_sharded")
        self._apply_schedule_lr_scale()

    # ---- accessors onto the 0.9 engine ------------------------------------------------------
    #
    # The helpers below were written against 0.7's worker, which owned `actor_module_fsdp` and
    # `actor_optimizer` directly. On 0.9 the engine owns both. These three keep that difference in
    # one place instead of spread through the helpers.

    @property
    def _dyad_actor_module(self):
        return self.actor.engine.module

    @property
    def _dyad_actor_optimizer(self):
        return getattr(self.actor.engine, "optimizer", None)

    @property
    def _dyad_tokenizer(self):
        return self.actor.engine.model_config.tokenizer

    def _dyad_action_head(self):
        """The action head, wherever FSDP has put it.

        Registered on the module before the wrap, so after wrapping it is one level down behind
        `_fsdp_wrapped_module`. Looked up rather than cached because the reference actor has none.
        """
        module = self._dyad_actor_module
        for candidate in (module, getattr(module, "_fsdp_wrapped_module", None)):
            if candidate is None:
                continue
            head = getattr(candidate, "action_head", None)
            if head is not None:
                return head
        return None

    # ---- the two per-step Dyad obligations ---------------------------------------------------

    @contextmanager
    def _dyad_actor_device_context(self, data: TensorDict, *, mode: str):
        """Extend engine placement to Dyad's pre/post-batch work.

        TrainingWorker owns the FSDP train/eval contexts, but they only cover its batch call.
        Use the base placement context outside it, suppressing only the inner load/offload.
        Do not nest FSDP eval contexts: each exit reshards the root independently.
        The caller's manual-placement flag and metadata survive success and failure alike.
        """
        had_flag = "disable_auto_offload" in data.keys()
        original_flag = data.get("disable_auto_offload") if had_flag else None
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        try:
            with BaseEngineCtx(self.actor.engine, mode=mode, disable_auto_offload=disable_auto_offload):
                tu.assign_non_tensor(data, disable_auto_offload=True)
                yield
        finally:
            if had_flag:
                data["disable_auto_offload"] = original_flag
            else:
                data.pop("disable_auto_offload", None)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    @_with_routing_replay_flag(enabled=True)
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        # Materialize the direct head for replay using the synchronized encoder and projector.
        # The rollout side rebuilds its corresponding head after each weight sync.
        with self._dyad_actor_device_context(data, mode="eval"):
            self._reinit_actor_action_head_from_lm_head()
            output = self.actor.infer_batch(data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    @_with_routing_replay_flag(enabled=True)
    def update_actor(self, data: TensorDict) -> TensorDict:
        # All three schedules are single phase, so the freezing and the idle learning rates are
        # decided once at construction (`_apply_schedule_lr_scale`) and hold for the run. This used
        # to re-derive the phase from `global_steps` every step, because `encoder_then_policy_lm`
        # switched at the midpoint and the switch could only happen here.
        # V1 may issue several mini-batch RPCs against one rollout's replay anchor.
        # The materialized encoder cache stays fixed until the last such update.
        update_dyad_cache = tu.get_non_tensor_data(data, key="update_dyad_cache", default=True)
        with self._dyad_actor_device_context(data, mode="train"):
            output = self.actor.train_mini_batch(data=data)

            # Refresh before offload, while the projector buffers still share the model's device.
            # Encoder gradients were reduced before gradient clipping and the optimizer step.
            if (update_dyad_cache and getattr(self, "dyad_setting", None) is not None
                    and self.dyad_setting.trains_encoder_lm):
                refreshed = self._dyad_refresh_encoder_cache()
                if self.rank == 0:
                    print(
                        f"[Dyad] encoder LLM post-step: gradients reduced before clipping; "
                        f"buffer_refreshed={refreshed}",
                        flush=True,
                    )
        return output.cpu() if output is not None else None

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def set_dyad_val_mode(self, on: bool):
        """Cross-env validation: switch the rollout-side Dyad action head/config.

        Forwarded to the vLLM rollout and then the Dyad model runner. No-op when the runner was not
        configured with a separate validation schema.
        """
        setter = getattr(self.rollout, "set_dyad_val_mode", None)
        if setter is not None:
            await setter(bool(on))
        return True

    # ---- Dyad helpers, carried over unchanged except for the three accessors above ------------

    def _apply_schedule_lr_scale(self) -> None:
        """Zero the learning rate of the side this schedule does not train. Once, at construction.

        Belt and braces on top of `apply_training_schedule`'s `requires_grad` freezing, which
        already stops the optimizer (a frozen parameter's grad stays None and AdamW skips it). The
        learning-rate scale is what makes the freezing survive anything that hands a gradient to a
        parameter it should not, and it is also where the per-side report comes from.

        **Why a learning-rate scale rather than more `requires_grad`.** Toggling `requires_grad` on
        FSDP-managed parameters *after* the wrap breaks FSDP's bookkeeping: resharding re-installs
        each original parameter as a view of the flat parameter, and a view whose `requires_grad`
        was flipped underneath it is no longer a leaf, so `register_parameter` refuses it --

            ValueError: Cannot assign non-leaf Tensor to parameter 'weight'
              ... _flat_param.py::_use_sharded_views -> _safe_setattr_tensor_or_param

        measured on a real frozen_llm_adaptation run. So freezing is decided once before the wrap, and this
        expresses the same decision on the optimizer side, where it is safe to state.

        This used to be `_apply_encoder_recipe(step, total_steps)`, called from `update_actor` every
        step, because `encoder_then_policy_lm` switched phase at the midpoint of the run. That
        schedule was dropped on 2026-09-01 and all three survivors are single phase, so the answer
        cannot change over a run and is settled here.
        """
        setting = getattr(self, "dyad_setting", None)
        head = getattr(self, "dyad_residual_head", None)
        if setting is None or head is None:
            return

        actor_idle = not setting.trains_actor
        adapter_idle = not setting.trains_action_head
        applied = self._set_param_group_scale(actor_idle=actor_idle, adapter_idle=adapter_idle)
        if self.rank == 0:
            print(
                f"[Dyad] training_schedule {setting.training_schedule.value}: "
                f"policy_lm={'idle' if actor_idle else 'training'} "
                f"encoder={'idle' if adapter_idle else 'training'} trainable={applied}",
                flush=True,
            )

    def _set_param_group_scale(self, *, actor_idle: bool, adapter_idle: bool) -> dict:
        """Scale each optimizer group's learning rate by 0 or 1 for this run's schedule.

        The scale multiplies whatever the scheduler produces, rather than replacing the learning
        rate: overwriting `group["lr"]` would be undone by the next `scheduler.step()`, so the
        idling would silently stop being in effect after one step.

        The counts are read from the **modules**, not from the optimizer's groups. Under FSDP the
        group holds this rank's shard, so a rank whose projector shard happens to be empty reports
        `projector: 0` for a perfectly normal phase -- which is exactly what the construction-time
        line does not say, and two numbers for the same quantity that disagree are worse than one.
        """
        optimizer = self._dyad_actor_optimizer
        if optimizer is None:
            return {}
        projector_ids = {id(p) for p in self.dyad_residual_head.parameters() if p.requires_grad}
        for group in optimizer.param_groups:
            is_adapter = group.get("dyad_group") in {"projector", "encoder"} or any(
                id(p) in projector_ids for p in group["params"]
            )
            group["dyad_phase_scale"] = 0.0 if (adapter_idle if is_adapter else actor_idle) else 1.0

        # Counted from the snapshot taken before the FSDP wrap, not from the live modules.
        # Outside a forward/backward FSDP keeps the original parameters resharded -- their views are
        # empty and their `requires_grad` is not meaningful -- so counting here returns 0 for both
        # sides on a run that is training perfectly well. Measured: an frozen_llm_adaptation run whose
        # projector had grad_norm 0.909 reported `projector: 0`.
        snapshot = getattr(self, "_dyad_param_counts", {"policy_lm": 0, "projector": 0})
        return {"policy_lm": 0 if actor_idle else snapshot["policy_lm"],
                "projector": 0 if adapter_idle else snapshot["projector"]}

    def _reinit_actor_action_head_from_lm_head(self) -> None:
        """Materialize the current direct head under FSDP without retaining a gradient graph.

        The forward hook separately supplies a differentiable head when the schedule trains it.
        Sampling and replay must share the projector parameters and encoder representations.
        """
        if not self._is_actor:
            return
        if dynamic_actions_enabled():
            # Per-microbatch heads are built inside the forward hook under FSDP.
            return
        head = self._dyad_action_head()
        action_config = getattr(self, "dyad_action_config", None)
        if head is None or action_config is None:
            return
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from agent_system.policies.dyad import models as action_encoder

        module = self._dyad_actor_module
        inner = getattr(module, "_fsdp_wrapped_module", module)
        with FSDP.summon_full_params(module, writeback=True, recurse=True):
            lm_head = inner.get_output_embeddings()
            if lm_head is None or not hasattr(lm_head, "weight"):
                raise RuntimeError("Dyad actor head re-derive requires model output embeddings (lm_head).")
            # Under no_grad: this materialises frozen action rows, and the gradient path to
            # the projector is the forward hook's `weight_fn`, not this. Without it the projector's
            # forward builds a graph **on the parameters FSDP has summoned**, and `writeback=True`
            # then tries to re-install a non-leaf view on exit:
            #     ValueError: Cannot assign non-leaf Tensor to parameter 'weight'
            #       ... _flat_param.py::_use_sharded_views
            # Measured: it fires the first time a *trainable* projector is read from in here, which is
            # every run with a training_schedule that trains the action encoder.
            with torch.no_grad():
                new_w = action_encoder.build_action_head(
                    action_config,
                    lm_head.weight,
                    self._dyad_tokenizer,
                    llm_cfg=getattr(self, "dyad_llm_encoder_config", None),
                    encoder=getattr(self, "dyad_llm_encoder", None),
                    residual_head=getattr(self, "dyad_residual_head", None),
                )
                head.weight.copy_(new_w.to(device=head.weight.device, dtype=head.weight.dtype))
                if head.bias is not None:
                    head.bias.zero_()
            self._install_action_head_weight_fn(inner, action_config, lm_head.weight)
            # Printed, not just logged: this is one of the two halves of the AGENTS.md section 1
            # pairing, and the other half is printed by a different process. A grep over one log is
            # the only way to compare them after a run.
            if self.rank == 0:
                from agent_system.utils.diagnostics import action_head_summary

                self._action_head_rederive_count = getattr(self, "_action_head_rederive_count", 0) + 1
                print(
                    f"[Dyad-HEAD] side=actor n={self._action_head_rederive_count} "
                    f"{action_head_summary(head.weight)}",
                    flush=True,
                )

    def _dyad_refresh_encoder_cache(self) -> bool:
        """Refresh encoder outputs in the projector buffers after an update.

        These buffers are synchronized to the rollout engine. Stale buffers would make
        sampling and training log-probabilities use different action heads.
        """
        encoder = getattr(self, "dyad_llm_backbone", None)
        head = getattr(self, "dyad_residual_head", None)
        prompts = getattr(self, "dyad_encoder_prompts", None)
        if dynamic_actions_enabled():
            return False  # Dynamic catalogues are re-encoded during replay and rollout.
        if encoder is None or head is None or not prompts:
            return False
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
        ):
            cached = encoder.encode_hidden(prompts, use_cache=False)
        head.set_encoder_cache(cached.hidden, cached.mask)
        return True

    def _install_action_head_weight_fn(self, inner, action_config, lm_head_weight) -> None:
        """Give the forward hook a differentiable head, so the projector can actually learn.

        `head.weight.copy_(...)` above severs the graph, and `action_head` is a frozen materialization, so
        the materialised head can never carry a gradient back to the projector. Under
        a schedule that trains the encoder that would mean freezing the policy LLM backbone and then updating
        nothing at all --
        with pg_loss still moving and every metric still looking normal.

        The closure pools cached representations, or computes fresh representations when the
        encoder backbone trains. The vocabulary matrix is a detached scale/device/dtype reference,
        not an additive head component.
        """
        residual_head = getattr(self, "dyad_residual_head", None)
        setting = getattr(self, "dyad_setting", None)
        # `trains_action_head` is the schedule asking for the head to be live: joint and
        # frozen_llm_adaptation need the gradient path, policy_lm_only does not. This used to read a
        # separate `TRAIN_TARGET` env var, which could contradict the schedule -- `joint_optimization` +
        # `TRAIN_TARGET=policy_lm` unfroze the projector and then gave it no gradient path, so the
        # run trained nothing while every metric looked normal. One dimension, one answer.
        if residual_head is None or setting is None or not setting.trains_action_head:
            # policy_lm_only reads the materialized frozen head without a differentiable closure.
            if hasattr(inner, "action_head_weight_fn"):
                del inner.action_head_weight_fn
            return

        # Detach the norm reference so head scaling cannot train the vocabulary matrix.
        inner.action_head_base = lm_head_weight.detach()

        # Encoder gradients follow the selected training scope.
        trains_encoder_lm = getattr(
            getattr(self, "dyad_setting", None), "trains_encoder_lm", False)
        encoder_backbone = getattr(self, "dyad_llm_backbone", None)
        worker = self

        def _weight_fn():
            # Read prompts at call time; capturing before initialization would silently fall back to frozen buffers.
            encoder_prompts = getattr(worker, "dyad_encoder_prompts", None)
            if trains_encoder_lm:
                if encoder_backbone is None or not encoder_prompts:
                    raise RuntimeError("Trainable encoder is missing its backbone or action prompts")
                # Recompute without the cache to retain the head -> projector -> encoder gradient path.
                cached = encoder_backbone.encode_hidden(encoder_prompts, use_cache=False)
                return residual_head(cached.hidden, cached.mask, inner.action_head_base)
            # Projector-only training reads the same frozen buffers synchronized to the rollout engine.
            cached = residual_head.cached_encoder_output()
            return residual_head(cached.hidden, cached.mask, inner.action_head_base)

        inner.action_head_weight_fn = _weight_fn


# The name main_dyad resolves for `actor.strategy=dyad`. Kept as an alias rather than renamed at
# the call site so the strategy string, the config, and the class stay one lookup apart.
AsyncActorRolloutRefWorker = DyadActorRolloutRefWorker
