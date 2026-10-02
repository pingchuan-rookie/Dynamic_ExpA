"""Installing Dyad's action head and encoder onto a freshly built module.

Extracted from `dyad_workers.ActorRolloutRefWorker._build_model_optimizer` rather than copied, so
the verl 0.9 engine path (`dyad_engine.DyadFSDPEngineWithLMHead`) and the worker path run the same
code. A copy would drift, and the way it drifts is silent: both paths would keep training, and the
difference would show up as one path's projector never receiving gradients.

Every step below has to happen **before the FSDP wrap**, and the ordering constraints are the whole
reason this is a function and not a few lines inline:

  - `action_head` and `dyad_residual_head` are registered as children of the module, so FSDP
    broadcasts them from rank 0 (`sync_module_states`), shards their gradients, and includes them
    in `parameters()` for the optimizer and in the weight stream to the rollout side.
  - the encoder's backbone runs exactly once, here, and its output is stored in the projector's
    **buffers**. FSDP only carries buffers that exist at wrap time; filled afterwards they would be
    missing from the state dict and the rollout side would receive a projector with nothing to
    feed it.
  - `requires_grad` is set to the union over every training_schedule phase, never phase 1's. After
    the wrap it cannot be turned back on (FSDP reshards the original parameters into views of the
    flat parameter), so a side frozen here is frozen for the run.
"""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Install project model components before the official FSDP wrapping stage.
# Extension point: DyadFSDPEngineWithLMHead._build_module after super()._build_module()

from dataclasses import dataclass, field, replace
from typing import Any, Optional

import torch
from torch import nn

from agent_system.policies.dyad.actions.schema_config import load_default_action_config
from agent_system.policies.dyad.actions.task_context import dynamic_actions_enabled
from agent_system.policies.dyad.models.action_head import create_action_head
from agent_system.policies.dyad.models.policy_forward import attach_dyad_forward_hook
from agent_system.utils.hf_config import text_hidden_size
from verl.utils.device import get_device_id, get_device_name


@dataclass
class DyadModelState:
    """Handles the training side needs after the module is built.

    The worker copies these onto itself under the same `dyad_*` names it used before the
    extraction, so the ~1000 lines that read `self.dyad_residual_head` and friends are untouched.
    """

    action_config: dict
    use_orig_params: bool = False

    # Encoder handles are populated during installation.
    encoder_config: Any = None
    setting: Any = None
    llm_encoder: Any = None
    llm_backbone: Any = None
    # Historical attribute name retained for the DirectActionHead state_dict prefix.
    residual_head: Optional[nn.Module] = None
    encoder_prompts: Optional[list] = None
    total_steps: int = 0
    param_counts: dict = field(default_factory=dict)


def install_dyad_action_head(module: nn.Module, tokenizer, torch_dtype) -> DyadModelState:
    """Create the action head, register it on the module, and install the forward hook.

    The hook is what puts the action logits where the training side reads them
    (`output.hidden_states`); see `attach_dyad_forward_hook` for why that field and not a new one.
    """
    vocab_size = module.get_output_embeddings().weight.shape[0]
    action_config = load_default_action_config(tokenizer, vocab_size)
    module.action_head = create_action_head(module, tokenizer, action_config, torch_dtype)
    attach_dyad_forward_hook(module)
    return DyadModelState(action_config=action_config)


def install_dyad_encoder(
    module: nn.Module,
    state: DyadModelState,
    *,
    tokenizer,
    local_path: str,
    total_training_steps: int,
) -> None:
    """Build the LLM encoder and projector and register the projector on the module.

    No-op unless the encoder is enabled. Mutates `state` and `module` in place.
    """
    from agent_system.policies.dyad import models as _em
    from agent_system.policies.dyad.models.encoder_config import EncoderSetting

    state.encoder_config = _em.llm_encoder_config_from_env()
    # The setting is parsed here, not deeper: `EncoderSetting.build` is where the illegal corners
    # are refused, and refusing them at construction means a misconfigured run dies before it loads
    # a model rather than after it has trained nothing for an hour.
    state.setting = EncoderSetting.from_env()
    print(f"[Dyad] encoder setting: {state.setting.describe()}", flush=True)

    # backbone=policy_lm reads the policy LLM backbone, and representation=final_layer_hidden_states needs one real
    # forward pass to fill the encoder cache. At this point the policy LLM backbone is still on CPU (FSDP moves
    # it shortly), and it was loaded with flash-attention, which has no CPU kernel:
    #   NotImplementedError: Could not run 'flash_attn::_flash_attn_varlen_forward'
    #   with arguments from the 'CPU' backend
    # So move it now. FSDP is about to put it on this device anyway; the only cost is holding the
    # unsharded model on one GPU for that single forward, a few GB for a 3-4B model in bf16.
    if state.setting.backbone.is_actor:
        module = module.to(get_device_id())

    # The remote encoder is a sampling replica, never an autograd RPC endpoint.
    trainer_config = state.encoder_config
    if state.setting.trains_encoder_lm:
        # get_device_id returns a worker-local ordinal; PyTorch needs the device type too.
        trainer_config = replace(
            trainer_config, remote=False, gpu_ids=(), device=f"{get_device_name()}:{get_device_id()}"
        )
    state.llm_encoder, state.residual_head = _em.build_llm_encoder(
        trainer_config,
        actor_model_path=local_path,
        actor_hidden=text_hidden_size(module.config),
        # backbone=policy_lm reads the policy LLM backbone instead of loading a second LLM. Passing these is what
        # makes that possible; build_llm_encoder refuses it without them.
        actor_module=module,
        actor_tokenizer=tokenizer,
    )

    if state.residual_head is None:
        return

    # Staged onto the policy LLM backbone's device first. build_llm_encoder placed it on the encoder's device,
    # and handing FSDP a child that is already somewhere else is the kind of thing that works on a
    # one-GPU box and fails under the split.
    state.residual_head = state.residual_head.to(next(module.parameters()).device)

    from agent_system.policies.dyad.models.action_head_factory import encoder_prompts as _prompts_of

    # `description` must be passed. It used to default to "mcp", so leaving it out made
    # DYAD_ENCODER_DESCRIPTION=natural_language produce an mcp run: correct shapes, correct
    # metrics, and a description ablation comparing mcp against itself.
    prompts = _prompts_of(state.action_config, description=state.encoder_config.description)

    # autocast for the one encode pass. The actor is deliberately built in fp32 (so the optimizer
    # states are fp32); FSDP's mixed precision casts it to bf16 during every normal forward, but
    # this one runs outside FSDP, and flash-attention refuses fp32:
    #   RuntimeError: FlashAttention only support fp16 and bf16 data type
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        if dynamic_actions_enabled():
            from agent_system.policies.dyad.models.encoder_cache import CachedHidden
            cached = CachedHidden(torch.zeros(1, 1, state.llm_encoder.hidden_size),
                                  torch.ones(1, 1, dtype=torch.long), "codegym-dynamic")
        else:
            cached = state.llm_encoder.encode_hidden(prompts)
    state.residual_head.set_encoder_cache(cached.hidden, cached.mask)
    state.encoder_prompts = prompts

    # The prompts themselves, reduced to a fingerprint. The two description forms differ **only**
    # in this text: same backbone, same projector, same representation, same training_schedule,
    # same shapes, same metrics. Without something to compare, "I ran the natural-language
    # ablation" and "the env var never reached the worker" produce identical runs and logs.
    print(
        f"[Dyad] encoder prompts: n={len(prompts)} "
        f"description={state.setting.description.value} "
        f"chars={sum(len(p) for p in prompts)} "
        f"fingerprint={cached.fingerprint}",
        flush=True,
    )

    # From here on the backbone is done. Every later read goes through the projector's buffers, so
    # the trainer and the engine build the head from literally the same tensors and neither pays an
    # RPC per step. Keep a handle on the real encoder before swapping in the cache view: LoRA and
    # the per-phase freezing act on the backbone, and CachedHiddenSource has none.
    state.llm_backbone = state.llm_encoder
    state.llm_encoder = _em.CachedHiddenSource(state.residual_head)
    module.dyad_residual_head = state.residual_head
    state.use_orig_params = True

    # The engine owns the replicated encoder optimizer, reductions and checkpoints.
    if state.setting.trains_encoder_lm:
        n = state.llm_backbone.unfreeze_backbone()
        state.llm_backbone.backbone.float()
        # Replicas start identically even if the model loader initializes missing tensors.
        if torch.distributed.is_initialized():
            for tensor in state.llm_backbone.backbone.state_dict().values():
                torch.distributed.broadcast(tensor, src=0)
        if not dynamic_actions_enabled():
            # Promotion to optimizer precision can change normalization arithmetic;
            # seed rollout with the same computation used by current-policy replay.
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=torch.cuda.is_available()):
                current = state.llm_backbone.encode_hidden(prompts, use_cache=False)
            state.residual_head.set_encoder_cache(current.hidden, current.mask)

        # The reduction hook hangs off the optimizer, so it is installed once that exists.
        print(f"[Dyad] encoder LLM fully trainable: {n:,} parameters (cache disabled)", flush=True)

    # The encoder LLM backbone has done its one job. Its output is in the projector's buffers, and
    # `state.llm_encoder` is already a CachedHiddenSource, so nothing can reach the backbone again.
    # Holding it costs the rollout engine its wake-up allocation on every weight sync:
    #     CUDA Error: out of memory at cumem_allocator.cpp:163 ... checkpoint_manager.update_weights
    # Guarded on *owning* the backbone, because backbone=policy_lm shares the policy LLM backbone's and freeing that
    # deletes the policy. Not released while training the encoder LLM backbone: every later step needs it.
    if (
        getattr(state.llm_backbone, "owns_backbone", False)
        and not getattr(state.llm_backbone, "lora_enabled", False)
        and not state.setting.trains_encoder_lm
    ):
        state.llm_backbone.release_backbone()
        print(
            "[Dyad] encoder backbone released after cache fill "
            "(its output lives in the projector buffers from here on)",
            flush=True,
        )

    # The training_schedule decides the freezing. All three schedules are single phase, so this
    # runs once here and the answer holds for the run. The `union` dance that used to sit here --
    # widening `encoder_then_policy_lm` to `joint_optimization` so the FSDP wrap saw every parameter any phase
    # would train -- went away with that schedule on 2026-09-01.
    report = _em.apply_training_schedule(
        state.setting,
        actor_module=module,
        residual_head=state.residual_head,
        encoder=state.llm_backbone,
    )
    print(
        f"[Dyad] training_schedule={report['phase']} trainable={report['counts']}",
        flush=True,
    )
    counts = {
        "policy_lm": report["counts"]["policy_lm"],
        "encoder": report["counts"]["projector"] + report["counts"]["encoder_lm"],
    }
    # Taken here, while the modules are still plain nn.Modules, and from the **union** freezing, so
    # encoder_then_actor's phase 2 reports the real size instead of phase 1's zero.
    state.param_counts = {
        "policy_lm": report["counts"]["policy_lm"],
        "projector": report["counts"]["projector"] + report["counts"]["encoder_lm"],
    }
    print(
        f"[Dyad] encoder enabled: projector={state.encoder_config.projector} "
        f"remote={state.encoder_config.remote} "
        f"training_schedule={state.setting.training_schedule.value} trainable "
        f"policy_lm={counts['policy_lm']} encoder={counts['encoder']}",
        flush=True,
    )
    if state.setting.trains_action_head and counts["encoder"] == 0:
        raise RuntimeError(
            f"training_schedule={state.setting.training_schedule.value} trains the action encoder, but the "
            "projector has 0 trainable parameters. projector=mean has none by construction, so this "
            "run would update nothing on the encoder side -- pg_loss would still move and every "
            "metric would look normal. Use projector=attention, or training_schedule=policy_lm_only."
        )


def freeze_dyad_action_head(module: nn.Module) -> bool:
    """Freeze the materialized action rows; the differentiable head is computed by the forward hook.

    Frozen before the FSDP wrap, with `use_orig_params=True` in return (mirroring vision_tower) so
    FSDP handles the mixed `requires_grad` on the original params. `build_optimizer` does not
    filter it out, but a frozen param's grad stays None, so AdamW skips it.

    Returns whether the caller must set `use_orig_params`.
    """
    head = getattr(module, "action_head", None)
    if head is None:
        return False
    head.requires_grad_(False)
    return True
