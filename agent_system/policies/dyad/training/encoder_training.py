"""Lifecycle of the replicated, differentiable Agentic RL encoder.

Policy/projector are FSDP-managed. The independent encoder is replicated per
training rank so differing task catalogues never issue mismatched FSDP forwards.
Its optimizer group, gradients, placement and checkpoint are managed explicitly.
"""
# DYAD-ADD(module): Project extension relative to official verl GRPO.
# Extend optimizer, gradient reduction, checkpoint, and weight-export hooks for the encoder.
# Extension point: DyadFSDPEngineWithLMHead -> EncoderTrainingMixin -> FSDPEngineWithLMHead
from pathlib import Path
import uuid

import torch
import torch.distributed as dist

ENCODER_CHECKPOINT = 'encoder_backbone.pt'


def trainable_encoder(state):
    if state is None or state.setting is None or not state.setting.trains_encoder_lm:
        return None
    encoder = state.llm_backbone
    if encoder is None or getattr(encoder, 'backbone', None) is None:
        raise RuntimeError('Trainable encoder requires a local differentiable backbone')
    return encoder


def reduce_encoder_gradients(parameters, group=None):
    """Average before clipping; unused parameters participate in identical order."""
    params = list(parameters)
    distributed = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size(group) if distributed else 1
    for param in params:
        used = torch.tensor(int(param.grad is not None), device=param.device)
        if world > 1:
            dist.all_reduce(used, group=group)
        if not used.item():
            continue  # Preserve None: AdamW must not decay globally unused parameters.
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        if world > 1:
            dist.all_reduce(param.grad, group=group)
            param.grad.div_(world)


def replay_context(payload, encoder):
    """Recompute current-policy features, preserving the sampled schema and text."""
    prompts = payload.get('prompts')
    if not prompts or len(prompts) != len(payload['action_config']['action_ids']):
        raise ValueError('Trainable encoder replay requires the original action-description prompts')
    hidden = encoder.encode_task_hidden(prompts)
    return dict(payload, hidden=hidden.hidden, mask=hidden.mask)


class EncoderTrainingMixin:
    def _trainable_encoder(self):
        return trainable_encoder(getattr(self, 'dyad_state', None))

    def _build_optimizer(self, module):
        optimizer = self._build_policy_optimizer(module)
        encoder = self._trainable_encoder()
        if encoder is not None:
            params = [p for p in encoder.backbone.parameters() if p.requires_grad]
            if not params:
                raise ValueError('Trainable encoder has no optimizer parameters')
            existing = {id(p) for g in optimizer.param_groups for p in g['params']}
            if any(id(p) in existing for p in params):
                raise ValueError('Encoder parameters already belong to the policy optimizer')
            optimizer.add_param_group({'params': params, 'dyad_group': 'encoder',
                                       'lr': self.optimizer_config.lr})
        return optimizer

    def to(self, device, model=True, optimizer=True, grad=True):
        super().to(device, model=model, optimizer=optimizer, grad=grad)
        encoder = self._trainable_encoder()
        if encoder is not None and model:
            from verl.utils.device import get_device_id
            target = 'cpu' if str(device) == 'cpu' else get_device_id()
            encoder.backbone.to(target)
            encoder.device = target

    def optimizer_step(self):
        encoder = self._trainable_encoder()
        if encoder is not None:
            if getattr(self, 'scaler', None) is not None:
                raise ValueError('Independent encoder training currently requires bf16/fp32, not fp16 scaling')
            params = list(encoder.backbone.parameters())
            reduce_encoder_gradients(params, self.get_data_parallel_group())
            # Each side is clipped at the configured threshold. A nonfinite encoder
            # gradient must skip the entire joint update, including policy/projector.
            norm = torch.nn.utils.clip_grad_norm_(params, self.optimizer_config.clip_grad)
            if not torch.isfinite(norm):
                self.optimizer.zero_grad(set_to_none=True)
                return float('nan')
        return super().optimizer_step()

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs):
        encoder = self._trainable_encoder()
        if encoder is not None:
            from verl.utils.fs import copy, makedirs
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                path = Path(local_path) / ENCODER_CHECKPOINT
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.tmp')
                torch.save(encoder.backbone.state_dict(), temporary)
                temporary.replace(path)
                if hdfs_path:
                    makedirs(hdfs_path, exist_ok=True)
                    copy(str(path), str(hdfs_path).rstrip('/') + '/' + ENCODER_CHECKPOINT)
            if dist.is_initialized():
                dist.barrier()
        return super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep, **kwargs)

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True, **kwargs):
        encoder = self._trainable_encoder()
        if encoder is not None and local_path:
            path = Path(local_path) / ENCODER_CHECKPOINT
            if not path.is_file():
                raise ValueError(f'Trainable encoder checkpoint is missing: {path}')
            state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            encoder.backbone.load_state_dict(state, strict=True)
            encoder.clear_cache()
        return super().load_checkpoint(local_path, hdfs_path, del_local_after_load, **kwargs)

    def get_per_tensor_param(self, *args, **kwargs):
        encoder = self._trainable_encoder()
        if encoder is not None:
            # Called at the normal weight-sync boundary, before rollout resumes.
            # All ranks have already applied the same reduced gradient. Only rank 0
            # publishes the sampling replica; no encoder tensors enter vLLM's LM loader.
            from agent_system.policies.dyad.actions.task_context import dynamic_actions_enabled
            if dynamic_actions_enabled():
                rank = dist.get_rank() if dist.is_initialized() else 0
                error = None
                if rank == 0:
                    try:
                        import ray
                        from agent_system.policies.dyad.rollout.encoder_worker import get_or_create_encoder_actor
                        cfg = self.dyad_state.encoder_config
                        actor = get_or_create_encoder_actor(cfg, self.model_config.local_path)
                        version = uuid.uuid4().hex
                        ray.get(actor.begin_weight_update.remote(version))
                        for name, tensor in encoder.backbone.state_dict().items():
                            ray.get(actor.stage_weight.remote(version, name, tensor.detach().cpu()))
                        ray.get(actor.commit_weight_update.remote(version))
                    except Exception as exc:
                        error = f'{type(exc).__name__}: {exc}'
                if dist.is_initialized():
                    result = [error]
                    dist.broadcast_object_list(result, src=0)
                    error = result[0]
                if error is not None:
                    raise RuntimeError(f'Encoder sampling replica synchronization failed: {error}')
            if getattr(self, '_is_offload_param', False):
                encoder.backbone.to('cpu')
                encoder.device = 'cpu'
        return super().get_per_tensor_param(*args, **kwargs)
