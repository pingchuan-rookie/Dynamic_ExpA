"""Named, safe-serialized RPCs inside the real DyadGPUModelRunner process."""
from __future__ import annotations

import hashlib


class InferenceWorkerExtension:
    """vLLM's standard worker extension exposes named RPCs, not pickled code."""
    def dyad_inference_restore_native(self, artifact_directory):
        return restore_native_projector(self, artifact_directory)

    def dyad_inference_restore_alignment(self, projector_path):
        return restore_alignment(self, projector_path)

    def dyad_inference_attest_head(self, context_path):
        return attest_head(self, context_path)


def tensor_digest(items):
    h = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def restore_native_projector(worker, artifact_directory):
    import os
    import torch
    from pathlib import Path
    from agent_system.policies.dyad.inference.checkpoint import verify_native_artifact
    runner = worker.model_runner
    root = Path(artifact_directory)
    manifest = verify_native_artifact(
        root, model_config_path=os.environ.get('DYAD_NATIVE_MODEL_CONFIG') or None)
    state = torch.load(root / 'projector_state.pt', map_location='cpu', weights_only=True)
    head = runner._dyad_residual_head
    head.load_state_dict(state, strict=True)
    from agent_system.utils.hf_config import text_hidden_size
    hidden = text_hidden_size(runner.model_config.hf_config)
    head.set_encoder_cache(torch.zeros(1, 1, hidden), torch.ones(1, 1, dtype=torch.long))
    runner._init_dyad()
    runner.load_dyad_residual_head([('dyad_residual_head.' + k, v) for k, v in head.state_dict().items()])
    runner.reinit_action_head_from_lm_head()
    projector_hash = tensor_digest(list(state.items()))
    if projector_hash != manifest['projector_sha256'] or tensor_digest(list(head.named_parameters())) != projector_hash:
        raise ValueError('Native projector tensor digest mismatch')
    return {'projector_sha256': projector_hash,
            'policy_sha256': manifest['policy_sha256'],
            'runtime_policy_sha256': tensor_digest(list(runner.model.named_parameters())),
            'manifest_sha256': manifest['manifest_sha256']}


def restore_alignment(worker, projector_path):
    import dataclasses
    import torch
    from agent_system.policies.dyad.models.action_head import DirectActionHead
    from agent_system.policies.dyad.models.action_head_factory import load_projector_init
    runner = worker.model_runner
    cfg = dataclasses.replace(runner._dyad_llm_encoder_config, projector_init=projector_path)
    from agent_system.utils.hf_config import text_hidden_size
    hidden = text_hidden_size(runner.model_config.hf_config)
    runner._init_dyad()
    head = DirectActionHead(cfg.projector, hidden, hidden, scale=cfg.scale, projector_kwargs=cfg.projector_kwargs)
    load_projector_init(head, cfg, runner.model_config.model)
    # Bootstrap cache is allocation-only; actual task contexts are encoded separately and
    # consumed by the native context_head path. It is never a target action catalogue.
    head.set_encoder_cache(torch.zeros(1, 1, hidden), torch.ones(1, 1, dtype=torch.long))
    state = [('dyad_residual_head.' + n, t) for n, t in head.state_dict().items()]
    runner.load_dyad_residual_head(state)
    runner.reinit_action_head_from_lm_head()
    return {'projector_sha256': tensor_digest(list(head.named_parameters())),
            'policy_sha256': tensor_digest(list(runner.model.named_parameters()))}


def attest_head(worker, context_path):
    import torch
    from agent_system.policies.dyad.actions.codegym_tasks import load_context, context_head
    from agent_system.policies.dyad.rollout.vllm.dyad_gpu_model_runner import _lm_head_of
    runner = worker.model_runner
    runner._require_dyad_head_ready()
    payload = load_context(context_path)
    with torch.inference_mode():
        weight = context_head(payload, runner._dyad_residual_head,
                              _lm_head_of(runner.model).weight.detach(), runner._dyad_tokenizer)
    return {'head_sha256': tensor_digest([('head', weight)]), 'shape': list(weight.shape),
            'projector_sha256': tensor_digest(list(runner._dyad_residual_head.named_parameters()))}
