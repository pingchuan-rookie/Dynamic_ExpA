"""Lossless, fail-closed CPU restoration of native one-dimensional FSDP DTensors.

Unlike the upstream HF merger, never casts tensors or concatenates replicated buffers.
The input checkpoint is read-only. Each restored tensor and discarded derived cache is
accounted for in the manifest before the serving model can be published.
"""
from __future__ import annotations

import json
import shutil

from agent_system.policies.dyad.checkpoint_compat import read_saved_model_config, normalize_parameter_state
from pathlib import Path


def validate_restore_directory(directory):
    """Keep generated weight artifacts inside the deployed outputs tree."""
    from agent_system.utils.artifact_paths import artifact_root

    path = Path(directory).expanduser()
    if not path.is_absolute():
        raise ValueError('--restore-dir must be an absolute path inside ARTIFACT_ROOT/outputs')
    outputs = (artifact_root() / 'outputs').resolve()
    path = path.resolve()
    if path == outputs or not path.is_relative_to(outputs):
        raise ValueError(f'--restore-dir must be a subdirectory of {outputs}')
    return path


def _file_digest(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def native_checkpoint_files(checkpoint):
    """Validate an entire native shard inventory without loading tensors."""
    import re
    actor = Path(checkpoint) / 'actor'
    files = sorted(actor.glob('model_world_size_*_rank_*.pt'))
    topology = []
    for path in files:
        match = re.fullmatch(r'model_world_size_(\d+)_rank_(\d+)\.pt', path.name)
        if not match or not path.is_file() or not path.stat().st_size:
            raise ValueError('Native checkpoint requires complete nonempty model shards')
        topology.append(tuple(map(int, match.groups())))
    sizes = {size for size, _ in topology}
    if (len(sizes) != 1 or next(iter(sizes)) < 1 or
            sorted(rank for _, rank in topology) != list(range(next(iter(sizes))))):
        raise ValueError('Native checkpoint requires a complete unmixed shard topology')
    size = next(iter(sizes))
    metadata = actor / 'fsdp_config.json'
    if metadata.is_file():
        declared = json.loads(metadata.read_text()).get('world_size')
        if type(declared) is not int or declared != size:
            raise ValueError('Native shard topology disagrees with fsdp_config.json')
    return [actor / f'model_world_size_{size}_rank_{rank}.pt' for rank in range(size)]


def verify_native_artifact(directory, *, source=None, model_config_path=None):
    """Verify the restoration manifest, complete artifacts and original frozen sources.

    This is shared by standalone inference and Ray model construction.
    It reads bytes but never deserializes checkpoint tensors or changes source files.
    """
    root = validate_restore_directory(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    unsigned = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    if _digest(unsigned) != manifest.get('manifest_sha256'):
        raise ValueError('Restore manifest digest mismatch')
    if (manifest.get('version') != 1 or
            manifest.get('restore_method') not in {'cpu_dtensor_exact_2rank_strict_policy_projector',
                                                   'cpu_dtensor_exact_fsdp_strict_policy_projector'} or
            any(manifest.get(k) is not True for k in
                ('dtype_preserved', 'policy_strict_load', 'projector_strict_load'))):
        raise ValueError('Unsupported or incomplete native restoration manifest')
    identity = manifest.get('source_identity', {})
    if identity.get('kind') != 'native_stage2' or _digest(identity) != manifest.get('source_identity_sha256'):
        raise ValueError('Native source identity digest mismatch')
    if source is not None:
        if (source.get('identity') != identity or
                source.get('identity_sha256') != manifest['source_identity_sha256']):
            raise ValueError('Existing restore artifact belongs to another native checkpoint')
        model_config_path = source['model_config']
    source_root = Path(identity['source']).resolve()
    native_files = native_checkpoint_files(source_root)
    expected_shards = {str(path.relative_to(source_root)) for path in native_files}
    if manifest.get('encoder_identity', {}).get('frozen') is False:
        expected_shards.add('actor/encoder_backbone.pt')
    source_files = identity.get('files_sha256', {})
    if set(source_files) != expected_shards:
        raise ValueError('Native source shard inventory differs from restoration identity')
    if manifest.get('world_size', 2) != len(native_files):
        raise ValueError('Native restoration world size differs from source topology')
    for name, expected in source_files.items():
        if _file_digest(source_root / name) != expected:
            raise ValueError('Native checkpoint changed since restoration')
    config_path = Path(model_config_path) if model_config_path else source_root.parent / 'model_config.json'
    if _file_digest(config_path) != identity.get('model_config_sha256'):
        raise ValueError('Native model configuration changed since restoration')
    config = read_saved_model_config(config_path)
    model = config.get('model', {})
    if (config.get('algo') != 'dyad' or config.get('benchmark') != identity.get('training_benchmark') or
            model.get('DYAD_ENCODER_BACKBONE') != 'encoder_lm' or
            model.get('DYAD_ENCODER_TRAINING') not in {'projector_only', 'projector_and_encoder_lm'} or
            model.get('DYAD_ENCODER_ENABLED') != '1'):
        raise ValueError('Native restoration requires the saved independent encoder')
    files = manifest.get('artifact_files_sha256', {})
    actual_files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if (not isinstance(files, dict) or set(files) != actual_files - {'manifest.json'} or
            not {'projector_state.pt', 'policy/config.json',
                 'policy/model.safetensors.index.json'} <= set(files)):
        raise ValueError('Native artifact inventory mismatch')
    for name, expected in files.items():
        path = root / name
        if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root):
            raise ValueError('Native artifact path escapes restoration directory')
        if _file_digest(path) != expected:
            raise ValueError('Restore artifact hash mismatch')
    index = json.loads((root / 'policy/model.safetensors.index.json').read_text())
    for name in index.get('weight_map', {}).values():
        if not isinstance(name, str) or f'policy/{name}' not in files:
            raise ValueError('Native policy index references an unverified shard')
    if not index.get('weight_map'):
        raise ValueError('Native policy index has no tensors')
    encoder = manifest.get('encoder_identity', {})
    encoder_root = Path(encoder.get('source', '')).resolve()
    trained = model.get('DYAD_ENCODER_TRAINING') == 'projector_and_encoder_lm'
    if encoder.get('frozen') is not (not trained) or encoder_root != Path(model['MODEL_PATH']).resolve():
        raise ValueError('Encoder source/training mode differs from native training configuration')
    if _file_digest(encoder_root / 'config.json') != encoder.get('config_sha256'):
        raise ValueError('Frozen encoder configuration changed since restoration')
    weights = encoder.get('weight_files_sha256', {})
    actual_weights = {p.name for pattern in ('model*.safetensors', 'pytorch_model*.bin')
                      for p in encoder_root.glob(pattern)}
    if not weights or set(weights) != actual_weights:
        raise ValueError('Frozen encoder weight inventory changed since restoration')
    for name, expected in weights.items():
        if Path(name).name != name or _file_digest(encoder_root / name) != expected:
            raise ValueError('Frozen encoder base weights changed since native restoration')
    if trained:
        if ('encoder_state.pt' not in files or not manifest.get('encoder_strict_load')
                or files['encoder_state.pt'] != source_files.get('actor/encoder_backbone.pt')):
            raise ValueError('Native artifact does not preserve the trained encoder checkpoint')
    return manifest


def load_native_projector(head, cfg, actor_model_path, directory, model_config_path):
    """Initialize a new Ray head from an audited AgenticRL artifact, not a Alignment payload."""
    import torch
    from agent_system.policies.dyad.inference.worker import tensor_digest
    from agent_system.policies.dyad.models.action_head_factory import llm_encoder_config_from_env

    if cfg.projector_init:
        raise ValueError('Native restoration cannot also initialize a Alignment projector')
    root = validate_restore_directory(directory)
    if Path(actor_model_path).resolve() != root / 'policy':
        raise ValueError('Native restoration requires the verified artifact policy directory')
    manifest = verify_native_artifact(root, model_config_path=model_config_path)
    saved = read_saved_model_config(model_config_path)['model']
    expected = llm_encoder_config_from_env(saved)
    for field in ('enabled', 'projector', 'scale', 'backbone', 'representation', 'description',
                  'max_length', 'projector_kwargs'):
        if getattr(cfg, field) != getattr(expected, field):
            raise ValueError(f'{field} conflicts with native checkpoint model configuration')
    if Path(cfg.resolved_model_path(actor_model_path)).resolve() != Path(manifest['encoder_identity']['source']).resolve():
        raise ValueError('Native restoration must retain the original frozen encoder model')
    state = torch.load(root / 'projector_state.pt', map_location='cpu', weights_only=True)
    if tensor_digest(list(state.items())) != manifest['projector_sha256']:
        raise ValueError('Native projector tensor digest mismatch')
    # Heads are newly constructed and no optimizer exists yet; retain saved tensor dtypes.
    device = next(head.parameters()).device
    state = {name: value.to(device=device) for name, value in state.items()}
    head.load_state_dict(state, strict=True, assign=True)
    if tensor_digest(list(head.state_dict().items())) != manifest['projector_sha256']:
        raise ValueError('Strict native projector loading changed tensor values or dtype')
    return {'source_kind': 'native_stage2', 'source': manifest['source_identity']['source'],
            'manifest_sha256': manifest['manifest_sha256'],
            'projector_sha256': manifest['projector_sha256']}


def restore_tensor(name, values):
    import torch
    from torch.distributed.tensor import DTensor
    from agent_system.policies.dyad.inference.worker import tensor_digest
    size = len(values)
    if not size or any(not isinstance(v, torch.Tensor) for v in values):
        raise ValueError('Native checkpoint requires nonempty tensor values')
    first = values[0]
    if any(v.dtype != first.dtype or tuple(v.shape) != tuple(first.shape) for v in values):
        raise ValueError(f'{name}: rank dtype/global shape mismatch')
    if any(v.layout != torch.strided for v in values):
        raise ValueError(f'{name}: only dense strided checkpoint tensors are supported')
    if isinstance(first, DTensor):
        if not all(isinstance(v, DTensor) for v in values):
            raise ValueError(f'{name}: mixed tensor layouts')
        placement = first.placements
        mesh = first.device_mesh.mesh
        if mesh.tolist() != list(range(size)) or len(placement) != 1:
            raise ValueError(f'{name}: unsupported device mesh/placement {mesh}/{placement}')
        for rank, value in enumerate(values):
            if getattr(value.device_mesh, '_rank', None) != rank:
                raise ValueError(f'{name}: serialized rank does not match checkpoint shard filename')
            if value.device_mesh.mesh_dim_names != ('fsdp',):
                raise ValueError(f'{name}: only the native fsdp mesh dimension is supported')
            if value.device_mesh.mesh.tolist() != list(range(size)) or value.placements != placement:
                raise ValueError(f'{name}: rank mesh/placement disagreement')
        locals_ = [v._local_tensor.detach().cpu() for v in values]
        rule = placement[0]
        if rule.is_shard():
            dim = rule.dim
            if dim < 0 or dim >= first.ndim:
                raise ValueError(f'{name}: invalid shard dimension')
            width = (first.shape[dim] + size - 1) // size
            expected = []
            for rank in range(size):
                shape = list(first.shape)
                shape[dim] = max(0, min(width, first.shape[dim] - rank * width))
                expected.append(tuple(shape))
            if [tuple(v.shape) for v in locals_] != expected:
                raise ValueError(f'{name}: shard sizes do not match exact DTensor chunk placement')
            result = torch.cat(locals_, dim=dim).contiguous()
            layout = {'kind': 'shard', 'dim': dim, 'mesh': list(range(size))}
        elif rule.is_replicate():
            if any(not torch.equal(locals_[0], v) for v in locals_[1:]):
                raise ValueError(f'{name}: replicated DTensor values disagree')
            result = locals_[0].contiguous()
            layout = {'kind': 'replicate', 'mesh': list(range(size))}
        else:
            raise ValueError(f'{name}: partial/unknown placement cannot be proven lossless')
    else:
        if not all(type(v) is torch.Tensor for v in values):
            raise ValueError(f'{name}: unsupported checkpoint value type')
        locals_ = values
        if any(not torch.equal(first, v) for v in values[1:]):
            raise ValueError(f'{name}: ordinary replicated buffers disagree across ranks')
        result = first.detach().cpu().contiguous()
        layout = {'kind': 'plain_replicated'}
    if tuple(result.shape) != tuple(first.shape) or result.dtype != first.dtype:
        raise ValueError(f'{name}: reconstruction changed shape or dtype')
    record = {'shape': list(result.shape), 'dtype': str(result.dtype), 'layout': layout,
              'local_shapes': [list(v.shape) for v in locals_],
              'sha256': tensor_digest([(name, result)])}
    return result, record


def prepare_native(source, directory, *, target_benchmark='external_tau_dynamic'):
    """Validate every tensor, strict-load HF policy/projector, then publish an artifact."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForImageTextToText, AutoModelForCausalLM
    from safetensors.torch import save_file
    from agent_system.policies.dyad.models.action_head import DirectActionHead
    from agent_system.policies.dyad.models.action_head_factory import llm_encoder_config_from_env
    from agent_system.utils.hf_config import text_hidden_size
    from agent_system.policies.dyad.inference.worker import tensor_digest
    import os

    if target_benchmark not in {'external_tau_dynamic', 'request_defined', 't2bench', 'gsm8k', 'alfworld', 'codegym', 'dive', 'webshop', 'swebench_verified'}:
        raise ValueError('Unsupported native restoration target benchmark')
    directory = validate_restore_directory(directory)
    if directory.exists():
        raise ValueError('Restore artifact directory already exists; use a new directory to avoid stale weights')
    config = read_saved_model_config(source['model_config'])
    model = config['model']
    if model.get('DYAD_ENCODER_BACKBONE') != 'encoder_lm' or model.get('DYAD_ENCODER_TRAINING') not in {'projector_only', 'projector_and_encoder_lm'}:
        raise ValueError('Unsupported independent encoder configuration in native checkpoint')
    if model.get('DYAD_ENCODER_ENABLED') != '1':
        raise ValueError('Native checkpoint must enable the actual encoder')
    base = Path(model['MODEL_PATH']).resolve()
    if not base.is_dir():
        raise ValueError('Saved original policy/encoder model path is unavailable')
    actor = Path(source['source']) / 'actor'
    files = native_checkpoint_files(source['source'])
    shards = [torch.load(path, map_location='cpu', weights_only=False, mmap=True) for path in files]
    if any(not isinstance(shard, dict) or set(shard) != set(shards[0]) for shard in shards):
        raise ValueError('Native rank tensor key sets disagree')
    restored, records = {}, {}
    for name in sorted(shards[0]):
        restored[name], records[name] = restore_tensor(name, [s[name] for s in shards])
    del shards
    restored = normalize_parameter_state(restored)
    # Validate policy keys and every tensor against the original architecture, not a
    # guessed prefix mapping. assign=True preserves checkpoint dtype, including fp32.
    hf_config = AutoConfig.from_pretrained(base)
    architecture = (hf_config.architectures or [''])[0]
    cls = AutoModelForImageTextToText if 'ForConditionalGeneration' in architecture else AutoModelForCausalLM
    with torch.device('meta'):
        policy = cls.from_config(hf_config)
    policy_keys = set(policy.state_dict())
    projector_prefix = 'dyad_residual_head.'
    cache_keys = {projector_prefix + 'encoder_hidden', projector_prefix + 'encoder_mask'}
    projector_keys = {k for k in restored if k.startswith(projector_prefix)} - cache_keys
    derived_keys = {'action_head.weight'} | cache_keys
    expected = policy_keys | projector_keys | derived_keys
    if set(restored) != expected or not projector_keys:
        raise ValueError(f'Native keys differ from policy/projector/derived contract: missing={sorted(expected-set(restored))}, unexpected={sorted(set(restored)-expected)}')
    policy_state = {k: restored[k] for k in policy_keys}
    for name, value in policy.state_dict().items():
        if tuple(value.shape) != tuple(policy_state[name].shape):
            raise ValueError(f'{name}: native policy shape differs from saved model architecture')
    policy.load_state_dict(policy_state, strict=True, assign=True)
    if getattr(hf_config, 'tie_word_embeddings', False):
        embedding = policy.get_input_embeddings().weight
        output = policy.get_output_embeddings().weight
        if not torch.equal(embedding, output):
            raise ValueError('Checkpoint tied input/output embedding values disagree')
    previous = dict(os.environ)
    try:
        for key in list(os.environ):
            if key.startswith('DYAD_'):
                del os.environ[key]
        os.environ.update({k: str(v) for k, v in model.items() if k.startswith('DYAD_')})
        cfg = llm_encoder_config_from_env()
    finally:
        os.environ.clear()
        os.environ.update(previous)
    encoder_path = Path(cfg.resolved_model_path(str(base))).resolve()
    if encoder_path != base:
        raise ValueError('Current native restoration requires the saved frozen encoder source to equal the original policy base model; no implicit substitution')
    hidden = text_hidden_size(hf_config)
    projector = DirectActionHead(cfg.projector, hidden, hidden, scale=cfg.scale, projector_kwargs=cfg.projector_kwargs)
    projector_state = {k.removeprefix(projector_prefix): restored[k] for k in projector_keys}
    projector.load_state_dict(projector_state, strict=True, assign=True)
    if tensor_digest(list(projector.state_dict().items())) != tensor_digest(list(projector_state.items())):
        raise ValueError('Strict projector restoration changed tensor values or dtype')
    h, m = restored[projector_prefix + 'encoder_hidden'], restored[projector_prefix + 'encoder_mask']
    old_head = restored['action_head.weight']
    if h.ndim != 3 or m.shape != h.shape[:2] or h.shape[2] != hidden or old_head.ndim != 2 or old_head.shape[1] != hidden:
        raise ValueError('Original derived action head and encoder cache shapes do not match')
    if model.get('DYAD_CODEGYM_ALL') == '1' or model.get('DYAD_DYNAMIC_ACTIONS') == '1':
        capacity = model['DYAD_CODEGYM_ACTION_CAPACITY'] if model.get('DYAD_CODEGYM_ALL') == '1' else model['DYAD_ACTION_CAPACITY']
        if old_head.shape[0] != int(capacity) or tuple(h.shape[:2]) != (1, 1):
            raise ValueError('Native dynamic training head capacity/bootstrap cache differs from saved configuration')
    elif old_head.shape[0] != h.shape[0]:
        raise ValueError('Native static training action head/cache row counts disagree')
    from agent_system.policies.dyad.models.action_head_factory import _model_identity
    file_digest, digest = _file_digest, _digest
    encoder_files = sorted([*base.glob('model*.safetensors'), *base.glob('pytorch_model*.bin')])
    if not encoder_files:
        raise ValueError('Frozen encoder source contains no base model weights')
    encoder_identity = {'source': str(base), 'model_identity': _model_identity(str(base)),
                        'config_sha256': file_digest(base / 'config.json'),
                        'weight_files_sha256': {p.name: file_digest(p) for p in encoder_files},
                        'frozen': True, 'reason': 'saved encoder_lm + projector_only'}
    manifest = {'version': 1, 'source_identity': source['identity'], 'source_identity_sha256': source['identity_sha256'],
                'training_benchmark': config['benchmark'], 'target_benchmark': target_benchmark,
                'restore_method': 'cpu_dtensor_exact_fsdp_strict_policy_projector',
                'world_size': len(files),
                'dtype_preserved': True, 'policy_strict_load': True, 'projector_strict_load': True,
                'runtime_policy_dtype': 'bfloat16 (explicit native rollout precision; lossless artifacts retain original dtypes)',
                'policy_sha256': tensor_digest(list(policy_state.items())),
                'projector_sha256': tensor_digest(list(projector_state.items())),
                'encoder_identity': encoder_identity, 'tensors': records,
                'derived_not_used_for_target': sorted(derived_keys), 'policy_tensor_count': len(policy_state),
                'projector_tensor_count': len(projector_state), 'tensor_count': len(records)}
    trained = model.get('DYAD_ENCODER_TRAINING') == 'projector_and_encoder_lm'
    if trained:
        encoder_file = actor / 'encoder_backbone.pt'
        if source['identity']['files_sha256'].get('actor/encoder_backbone.pt') != file_digest(encoder_file):
            raise ValueError('Trained encoder is absent from the verified checkpoint identity')
        encoder_state = torch.load(encoder_file, map_location='cpu', weights_only=True, mmap=True)
        with torch.device('meta'):
            encoder_model = AutoModel.from_config(AutoConfig.from_pretrained(str(encoder_path)))
        encoder_model.load_state_dict(encoder_state, strict=True, assign=True)
        manifest['encoder_strict_load'] = True
        manifest['encoder_sha256'] = tensor_digest(list(encoder_model.state_dict().items()))
        encoder_identity.update(frozen=False, reason='saved encoder_lm + projector_and_encoder_lm')
    directory.mkdir(parents=True, exist_ok=False)
    try:
        if trained:
            shutil.copyfile(encoder_file, directory / 'encoder_state.pt')
        hf = directory / 'policy'
        hf.mkdir()
        # Copy only model/tokenizer metadata, never mutate the input or the saved benchmark.
        for path in base.iterdir():
            if path.is_file() and path.suffix in ('.json', '.model', '.txt', '.jinja') and not path.name.endswith('.index.json'):
                shutil.copyfile(path, hf / path.name)
        # Bounded safetensor shards; every tensor keeps its original dtype.
        groups, group, size = [], {}, 0
        for name in sorted(policy_state):
            tensor = policy_state[name]
            n = tensor.numel() * tensor.element_size()
            if group and size + n > 1024**3:
                groups.append(group)
                group, size = {}, 0
            group[name] = tensor.clone() if tensor._base is not None else tensor
            size += n
        if group:
            groups.append(group)
        index = {'metadata': {'total_size': sum(v.numel()*v.element_size() for v in policy_state.values())}, 'weight_map': {}}
        for i, tensors in enumerate(groups):
            filename = f'model-{i+1:05d}-of-{len(groups):05d}.safetensors'
            save_file(tensors, str(hf / filename), metadata={'format': 'pt'})
            index['weight_map'].update({k: filename for k in tensors})
        (hf / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
        torch.save(projector_state, directory / 'projector_state.pt')
        manifest['artifact_files_sha256'] = {str(p.relative_to(directory)): file_digest(p)
                                             for p in sorted(directory.rglob('*')) if p.is_file()}
        manifest['manifest_sha256'] = digest(manifest)
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        verify_native_artifact(directory, source=source)
    except BaseException:
        shutil.rmtree(directory)
        raise
    return manifest


def restore_native_encoder(encoder, directory, model_config_path):
    """Restore the independent encoder as well as the policy/projector artifact."""
    import torch
    from agent_system.policies.dyad.inference.worker import tensor_digest
    manifest = verify_native_artifact(directory, model_config_path=model_config_path)
    if manifest['encoder_identity']['frozen']:
        return
    state = torch.load(Path(directory) / 'encoder_state.pt', map_location='cpu', weights_only=True, mmap=True)
    if tensor_digest(list(state.items())) != manifest['encoder_sha256']:
        raise ValueError('Native encoder tensor digest mismatch')
    encoder.backbone.float()
    encoder.backbone.load_state_dict(state, strict=True)
    encoder.clear_cache()
    encoder.native_weight_version = manifest['encoder_sha256']
