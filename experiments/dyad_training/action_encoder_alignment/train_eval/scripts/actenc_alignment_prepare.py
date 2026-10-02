#!/usr/bin/env python3
"""Resolve Alignment training settings, prepare output paths and check training inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

import yaml

PROJECT = Path(__file__).resolve().parents[5]
CONFIG = Path(__file__).resolve().parents[1] / 'config/actenc_alignment_training.yaml'


def resolve(model=None, hardware=None, debug=False, environ=None):
    env = dict(os.environ if environ is None else environ)
    config = yaml.safe_load(CONFIG.read_text())
    selector = model or env.get('POLICY_MODEL') or config['default_model']
    key = next((k for k, v in config['models'].items()
                if selector in (k, v['path'])), None)
    if key is None:
        raise ValueError(f'Unknown model {selector!r}; add its path and hardware settings to {CONFIG}')
    selected = config['models'][key]
    hardware = hardware or env.get('SCALE_PROFILE')
    debug = debug or env.get('RUN_IS_DEBUG') == '1' or hardware == 'debug'
    if hardware == 'debug':
        hardware = None
    if not hardware:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                capture_output=True, text=True, check=True)
        name = result.stdout.lower()
        hardware = next((h for h in config['hardware'] if h in name), None)
    if hardware not in selected['hardware']:
        raise ValueError(f'No settings for {key} / {hardware}; add an explicit model/hardware combination')
    values = {**config['defaults'], **config['hardware'][hardware],
              **selected.get('defaults', {}), **selected['hardware'][hardware]}
    if debug:
        values.update(config['debug'])
    fixed_batch = str(config['defaults']['BATCH_SIZE'])
    if str(values['BATCH_SIZE']) != fixed_batch or str(env.get('BATCH_SIZE', fixed_batch)) != fixed_batch:
        raise ValueError(f'BATCH_SIZE must remain {fixed_batch} across Alignment models and hardware; '
                         'reduce MICRO_BATCH_SIZE to save GPU memory')
    values = {k: str(env.get(k, v)) for k, v in values.items()}
    for name in ('MICRO_BATCH_SIZE', 'ENCODER_BATCH_SIZE', 'EVAL_BATCH_SIZE'):
        if int(values[name]) < 1:
            raise ValueError(f'{name} must be positive')
    if 0 < int(values['LIMIT']) < int(fixed_batch):
        raise ValueError('LIMIT must be zero or at least BATCH_SIZE for a complete optimizer update')
    values.update(POLICY_MODEL=selected['path'], SCALE_PROFILE=hardware, RUN_IS_DEBUG=str(int(debug)))
    # A CLI model selection takes precedence over inherited POLICY_MODEL.
    return key, values


def storage(values, model, environ):
    from agent_system.policies.dyad.training.action_encoder_alignment import actenc_alignment_config as config
    from agent_system.utils.artifact_paths import artifact_root, run_site
    root = artifact_root(PROJECT, environ)
    site = run_site(environ)
    name = environ.get('RUN_NAME') or f"{model}_{datetime.now():%Y%m%d_%H%M%S_%f}" + ('_debug' if values['RUN_IS_DEBUG'] == '1' else '')
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('RUN_NAME must be a single directory name')
    out_root = root / 'outputs' / site / 'alignment'
    ckpt_root = root / 'ckpt' / site / 'alignment'
    out = Path(environ.get('OUT', str(out_root / name))).expanduser().absolute()
    checkpoint = Path(environ.get('CHECKPOINT_DIR', str(ckpt_root / name))).expanduser().absolute()
    for raw in (out, checkpoint):
        if any(path.is_symlink() for path in (raw, *raw.parents)):
            raise ValueError(f'Output path must not use symlinks: {raw}')
    out, checkpoint = out.resolve(), checkpoint.resolve()
    for path, parent in [(out, root / 'outputs'), (checkpoint, root / 'ckpt')]:
        if not path.is_relative_to(parent.resolve()) or path == parent.resolve():
            raise ValueError(f'{path} must be inside {parent}')
    if out.name != checkpoint.name:
        raise ValueError('OUT and CHECKPOINT_DIR must use the same run directory name')
    return dict(PROJECT=str(PROJECT), ARTIFACT_ROOT=str(root), RUN_SITE=site,
                OUT=str(out), CHECKPOINT_DIR=str(checkpoint),
                RUN_NAME=out.name, DATASET=environ.get('DATASET', str(config.dataset_path())))


def preflight(values):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    from agent_system.policies.dyad.data.actenc_alignment_dataset import load_split

    for split in ('train', 'val'):
        rows = load_split(values['DATASET'], split=split)
        if not rows:
            raise ValueError(f'Empty Alignment dataset split: {split}')
        if split == 'train' and len(rows) < int(values['BATCH_SIZE']):
            raise ValueError('Training split is smaller than BATCH_SIZE; no complete update is possible')
        print(f'[prepare] {split}: {len(rows)} rows')
    for model in set((values['POLICY_MODEL'], values['ENCODER_MODEL'] or values['POLICY_MODEL'])):
        local = os.environ.get('HF_HUB_OFFLINE') == '1'
        path = Path(model)
        if not path.is_dir():
            path = Path(snapshot_download(model, local_files_only=local,
                        allow_patterns=['*.json', '*.safetensors', '*.bin', '*.model', '*.txt', '*.jinja']))
        if not any(path.glob('*.safetensors')) and not any(path.glob('pytorch_model*.bin')):
            raise ValueError(f'No model weights in {path}')
        AutoConfig.from_pretrained(path, local_files_only=True)
        AutoTokenizer.from_pretrained(path, local_files_only=True)
    visible = torch.cuda.device_count()
    replicas = int(os.environ.get('REPLICAS', visible // 2))
    if replicas < 1 or replicas * 2 > visible:
        raise ValueError(f'{replicas} replicas require two GPUs each; visible GPUs: {visible}')
    batch = int(values['BATCH_SIZE'])
    if batch < 1 or batch % replicas:
        raise ValueError(f'Global batch {batch} must be positive and divisible by {replicas} replicas')
    for key in ('EPOCHS', 'EVAL_BATCH_SIZE', 'EVAL_EVERY', 'MAX_LENGTH', 'ENCODER_MAX_LENGTH'):
        if int(values[key]) <= 0:
            raise ValueError(f'{key} must be positive')
    if int(values['LIMIT']) < 0:
        raise ValueError('LIMIT must be nonnegative')
    per_replica = batch // replicas
    micro = min(int(values['MICRO_BATCH_SIZE']), per_replica)
    values.update(REPLICAS=str(replicas), PER_REPLICA_BATCH=str(per_replica),
                  MICRO_BATCH_SIZE=str(micro),
                  GRAD_ACCUMULATION_STEPS=str((per_replica + micro - 1) // micro))
    print(f'[prepare] samples/update={batch}; replicas={replicas}; '
          f'samples/replica={per_replica}; micro-batch/replica={micro}; '
          f'accumulation={values["GRAD_ACCUMULATION_STEPS"]}; incomplete epoch tail dropped')


def check_wandb(values, environ):
    required = environ.get('ALIGNMENT_REQUIRE_UPLOAD') == '1'
    upload = required and environ.get('SKIP_UPLOAD') != '1' and values['RUN_IS_DEBUG'] != '1'
    values['ALIGNMENT_POST_UPLOAD'] = str(int(upload))
    if environ.get('SKIP_UPLOAD') == '1':
        values['ALIGNMENT_WANDB'] = 'off'
    if not upload:
        return
    values.update(WANDB_MODE='online', ALIGNMENT_WANDB='on')
    os.environ['WANDB_MODE'] = 'online'
    import wandb
    from agent_system.policies.dyad.training.action_encoder_alignment.actenc_alignment_wandb_run import resolve_api_key
    credential = resolve_api_key()
    if credential is None:
        raise ValueError('Online Alignment recording requires a WandB credential')
    if not wandb.login(key=credential[0], relogin=True, timeout=30, verify=True):
        raise ValueError('WandB login failed')
    if not wandb.Api().default_entity:
        raise ValueError('WandB account has no default entity')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='model key or Hugging Face id from config/actenc_alignment_training.yaml')
    parser.add_argument('--hardware', help='hardware profile, e.g. a6000, a100 (80GB), h100, h200')
    parser.add_argument('--debug', action='store_true', help='small batch and dataset; preserve the selected model')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true', help='print merged settings without checking assets or starting training')
    modes.add_argument('--check', action='store_true', help='check model, data and GPUs without training')
    args = parser.parse_args()
    try:
        model, values = resolve(args.model, args.hardware, args.debug)
        values.update(storage(values, model, os.environ))
        debug = values['RUN_IS_DEBUG'] == '1'
        values['WANDB_MODE'] = 'offline' if debug else os.environ.get('WANDB_MODE', 'online')
        values['ALIGNMENT_WANDB'] = os.environ.get('ALIGNMENT_WANDB', 'on' if debug else 'auto')
        values.update({k: str(Path(values['OUT']) / sub) for k, sub in
                       [('WANDB_DIR', 'wandb'), ('WANDB_CACHE_DIR', 'wandb/cache'),
                        ('WANDB_CONFIG_DIR', 'wandb/config'), ('WANDB_DATA_DIR', 'wandb/data')]})
        if args.dry_run:
            print(json.dumps(values, indent=2))
            return 0
        preflight(values)
        check_wandb(values, os.environ)
        for path in (Path(values['OUT']), Path(values['CHECKPOINT_DIR'])):
            if path.exists() and any(path.iterdir()) and os.environ.get('FORCE') != '1':
                raise ValueError(f'Run already exists: {path}; choose a new RUN_NAME')
        if args.check:
            print('[prepare] PASS')
            return 0
        import actenc_alignment_run as run
        return run.training(values)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'[prepare] ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
