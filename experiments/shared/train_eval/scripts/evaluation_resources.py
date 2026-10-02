"""CPU scheduling lower bounds for the assembled Agentic RL evaluation command.

This is not a performance budget or a guarantee that every actor will fit.
Only known, simultaneous Ray reservations are counted; default Ray actors have
zero lifetime CPUs, so agent/reward workers and TaskRunnerV1 must not each be
charged a speculative CPU. No Ray, model or training modules are imported.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]


def _resource_config(values: Mapping[str, str]):
    """Read vendor defaults, then apply the final command's last-wins overrides.

    Resolve selected fields with OmegaConf rather than interpreting interpolation
    strings as integers. Loading YAML does not instantiate any configured classes.
    """
    from omegaconf import OmegaConf

    root = PROJECT / 'verl/trainer/config'
    trainer = OmegaConf.load(root / 'ppo_trainer.yaml')
    transfer = OmegaConf.load(root / 'transfer_queue/transfer_queue.yaml')
    defaults = OmegaConf.create({'trainer': trainer.trainer, 'transfer_queue': transfer})
    return OmegaConf.merge(defaults, OmegaConf.from_dotlist([f'{k}={v}' for k, v in values.items()]))


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    try:
        number = int(str(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must be a positive integer, got {value!r}') from exc
    if number < 1:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    return number


def check_evaluation_resources(module: str, values: Mapping[str, str], env: Mapping[str, str]) -> dict:
    """Reject impossible local CPU reservations before any runtime preflight.

    run.py owns a fresh local Ray head: RAY_NUM_CPUS overrides effective_cpus(),
    whereas RAY_PREFLIGHT_CPUS and ray_kwargs.ray_init.num_cpus do not size that
    head. EFFECTIVE_CPUS was obtained from the same detector by prepare.py using
    CPU affinity and cgroup v1/v2 quotas. An explicit logical CPU count is not
    clamped to physical CPUs, matching RaySession's actual behavior.
    """
    source = 'RAY_NUM_CPUS' if env.get('RAY_NUM_CPUS') else 'EFFECTIVE_CPUS'
    if source == 'EFFECTIVE_CPUS' and not env.get(source):
        from run import effective_cpus
        available = effective_cpus()
    else:
        available = _positive_int(env[source], source)

    config = _resource_config(values)
    use_v1 = config.trainer.use_v1
    if not isinstance(use_v1, bool):
        raise ValueError(f'trainer.use_v1 must be a boolean, got {use_v1!r}')
    official_v1 = use_v1
    is_dyad = module == 'agent_system.policies.dyad.training.main_dyad'
    if module not in {'verl.trainer.main_ppo', 'agent_system.policies.dyad.training.main_dyad'}:
        raise ValueError(f'Unknown evaluation runtime for resource preflight: {module}')

    reservations = {}
    if official_v1:
        # main_ppo.TaskRunnerV1.run forces enable=True even when the YAML/CLI
        # says transfer_queue.enable=False. transferqueue's controller is
        # @ray.remote(num_cpus=1); SimpleStorage requests an atomic N x CPU:1 PG.
        reservations['TransferQueue controller'] = 1
        if config.transfer_queue.backend.storage_backend == 'SimpleStorage':
            key = 'transfer_queue.backend.SimpleStorage.num_data_storage_units'
            reservations['SimpleStorage placement group'] = _positive_int(
                config.transfer_queue.backend.SimpleStorage.num_data_storage_units, key)

    tq_cpus = sum(reservations.values()) if official_v1 else 0
    # Both ResourcePoolManagers use max_colocate_count=3. RayResourcePool
    # reserves that many CPUs per GPU bundle, including during val_only.
    # Reference/critic colocate inside this pool and are not counted again.
    gpus = _positive_int(config.trainer.n_gpus_per_node, 'trainer.n_gpus_per_node')
    nodes = _positive_int(config.trainer.nnodes, 'trainer.nnodes')
    reservations['trainer GPU placement groups (3 CPU per GPU)'] = 3 * gpus * nodes
    if env.get('RUN_ENV') == 'dive':
        import math
        pool = _positive_int(env.get('DIVE_ENV_POOL_SIZE', '4'), 'DIVE_ENV_POOL_SIZE')
        workers = _positive_int(values.get('actor_rollout_ref.rollout.agent.num_workers', '1'), 'agent.num_workers')
        cpu = float(env.get('DIVE_ENV_CPUS_PER_WORKER', '0.25'))
        if not math.isfinite(cpu) or cpu <= 0:
            raise ValueError('DIVE_ENV_CPUS_PER_WORKER must be finite and positive')
        reservations['DIVE environment actors'] = pool * workers * cpu
    required = sum(reservations.values())
    report = {'runtime': 'official_v1' if official_v1 else ('dyad' if is_dyad else 'official_v0'),
              'cpu_source': source, 'ray_cpus': available,
              'required_cpu_lower_bound': required, 'reservations': reservations}
    if env.get('RUN_ENV') == 'dive':
        encoder_gpus = int(env.get('DYAD_ENCODER_NUM_GPUS', '0')) if is_dyad else 0
        report['policy_gpus'] = gpus * nodes
        report['encoder_gpus'] = encoder_gpus
        report['required_gpus'] = gpus * nodes + encoder_gpus
        if env.get('CUDA_VISIBLE_DEVICES'):
            visible = [gpu for gpu in env['CUDA_VISIBLE_DEVICES'].split(',') if gpu.strip() and gpu.strip() != '-1']
            if len(visible) < report['required_gpus']:
                raise ValueError('DIVE requires separate policy and encoder GPUs; CUDA_VISIBLE_DEVICES is insufficient')
    if available < required:
        details = ', '.join(f'{name}={count}' for name, count in reservations.items())
        stage = f'TransferQueue init alone requires {tq_cpus} CPU; ' if tq_cpus > available else ''
        raise ValueError(
            f'[resource-preflight] {report["runtime"]}: {source}={available}, '
            f'but final evaluation resources require at least {required} Ray CPU '
            f'({details}). {stage}These reservations cannot be scheduled on the local Ray head. '
            'Provide sufficient CPU resources and set RAY_NUM_CPUS accordingly, or explicitly '
            'revise resource overrides (for SimpleStorage: '
            'transfer_queue.backend.SimpleStorage.num_data_storage_units). '
            'No training budget or trainer selection was changed.')
    return report
