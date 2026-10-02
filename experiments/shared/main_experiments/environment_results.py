"""Read complete, identity-bearing environment evaluations without guessing missing evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def _value(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            if value in ('True', 'False'):
                return value == 'True'
    return value


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _score(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Environment score must be a finite ratio in [0, 1]')
    return float(value)


def _required(mapping, keys, description):
    if any(key not in mapping or mapping[key] is None for key in keys):
        raise ValueError(f'Missing {description}: {", ".join(keys)}')
    return {key: mapping[key] for key in keys}


def _native_identity(run, config, hydra, method, model):
    # Reuse the runtime verifier: hash all source, policy, projector and frozen
    # encoder files without deserializing checkpoint tensors or loading a model.
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from agent_system.policies.dyad.inference.checkpoint import verify_native_artifact
    from experiments.capability_eval.policy_identity import training_identity

    source = config['native_source']
    if not isinstance(source, dict) or not method.startswith('dyad-'):
        raise ValueError('Native restoration requires an explicit Dyad source')
    if (hydra.get('trainer.resume_mode') != 'disable' or hydra.get('trainer.resume_from_path')
            or config.get('alignment_projector') or config.get('stage1_projector') or config.get('target_actions_rebuilt') is not True):
        raise ValueError('Native restoration conflicts with resume/projector or target action provenance')
    restore = run / 'native_restore'
    policy = restore / 'policy'
    if Path(hydra.get('actor_rollout_ref.model.path', '')).resolve() != policy.resolve():
        raise ValueError('Evaluation policy path differs from native restoration')
    manifest = verify_native_artifact(restore, source=source)
    identity = manifest['source_identity']
    checkpoint = str(Path(identity['source']).resolve())
    if (source.get('source') != checkpoint or config.get('evaluation_checkpoint') != checkpoint
            or config.get('source_benchmark') != identity['training_benchmark']
            or manifest.get('training_benchmark') != identity['training_benchmark']
            or config.get('target_benchmark') != config['benchmark']
            or manifest.get('target_benchmark') != config['benchmark']):
        raise ValueError('Native source checkpoint or source/target benchmark provenance differs')
    saved = _json(Path(source['model_config']))
    if (saved.get('model', {}).get('MODEL_NAME') != model
            or training_identity(saved)['training_method'] != method):
        raise ValueError('Native source model or training method differs from evaluation')
    command = _json(run / 'command.json')
    if not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
        raise ValueError('Missing native evaluation command provenance')
    for key, expected in (('actor_rollout_ref.model.path', str(policy)), ('trainer.resume_mode', 'disable')):
        values = [_value(arg.lstrip('+').split('=', 1)[1]) for arg in command
                  if arg.lstrip('+').startswith(key + '=')]
        if values != [expected]:
            raise ValueError('Native evaluation command differs from saved restoration selection')
    if any(arg.lstrip('+').startswith('trainer.resume_from_path=') and
           _value(arg.split('=', 1)[1]) not in (None, '') for arg in command):
        raise ValueError('Native evaluation command also selects trainer checkpoint resume')
    return checkpoint


def _identity(config, model_config, hydra, run=None):
    model = config.get('model')
    if not isinstance(model, str) or not model:
        raise ValueError('Missing model identity')
    if model_config.get('model', {}).get('MODEL_NAME') != model:
        raise ValueError('Saved model identity differs from evaluation selection')
    method = config.get('algorithm')
    if method == 'dyad':
        advantage = config.get('adv_estimator')
        if advantage not in ('grpo', 'gigpo'):
            raise ValueError('Missing Dyad advantage identity')
        method = 'dyad-' + advantage
    if method not in ('grpo_react', 'gigpo', 'dyad-grpo', 'dyad-gigpo'):
        raise ValueError('Unknown evaluation method')
    checkpoint = hydra.get('trainer.resume_from_path')
    if 'native_source' in config:
        checkpoint = _native_identity(run, config, hydra, method, model)
    elif checkpoint:
        checkpoint = str(Path(checkpoint).expanduser().resolve())
        if hydra.get('trainer.resume_mode') != 'resume_path':
            raise ValueError('Checkpoint evaluation did not explicitly restore a checkpoint')
        name = Path(checkpoint).name
        if not name.startswith('global_step_') or not name[len('global_step_'):].isdigit():
            raise ValueError('Evaluation requires an exact global_step_N checkpoint')
    else:
        if method.startswith('dyad') or hydra.get('trainer.resume_mode') != 'disable':
            raise ValueError('Projector-only or ambiguous weights cannot be labelled Zero-shot')
        method = 'zero-shot'
    model_path = hydra.get('actor_rollout_ref.model.path')
    if not isinstance(model_path, str) or not model_path:
        raise ValueError('Missing actual policy model path')
    return model, checkpoint, method, model_path


def _decoding(hydra):
    keys = ('actor_rollout_ref.rollout.val_kwargs.n',
            'actor_rollout_ref.rollout.val_kwargs.temperature',
            'actor_rollout_ref.rollout.val_kwargs.top_p',
            'data.max_prompt_length', 'data.max_response_length',
            'actor_rollout_ref.rollout.multi_turn.max_assistant_turns',
            'data.apply_chat_template_kwargs.enable_thinking')
    result = _required(hydra, keys, 'saved decoding protocol')
    for key in ('actor_rollout_ref.rollout.val_kwargs.do_sample',
                'actor_rollout_ref.rollout.multi_turn.max_tool_response_length', 'data.seed'):
        if key in hydra:
            result[key] = hydra[key]
    return result


def _dataset(hydra):
    import pyarrow.parquet as pq
    filename = hydra.get('data.val_files')
    if isinstance(filename, list) and len(filename) == 1:
        filename = filename[0]
    if not isinstance(filename, str):
        raise ValueError('Expected one explicit evaluation parquet, not an inferred task list')
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise ValueError('Evaluation source parquet is unavailable; cannot verify the task denominator')
    return path, pq.read_table(path).to_pylist()


def _tau(run, config, hydra):
    tau = config.get('tau', {})
    if tau.get('debug') is not False or tau.get('limit') not in (None, -1) or tau.get('task_ids'):
        raise ValueError('Debug, limited or selected-task tau evaluation is not a full result')
    files = list(run.glob('val_generations/*/summary.json'))
    if len(files) != 1:
        raise ValueError('Expected exactly one official tau summary')
    summary = _json(files[0])
    if summary.get('complete') is not True:
        raise ValueError('Official tau evaluation is incomplete')
    source, data = _dataset(hydra)
    domains = tau.get('domains')
    split = tau.get('split')
    trials = tau.get('num_trials')
    if not domains or len(set(domains)) != len(domains) or type(trials) is not int or trials < 1 or not split:
        raise ValueError('Missing tau domains, split or trial count')
    selected = [row for row in data if row['domain'] in domains and row['split'] == split]
    task_keys = {(row['domain'], str(row['task_id'])) for row in selected}
    if len(task_keys) != len(selected) or not task_keys or set(domains) != {key[0] for key in task_keys}:
        raise ValueError('Tau dataset does not contain an unambiguous complete selected domain set')
    expected = {(domain, task, trial) for domain, task in task_keys for trial in range(trials)}
    episodes = summary.get('episodes', [])
    if summary.get('planned') != len(expected) or config.get('planned_episodes') != len(expected) or len(episodes) != len(expected):
        raise ValueError('Tau planned/actual denominator differs from the source dataset')
    source_by_key = {(row['domain'], str(row['task_id'])): row for row in selected}
    seen, scores, snapshots = set(), [], {}
    for episode in episodes:
        key = (episode.get('domain'), str(episode.get('task_id')), episode.get('trial'))
        if key not in expected or key in seen or episode.get('split') != split or episode.get('benchmark') != 't2bench':
            raise ValueError('Missing, duplicate or foreign tau task/trial')
        seen.add(key)
        if (episode.get('metric_valid') is not True or episode.get('official_scored') is not True
                or episode.get('status') not in ('success', 'task_failure')
                or episode.get('task_completed') is not True):
            raise ValueError('Tau episode lacks complete official scoring or has an infrastructure failure')
        row = source_by_key[key[:2]]
        snapshot = episode.get('snapshot_ref', {})
        expected_snapshot = {name: row[name] for name in ('source_commit', 'source_path', 'source_sha256')}
        expected_snapshot['resources'] = json.loads(row['resources_json'])
        if snapshot != expected_snapshot:
            raise ValueError('Tau episode source snapshot differs from the selected dataset')
        snapshots[key[0]] = snapshot
        scores.append(_score(episode.get('official_reward')))
    # Match official needs_judge(): only NL_ASSERTION reward basis with actual
    # assertions uses the judge. Derive this from selected pinned task payloads.
    judge_tasks = []
    for row in selected:
        if not isinstance(row.get('task_json'), str):
            raise ValueError('Missing tau task criteria for judge protocol verification')
        task = json.loads(row['task_json'])
        if not isinstance(task, dict) or 'evaluation_criteria' not in task:
            raise ValueError('Missing tau task evaluation criteria')
        criteria = task['evaluation_criteria']
        if criteria is not None and not isinstance(criteria, dict):
            raise ValueError('Invalid tau task evaluation criteria')
        if criteria and 'NL_ASSERTION' in criteria.get('reward_basis', []) and criteria.get('nl_assertions'):
            judge_tasks.append([row['domain'], str(row['task_id'])])
    settings = _required(tau, ('seed', 'num_trials', 'max_steps', 'max_errors', 'max_retries',
                              'temperature', 'thinking', 'user_model', 'user_provider', 'user_temperature',
                              'judge_provider', 'judge_temperature'), 'tau evaluation protocol')
    if 'judge_model' not in tau or (judge_tasks and not tau['judge_model']):
        raise ValueError('Missing judge model required by selected tau tasks or its explicit null declaration')
    settings.update(judge_model=tau['judge_model'], judge_required_task_ids=sorted(judge_tasks))
    for key in ('user_base_url', 'judge_base_url', 'user_max_tokens', 'judge_max_tokens',
                'user_thinking', 'judge_thinking', 'max_tokens', 'timeout', 'episode_timeout'):
        if key in tau:
            settings[key] = tau[key]
    protocol = {'metric': 'official_reward', 'aggregation': 'task_weighted_pass^1', 'split': split,
                'domains': sorted(domains), 'num_trials': trials, 'source_sha256': _digest(source),
                'task_ids': sorted([list(key) for key in task_keys]), 'snapshots': snapshots,
                'settings': settings, 'decoding': _decoding(hydra)}
    return scores, protocol, [source, files[0]]


def _swebench_settings(config, hydra):
    swe = config.get('swebench', {})
    task_ids = swe.get('task_ids')
    if (swe.get('debug') is not False or swe.get('limit') not in (None, -1)
            or not isinstance(task_ids, list) or len(task_ids) != 50
            or any(not isinstance(key, str) for key in task_ids) or len(set(task_ids)) != 50
            or type(swe.get('num_trials')) is not int or swe['num_trials'] != 1
            or type(config.get('planned_episodes')) is not int or config['planned_episodes'] != 50):
        raise ValueError('SWE main-table protocol requires the fixed non-debug 50-task one-attempt evaluation')
    settings = _required(swe, ('seed', 'num_trials', 'context_length', 'max_prompt_length', 'max_tokens',
                              'max_steps', 'history_length', 'temperature', 'command_timeout',
                              'grading_timeout', 'episode_timeout', 'environment_cpus', 'memory_limit',
                              'pids_limit'), 'SWE evaluation protocol')
    integer_fields = ('seed', 'context_length', 'max_prompt_length', 'max_tokens', 'max_steps',
                      'history_length', 'command_timeout', 'grading_timeout', 'episode_timeout', 'pids_limit')
    if any(type(settings[key]) is not int or settings[key] < (0 if key in ('seed', 'history_length') else 1)
           for key in integer_fields):
        raise ValueError('SWE evaluation budgets and seed must be valid integers')
    if (not 0 <= settings['seed'] < 2**31
            or settings['max_prompt_length'] + settings['max_tokens'] > settings['context_length']):
        raise ValueError('SWE evaluation seed or context budget is invalid')
    for key in ('temperature', 'environment_cpus'):
        if (type(settings[key]) not in (int, float) or not math.isfinite(settings[key])
                or settings[key] < 0 or (key == 'environment_cpus' and settings[key] == 0)):
            raise ValueError('SWE sampling and CPU budgets must be finite and valid')
    expected = {
        'actor_rollout_ref.rollout.val_kwargs.n': 1,
        'actor_rollout_ref.rollout.val_kwargs.temperature': settings['temperature'],
        'actor_rollout_ref.rollout.val_kwargs.do_sample': settings['temperature'] > 0,
        'actor_rollout_ref.rollout.val_kwargs.top_p': 1.0,
        'actor_rollout_ref.rollout.max_model_len': settings['context_length'],
        'data.max_prompt_length': settings['max_prompt_length'],
        'data.max_response_length': settings['max_tokens'],
        'data.apply_chat_template_kwargs.enable_thinking': False,
        'algorithm.step_rollout.profile': 'swebench_verified_native_v2',
        'algorithm.step_rollout.history_length': settings['history_length'],
        'algorithm.step_rollout.max_steps': settings['max_steps'],
        'data.filter_overlong_prompts': False,
        'data.truncation': 'error',
    }
    for key, value in expected.items():
        if key not in hydra or hydra[key] != value or (isinstance(value, bool) and hydra[key] is not value):
            raise ValueError(f'SWE saved decoding or interaction protocol differs: {key}')
    from agent_system.environments.env_package.swebench.tools import scaffold_identity
    mini = swe.get('mini_swe_agent')
    if mini is not None:
        from dataclasses import asdict
        from agent_system.environments.env_package.swebench.mini_agent import MiniConfig
        settings['mini_swe_agent'] = asdict(MiniConfig(**mini))
    scaffold, schema = scaffold_identity(mini is not None)
    if swe.get('schema_hash') != schema or swe.get('scaffold_version') != scaffold:
        raise ValueError('SWE saved scaffold differs from the pinned tool and prompt protocol')
    settings.update(schema_hash=schema, scaffold_version=scaffold)
    return settings, expected


def _swebench_command(run, hydra, decoding):
    command = _json(run / 'command.json')
    if not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
        raise ValueError('Missing SWE evaluation command provenance')
    expected = {**decoding, **_required(hydra, (
        'actor_rollout_ref.model.path', 'trainer.resume_mode', 'trainer.val_only',
        'data.val_files', 'data.val_max_samples', 'trainer.nnodes',
    ), 'SWE command selection')}
    if expected['trainer.nnodes'] != 1:
        raise ValueError('SWE offline driver evidence requires single-node model execution')
    if hydra.get('trainer.resume_from_path'):
        expected['trainer.resume_from_path'] = hydra['trainer.resume_from_path']
    for key, value in expected.items():
        actual = [_value(arg.lstrip('+').split('=', 1)[1]) for arg in command
                  if arg.lstrip('+').startswith(key + '=')]
        if actual != [value]:
            raise ValueError(f'SWE evaluation command differs from saved protocol: {key}')
    if not hydra.get('trainer.resume_from_path') and any(
            arg.lstrip('+').startswith('trainer.resume_from_path=') for arg in command):
        raise ValueError('SWE evaluation command selects an unrecorded checkpoint')


def _swebench_offline(run, config, hydra):
    offline = config.get('offline_evidence', {})
    if (offline.get('required') is not True or offline.get('driver_network') != 'loopback_only'
            or config['swebench'].get('require_network_isolation') is not True
            or hydra.get('trainer.nnodes') != 1):
        raise ValueError('SWE lacks enforced single-node driver/model offline evidence')
    for key in ('environment_python', 'policy_python'):
        if not isinstance(offline.get(key), str) or not Path(offline[key]).is_absolute():
            raise ValueError('SWE offline evidence lacks explicit interpreter identities')
    command = _json(run / 'command.json')
    if command[0] != offline['policy_python']:
        raise ValueError('SWE policy interpreter differs from offline execution evidence')
    import yaml
    tool = yaml.safe_load((run / 'environment_tool.yaml').read_text())
    tools = tool.get('tools', [])
    from agent_system.compat import resolve_module_path

    if (len(tools) != 1 or resolve_module_path(str(tools[0].get('class_name') or '')) != 'agent_system.environments.backends.swebench.tool.SwebenchLocalEnvTool'
            or tools[0].get('config', {}).get('python_executable') != offline['environment_python']
            or tools[0].get('config', {}).get('session_config') != config['swebench']):
        raise ValueError('SWE environment worker configuration differs from offline evaluation selection')
    return offline


def _swebench_worker_offline(evidence, offline, swe):
    """Validate required producer fields while allowing additional audit metadata."""
    expected = swe.get('expected_ray_runtime')
    keys = ('ray_version', 'python_version')
    if (not isinstance(expected, dict)
            or any(not isinstance(expected.get(key), str) or not expected[key].strip() for key in keys)):
        raise ValueError('SWE lacks explicit expected Ray/Python runtime version identity')
    runtime = evidence.get('ray_runtime') if isinstance(evidence, dict) else None
    if (not isinstance(evidence, dict) or evidence.get('worker_network') != 'loopback_only'
            or evidence.get('python_executable') != offline['environment_python']
            or not isinstance(runtime, dict)
            or any(runtime.get(key) != expected[key] for key in keys)):
        raise ValueError('SWE worker lacks matching offline execution or Ray/Python runtime evidence')
    return {'worker_network': 'loopback_only', 'python_executable': evidence['python_executable'],
            'ray_runtime': {key: runtime[key] for key in keys}}


def _swebench_container(inspection, image, settings, purpose):
    from agent_system.environments.backends.swebench.session import memory_bytes

    host = inspection.get('HostConfig', {})
    labels = inspection.get('Config', {}).get('Labels', {})
    networks = inspection.get('NetworkSettings', {}).get('Networks')
    if (not isinstance(inspection.get('Id'), str) or not inspection['Id']
            or inspection.get('Image') != image['image_id']
            or labels.get('dyad.swebench.owned') != 'true' or labels.get('dyad.swebench.purpose') != purpose
            or host.get('NetworkMode') != 'none' or host.get('Privileged') is not False
            or 'ALL' not in {str(cap).upper() for cap in host.get('CapDrop', [])}
            or not any(str(option) in ('no-new-privileges', 'no-new-privileges:true')
                       for option in host.get('SecurityOpt', []))
            or host.get('NanoCpus') != int(settings['environment_cpus'] * 1_000_000_000)
            or host.get('Memory') != memory_bytes(settings['memory_limit'])
            or host.get('MemorySwap') != memory_bytes(settings['memory_limit'])
            or host.get('PidsLimit') != settings['pids_limit'] or host.get('Binds')
            or inspection.get('Mounts') != [] or not isinstance(networks, dict) or set(networks) - {'none'}):
        raise ValueError('SWE container inspection differs from offline isolation, image or resource protocol')
    return inspection['Id']


def _swebench_inspections(episode, image, settings):
    containers = []
    for prefix, purpose in (('agent', 'agent'), ('grading', 'grade')):
        inspection = episode.get(prefix + '_container_inspect')
        if not isinstance(inspection, dict):
            raise ValueError('SWE result lacks actual task/grader container inspection evidence')
        digest = hashlib.sha256(json.dumps(inspection, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if digest != episode.get(prefix + '_container_inspect_sha256'):
            raise ValueError('SWE container inspection hash differs from the original result')
        containers.append(_swebench_container(inspection, image, settings, purpose))
    if len(set(containers)) != 2:
        raise ValueError('SWE task and official grader must use separate clean containers')
    path = Path(episode['grading_artifact_dir']) / 'grading_container_inspect.json'
    if not path.resolve().is_relative_to(Path(episode['grading_artifact_dir']).resolve()):
        raise ValueError('SWE grader inspection escapes its archive')
    if _json(path) != episode['grading_container_inspect']:
        raise ValueError('SWE archived grader inspection differs from the scored result')
    return containers, path


def _swebench_report(run, episode, image):
    directory = Path(episode.get('grading_artifact_dir') or '').resolve()
    if not directory.is_relative_to(run) or directory == run:
        raise ValueError('SWE grading archive must be an explicit run-owned directory')
    paths = [directory / name for name in ('report.json', 'test_output.txt', 'execution.json', 'prediction.json', 'eval.sh')]
    if any(not path.resolve().is_relative_to(directory) for path in paths):
        raise ValueError('SWE grading evidence escapes its owned archive')
    report_path, log_path = paths[:2]
    report = _json(report_path)
    digest = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    instance_id = episode['instance_id']
    if not isinstance(report, dict) or set(report) != {instance_id}:
        raise ValueError('SWE official report has a missing or foreign instance identity')
    item = report[instance_id]
    if (digest != episode.get('official_report_sha256') or _digest(log_path) != episode.get('test_output_sha256')
            or not isinstance(item, dict) or item.get('resolved') is not episode['resolved']
            or item.get('patch_successfully_applied') is not True or not isinstance(item.get('tests_status'), dict)):
        raise ValueError('SWE official report or test output differs from the scored result')
    execution = _json(directory / 'execution.json')
    if (execution.get('truncated') is not False or type(execution.get('exit_code')) is not int
            or execution['exit_code'] != 0 or episode.get('eval_guard_version') != 'setup-errexit-v1'):
        raise ValueError('SWE official test setup/cleanup execution was incomplete or truncated')
    prediction = _json(directory / 'prediction.json')
    patch = Path(episode['patch_path']).read_text()
    if (prediction.get('instance_id') != instance_id or prediction.get('model_patch') != patch
            or prediction.get('model_name_or_path') != 'dyad-offline' or episode.get('image_id') != image['image_id']):
        raise ValueError('SWE official prediction or grading image differs from the submitted task')
    if not (directory / 'eval.sh').is_file():
        raise ValueError('SWE official evaluation script is missing')
    from agent_system.environments.env_package.swebench.grading import guarded_eval_script
    guarded = directory / 'eval_guarded.sh'
    try:
        expected_script = guarded_eval_script((directory / 'eval.sh').read_text())
    except RuntimeError as error:
        raise ValueError('SWE official evaluation script has invalid test-phase markers') from error
    if (not guarded.resolve().is_relative_to(directory)
            or guarded.read_text() != expected_script):
        raise ValueError('SWE archived execution script differs from the official guarded test protocol')
    paths.append(guarded)
    if patch.strip():
        apply_path = directory / 'patch_apply.json'
        applied = _json(apply_path)
        commands = [['git', 'apply', '--verbose', '/tmp/model.patch'],
                    ['git', 'apply', '--verbose', '--reject', '/tmp/model.patch'],
                    ['patch', '--batch', '--fuzz=5', '-p1', '-i', '/tmp/model.patch']]
        if (not isinstance(applied, list) or not 1 <= len(applied) <= len(commands)
                or [item.get('command') for item in applied] != commands[:len(applied)]
                or any(type(item.get('exit_code')) is not int for item in applied)
                or any(item['exit_code'] == 0 for item in applied[:-1]) or applied[-1]['exit_code'] != 0
                or applied[-1].get('container_id') != episode.get('grading_container_inspect', {}).get('Id')
                or len({item.get('container_id') for item in applied}) != len(applied)
                or any(not isinstance(item.get('container_id'), str) or not item['container_id'] for item in applied)):
            raise ValueError('SWE submitted patch was not successfully applied by the official command sequence')
        if not apply_path.resolve().is_relative_to(directory):
            raise ValueError('SWE patch application evidence escapes its archive')
        paths.append(apply_path)
    return paths


def _swebench_action_audit(episode, settings):
    """Require actual expanded-head use, not just a restored Dyad checkpoint."""
    from agent_system.policies.dyad.actions.native_tools import verify_trace
    from agent_system.policies.dyad.actions.policy_replay import build_policy_trace, validate_policy_trace
    from agent_system.environments.env_package.swebench.tools import SCHEMA_HASH, action_tools

    names = {tool["function"]["name"] for tool in action_tools(settings.get("mini_swe_agent") is not None)}
    audits = episode.get('action_audit')
    if (not isinstance(audits, list) or not 1 <= len(audits) <= settings['max_steps']
            or type(episode.get('step_count')) is not int or episode['step_count'] != len(audits)):
        raise ValueError('SWE Dyad lacks complete action audit coverage')
    expanded = 0
    context = None
    for turn, audit in enumerate(audits, 1):
        if (not isinstance(audit, dict) or type(audit.get('assistant_turn')) is not int
                or audit['assistant_turn'] != turn
                or audit.get('schema_hash') != settings.get('schema_hash', SCHEMA_HASH)
                or audit.get('submitted') is not True
                or audit.get('submitted_raw_text') != audit.get('raw_text')):
            raise ValueError('SWE Dyad action audit turn, schema or submission differs')
        payload, ids = audit.get('action_content'), audit.get('token_ids')
        if (not isinstance(payload, dict) or not isinstance(ids, list)
                or any(type(token) is not int or token < 0 for token in ids)
                or not isinstance(payload.get('raw_token_ids'), list)
                or any(type(token) is not int or token < 0 for token in payload['raw_token_ids'])
                or audit.get('raw_token_ids') != payload['raw_token_ids']):
            raise ValueError('SWE Dyad action audit lacks sampled token identity')
        cfg = payload.get('action_config')
        if (not isinstance(cfg, dict)
                or set(cfg.get('action_name_ids', {})) != names
                or (context is not None and cfg != context)):
            raise ValueError('SWE Dyad action context differs from the shared tool scaffold')
        context = cfg
        try:
            trace = build_policy_trace(payload, ids)
            validate_policy_trace(trace, ids)
            selections = verify_trace(payload, cfg)
        except (KeyError, TypeError, RuntimeError, AssertionError, NotImplementedError) as error:
            raise ValueError('SWE Dyad sampled action trace cannot be replayed') from error
        fields = {'response_dyad': trace['response_dyad'], 'seq_mask': trace['seq_mask'],
                  'tool_mask': trace['tool_mask'], 'dyad_allowed_action_ids': trace['allowed_action_ids'],
                  'dyad_action_size': trace['action_size']}
        if (audit.get('policy_trace') != fields
                or (audit.get('selections') is not None
                    and audit['selections'] != [selection['name'] for selection in selections])
                or sum(trace['tool_mask']) != len(selections)):
            raise ValueError('SWE Dyad action head evidence differs from sampled trace replay')
        expanded += len(selections)
    # Vocabulary-only/invalid decisions are legitimate task failures, not a
    # reason to remove their official score from the fixed denominator.
    return expanded


def _swebench(run, config, hydra):
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    settings, decoding = _swebench_settings(config, hydra)
    if config.get('backend') == 'inference_api':
        from agent_system.evaluation.results import verify_command
        verify_command(run, config['inference_configuration'])
    else:
        _swebench_command(run, hydra, decoding)
    offline = _swebench_offline(run, config, hydra)
    swe = config['swebench']
    source = run / 'public_tasks.json'
    if Path(hydra.get('data.val_files', '')).resolve() != source:
        raise ValueError('SWE evaluation must use its saved public task snapshot')
    assets, snapshot = _swebench_assets(swe['assets'], source)
    if config.get('preflight') != snapshot:
        raise ValueError('SWE saved preflight differs from the evaluated public snapshot')
    worker_offline = _swebench_worker_offline(snapshot.get('offline_evidence'), offline, swe)
    files = list(run.glob('val_generations/*/summary.json'))
    if len(files) != 1:
        raise ValueError('Expected exactly one official SWE summary')
    summary = _json(files[0])
    from experiments.shared.evaluation_protocol import select_task_ids
    selected_ids = select_task_ids('swebench_verified', assets['full_instance_ids'])
    if swe['task_ids'] != selected_ids:
        raise ValueError('SWE configured tasks differ from the fixed 50-task evaluation subset')
    scores = _swebench_episodes(summary, assets['full_instance_ids'], settings['seed'],
                               mini_agent=settings.get('mini_swe_agent'))
    evidence = [source, files[0], run / 'command.json', run / 'environment_tool.yaml', Path(swe['assets']) / 'manifest.json']
    containers_seen = set()
    expanded_selections = 0
    for episode in summary['episodes']:
        if episode.get('source_identity') != snapshot['source_identity']:
            raise ValueError('SWE episode source identity differs from the verified snapshot')
        _swebench_worker_offline(episode.get('offline_evidence'), offline, swe)
        directory = run / 'episodes' / episode['episode_id']
        if (not directory.resolve().is_relative_to(run)
                or not (directory / 'result.json').resolve().is_relative_to(directory.resolve())
                or not (directory / 'model.patch').resolve().is_relative_to(directory.resolve())):
            raise ValueError('SWE episode evidence escapes its owned run')
        # Worker archives scoring before the controller adds generation audits.
        scored_episode = {key: value for key, value in episode.items() if key != 'action_audit'}
        if _json(directory / 'result.json') != scored_episode:
            raise ValueError('SWE summary differs from its original episode artifact')
        if 'native_source' in config:
            expanded_selections += _swebench_action_audit(episode, settings)
        patch = directory / 'model.patch'
        if (episode.get('patch_path') != str(patch) or episode.get('patch_sha256') != _digest(patch)):
            raise ValueError('SWE submission patch identity differs from the saved result')
        evidence.extend([directory / 'result.json', patch])
        evidence.extend(_swebench_mini_evidence(run, episode, settings))
        image = assets['images'][episode['instance_id']]
        evidence.extend(_swebench_report(run, episode, image))
        container_ids, inspect_path = _swebench_inspections(episode, image, settings)
        if containers_seen.intersection(container_ids):
            raise ValueError('SWE attempts reused task or grading containers')
        containers_seen.update(container_ids)
        evidence.append(inspect_path)
    if 'native_source' not in config and config.get('backend') != 'inference_api':
        raise ValueError('SWE result lacks saved complete baseline/initial weight identity; model paths alone are not evidence')
    if 'native_source' in config and expanded_selections == 0:
        raise ValueError('SWE Dyad run has no expanded action head activation evidence')
    # load_environment_result already verified every native policy/projector and
    # frozen encoder source byte via _native_identity, not merely a path label.
    protocol = {'metric': 'resolved_percentage', 'aggregation': 'single_attempt_task_mean',
                'split': 'test', 'num_trials': 1, 'task_ids': selected_ids,
                'source_task_count': len(assets['full_instance_ids']), 'selection_seed': 42,
                'dataset_revision': assets['dataset_revision'], 'harness_commit': assets['harness_commit'],
                'harness_version': assets['harness_version'], 'architecture': assets['architecture'],
                'asset_manifest_sha256': snapshot['source_identity']['manifest_sha256'],
                'settings': settings, 'decoding': decoding,
                'offline': {'driver_network': 'loopback_only', 'worker_network': 'loopback_only',
                            'agent_network': 'none', 'grading_network': 'none',
                            'ray_runtime': worker_offline['ray_runtime']}}
    return scores, protocol, evidence


def _swebench_assets(asset_dir, source):
    """Verify archived inputs without requiring Docker on the reporting host."""
    from experiments.shared.dataset.swebench_verified import verify
    from agent_system.environments.env_package.swebench.assets import verify_assets

    asset_dir, source = Path(asset_dir).resolve(), Path(source).resolve()
    assets = verify_assets(asset_dir, require_images=False, require_full=True)
    from experiments.shared.evaluation_protocol import select_task_ids
    selected_ids = select_task_ids('swebench_verified', assets['full_instance_ids'])
    if not set(selected_ids).issubset(assets['images']):
        raise ValueError('SWE result requires immutable registered image identities for all selected 50 tasks')
    if source.name == 'test.parquet':
        public = verify(source.parent, source_dir=asset_dir, require_full=True)
    elif source.name == 'public_tasks.json':
        from agent_system.environments.env_package.swebench.assets import load_public_tasks
        public = _json(source)
        identity = {'manifest_sha256': _digest(asset_dir / 'manifest.json'),
                    'dataset_revision': assets['dataset_revision'], 'harness_commit': assets['harness_commit'],
                    'architecture': assets['architecture'], 'is_full_verified': True,
                    'images': {key: assets['images'][key] for key in selected_ids}}
        if (public.get('format') != 'swebench_public_v1' or public.get('benchmark') != 'swebench_verified'
                or public.get('images_verified') is not True or public.get('source_identity') != identity
                or public.get('tasks') != load_public_tasks(asset_dir, instance_ids=selected_ids)):
            raise ValueError('SWE public snapshot differs from pinned subset tasks, images or source identity')
    else:
        raise ValueError('SWE result requires the audited public parquet or runtime snapshot')
    return assets, public


def _swebench_mini_evidence(run, episode, settings):
    """Keep delegated-model identity and its cost evidence distinct from outer policy results."""
    config = settings.get('mini_swe_agent')
    if config is None:
        if episode.get('mini_swe_agent') is not None:
            raise ValueError('Unexpected mini-swe-agent in the direct-tool protocol')
        return []
    from agent_system.environments.env_package.swebench.mini_agent import MINI_VERSION
    if episode.get('mini_swe_agent') != config:
        raise ValueError('SWE delegated model configuration differs from the saved protocol')
    runs = episode.get('mini_swe_runs')
    if not isinstance(runs, list) or not runs:
        raise ValueError('SWE delegated protocol requires a recorded mini-swe-agent invocation')
    paths = []
    for index, evidence in enumerate(runs, 1):
        directory = run / 'mini_swe_agent' / episode['episode_id'] / f'call_{index:04d}'
        if (evidence.get('config') != config or evidence.get('mini_swe_agent_version') != MINI_VERSION
                or evidence.get('official_scored') is not False
                or type(evidence.get('model_calls')) is not int
                or not 1 <= evidence['model_calls'] <= config['max_steps']):
            raise ValueError('SWE delegated version, budget or scoring evidence differs')
        for filename, path_key, hash_key in (
            ('model.patch', 'patch_path', 'patch_sha256'),
            ('trajectory.json', 'trajectory_path', 'trajectory_sha256'),
        ):
            path = directory / filename
            if (not path.resolve().is_relative_to(run.resolve()) or evidence.get(path_key) != str(path)
                    or evidence.get(hash_key) != _digest(path)):
                raise ValueError('SWE delegated artifact path or digest differs')
            paths.append(path)
        result = directory / 'result.json'
        if not result.resolve().is_relative_to(run.resolve()) or _json(result) != evidence:
            raise ValueError('SWE delegated summary differs from its saved result')
        paths.append(result)
    return paths


def _swebench_episodes(summary, instance_ids, seed, *, mini_agent=None):
    """Reject incomplete attempts before computing the fixed one-trial denominator."""
    from agent_system.environments.backends.swebench.metrics import summarize_episodes
    from agent_system.environments.env_package.swebench.assets import (
        FULL_INSTANCE_IDS_SHA256, FULL_TASK_COUNT, HARNESS_COMMIT, HARNESS_VERSION, value_digest,
    )
    from agent_system.environments.env_package.swebench.tools import scaffold_identity
    _, schema_hash = scaffold_identity(mini_agent is not None)

    if (not isinstance(instance_ids, list) or len(instance_ids) != FULL_TASK_COUNT
            or any(not isinstance(key, str) for key in instance_ids)
            or value_digest(sorted(instance_ids)) != FULL_INSTANCE_IDS_SHA256):
        raise ValueError('SWE result requires the pinned official 500 instance IDs')
    if type(seed) is not int or seed < 0:
        raise ValueError('SWE result requires an explicit nonnegative evaluation seed')
    from experiments.shared.evaluation_protocol import select_task_ids
    instance_ids = select_task_ids('swebench_verified', instance_ids)
    episodes = summary.get('episodes')
    if (summary.get('benchmark') != 'swebench_verified' or summary.get('complete') is not True
            or summary.get('planned') != len(instance_ids) or not isinstance(episodes, list)
            or len(episodes) != len(instance_ids)):
        raise ValueError('SWE summary is not a complete fixed 50-task official evaluation')
    seen, episode_ids, scores = set(), set(), []
    for episode in episodes:
        instance_id = episode.get('instance_id')
        episode_id = episode.get('episode_id')
        expected_episode = hashlib.sha256(json.dumps({
            'benchmark': 'swebench_verified', 'instance_id': instance_id, 'trial': 0, 'seed': seed,
        }, sort_keys=True).encode()).hexdigest()[:24]
        if (instance_id not in instance_ids or instance_id in seen
                or type(episode.get('trial')) is not int or episode['trial'] != 0
                or type(episode.get('seed')) is not int or episode['seed'] != seed
                or episode_id != expected_episode or episode_id in episode_ids):
            raise ValueError('Missing, duplicate or foreign SWE instance/trial/seed identity')
        seen.add(instance_id)
        episode_ids.add(episode_id)
        if (episode.get('metric_valid') is not True or episode.get('official_scored') is not True
                or episode.get('status') != 'completed'
                or type(episode.get('resolved')) is not bool
                or type(episode.get('official_reward')) not in (int, float)
                or episode['official_reward'] not in (0, 1)
                or episode['official_reward'] != int(episode['resolved'])):
            raise ValueError('SWE episode lacks valid official binary resolved scoring')
        if (episode.get('harness_commit') != HARNESS_COMMIT or episode.get('harness_version') != HARNESS_VERSION
                or episode.get('schema_hash') != schema_hash or episode.get('protocol') != 'native_tools'
                or episode.get('task_id') != instance_id):
            raise ValueError('SWE episode harness, scaffold or task identity differs from the pinned protocol')
        scores.append(float(episode['official_reward']))
    recomputed = summarize_episodes(episodes, episodes)
    if summary != recomputed:
        raise ValueError('SWE summary counters or protocol differ from its official episodes')
    return scores


def _standard(run, benchmark, config, hydra):
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if hydra.get('actor_rollout_ref.rollout.val_kwargs.n') != 1:
        raise ValueError('ALFWorld/WebShop table protocol requires exactly one trial per task')
    files = list(run.glob('val_generations/*.jsonl'))
    if len(files) != 1:
        raise ValueError('Expected one explicitly selected evaluation generation file')
    rows = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
    # Older dumps contain opaque rollout UIDs only. Never treat those as task IDs.
    if any(row.get('task_id') is None or row.get('metric_valid') is not True for row in rows):
        raise ValueError('Generation dump lacks task_id/metric_valid evidence; legacy scores cannot prove complete task coverage')
    source, data = _dataset(hydra)
    from experiments.shared.evaluation_protocol import selection_identity
    if benchmark == 'alfworld':
        from experiments.shared.dataset.alfworld import validate_full_test
        mapping_path = Path(__file__).resolve().parents[3] / 'agent_system/environments/configs/alfworld_mappings_unseen.json'
        game_ids = validate_full_test(source, mapping_path, split='test_unseen')
        mapping = {row['item_id']: row['task_id'] for row in _json(mapping_path)}
        selection = selection_identity('alfworld', list(mapping.values()))
        identity = config.get('alfworld_eval_data', {})
        expected_ids = selection['selected_task_ids']
        if (identity.get('source_sha256') != _digest(source) or identity.get('source_rows') != 134
                or identity.get('split') != 'valid_unseen' or len(data) != 134
                or identity.get('evaluation_selection') != selection
                or identity.get('sample_limit') not in (None, -1, 134)
                or identity.get('selected_game_ids') != game_ids
                or identity.get('selected_task_ids') != [mapping[key] for key in game_ids]):
            raise ValueError('ALFWorld evaluation requires the identity-verified valid_unseen 134-task population')
        dataset_protocol = {'source_sha256': identity['source_sha256'], 'selection': selection}
    else:
        identity = config.get('webshop_data', {})
        selection = identity.get('evaluation_selection', {})
        expected_selection = selection_identity('webshop', range(500))
        if (config.get('parameters', {}).get('WEBSHOP_EVAL_SPLIT', 'test') != 'test'
                or identity.get('validation_split') != 'test'
                or identity.get('split_ranges', {}).get('test') != [0, 500]
                or identity.get('rows', {}).get('test') != 500
                or not identity.get('assets_manifest_sha256')
                or any(selection.get(key) != value for key, value in expected_selection.items())
                or selection.get('selected_source_sha256') != _digest(source)):
            raise ValueError('WebShop evaluation requires the fixed identity-verified 100-task subset of test 500')
        full_filename = selection.get('source')
        if not isinstance(full_filename, str) or not full_filename:
            raise ValueError('WebShop subset identity lacks its full source parquet')
        full_path = Path(full_filename).expanduser()
        if not full_path.is_absolute():
            full_path = source.parent / full_path
        full_source, full_data = _dataset({'data.val_files': str(full_path)})
        if selection.get('source_sha256') != _digest(full_source):
            raise ValueError('WebShop full test source checksum differs from the subset identity')
        source_ids = [row.get('extra_info', {}).get('task_id') for row in full_data]
        if any(type(key) is not int for key in source_ids) or selection_identity('webshop', source_ids) != expected_selection:
            raise ValueError('WebShop source is not the official 500-task test population')
        expected_ids = expected_selection['selected_task_ids']
        source_by_id = dict(zip(source_ids, full_data))
        if data != [source_by_id[key] for key in expected_ids]:
            raise ValueError('WebShop evaluation parquet differs from the fixed selected test tasks')
        dataset_protocol = {'split': 'test', 'source_sha256': _digest(source),
                            'full_source_sha256': selection['source_sha256'], 'selection': expected_selection,
                            'assets_manifest_sha256': identity['assets_manifest_sha256']}
    expected_ids = {str(key) for key in expected_ids}
    seen = [str(row['task_id']) for row in rows]
    if len(expected_ids) != len(data) or len(seen) != len(expected_ids) or set(seen) != expected_ids:
        raise ValueError('Missing or duplicate evaluated task IDs')
    if any(row.get('service_error') or row.get('status') in ('incomplete', 'service_error', 'failed') for row in rows):
        raise ValueError('Environment evaluation contains incomplete or infrastructure-error records')
    scores = [_score(row.get('won') if benchmark == 'alfworld' else row.get('score')) for row in rows]
    if benchmark == 'alfworld' and any(score not in (0, 1) for score in scores):
        raise ValueError('ALFWorld success must be binary')
    protocol = {'metric': 'success_rate' if benchmark == 'alfworld' else 'mean_environment_score',
                'aggregation': 'task_mean', 'task_ids': sorted(expected_ids),
                'dataset': dataset_protocol, 'decoding': _decoding(hydra)}
    evidence = [source, files[0], mapping_path if benchmark == 'alfworld' else full_source]
    return scores, protocol, evidence


def load_environment_result(run_dir):
    run = Path(run_dir).expanduser().resolve()
    status = run / 'run.status'
    if status.read_text().strip() != 'exit_code=0':
        raise ValueError('Environment run did not finish successfully')
    standalone = _json(run / 'resolved_config.json')
    if standalone.get('backend') == 'inference_api':
        from agent_system.evaluation.results import load_result
        return load_result(run, standalone)
    config_file, model_file = run / 'resolved_config.json', run / 'model_config.json'
    config, model_config = _json(config_file), _json(model_file)
    benchmark = config.get('benchmark')
    if benchmark not in ('alfworld', 'webshop', 't2bench', 'swebench_verified'):
        raise ValueError('Unsupported table target benchmark')
    if config.get('debug') is True:
        raise ValueError('Debug/smoke evaluation cannot populate a main table')
    hydra = {key: _value(value) for key, value in config.get('hydra', {}).items()}
    if hydra.get('trainer.val_only') is not True:
        raise ValueError('Result must come from a saved evaluation-only configuration')
    allowed_cap = {'alfworld': 134, 'webshop': 100}.get(benchmark, -1)
    if hydra.get('data.val_max_samples') not in (None, -1, allowed_cap):
        raise ValueError('Limited evaluation cannot populate the fixed-protocol table')
    model, checkpoint, method, model_path = _identity(config, model_config, hydra, run)
    if benchmark == 'swebench_verified':
        if (config.get('target_benchmark') != benchmark or config.get('evaluation_checkpoint') != checkpoint
                or (checkpoint and config.get('source_benchmark') not in ('dive', 'codegym'))):
            raise ValueError('SWE main-table checkpoint source/target identity differs from the transfer protocol')
        if method.startswith('dyad-') and 'native_source' not in config:
            raise ValueError('SWE Dyad result requires complete verified native policy/encoder/projector restoration')
        scores, protocol, evidence = _swebench(run, config, hydra)
    elif benchmark == 't2bench':
        scores, protocol, evidence = _tau(run, config, hydra)
    else:
        scores, protocol, evidence = _standard(run, benchmark, config, hydra)
    if 'native_source' in config:
        evidence.extend([run / 'native_restore/manifest.json', run / 'command.json',
                         Path(config['native_source']['model_config'])])
    if not scores:
        raise ValueError('Empty evaluation cannot produce a score')
    return {'benchmark': benchmark, 'model': model, 'checkpoint': checkpoint, 'method': method,
            'model_path': model_path, 'score_percent': 100 * math.fsum(scores) / len(scores),
            'num_episodes': len(scores), 'protocol': protocol,
            'evidence': [str(path) for path in [status, config_file, model_file, *evidence]]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(load_environment_result(args.run_dir), indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'environment results: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
