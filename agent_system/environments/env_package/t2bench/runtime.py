"""Text-only agent boundary around official t2bench user, tools, orchestration and scoring."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import FunctionType
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import jsonschema
import litellm
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.tasks import RewardType
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.evaluator import evaluator_nl_assertions as nl_evaluator
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner import build_environment, build_user, get_tasks
from tau2.user import user_simulator
from tau2.utils import llm_utils

from .config import COUNTS, thinking_settings

# The standalone evaluator also runs in the isolated benchmark environment.
_project = str(Path(__file__).resolve().parents[4])
if _project not in sys.path:
    sys.path.insert(0, _project)

from agent_system.environments.prompts.t2bench import (
    T2BENCH_INSTRUCTION as REACT_INSTRUCTION,
    build_t2bench_prompt,
    build_t2bench_messages,
)
from agent_system.environments.prompts.protocol import task_prompt_protocol

TEXT_PROTOCOL = 'tau_bench_action_v4'
TASK_PROMPT_PROTOCOL = task_prompt_protocol('t2bench')
RESPOND_SCHEMA = {'type': 'object', 'properties': {'content': {'type': 'string', 'minLength': 1}}, 'required': ['content'], 'additionalProperties': False}


class ParseError(ValueError):
    pass


class ServiceError(RuntimeError):
    def __init__(self, role, category, error_type, status=None):
        self.role, self.category, self.error_type, self.status = role, category, error_type, status
        self.status_code = status
        super().__init__(f'{role}: {category} ({error_type}, HTTP {status})')


def error_category(exc):
    name = (getattr(exc, 'error_type', None) or type(exc).__name__).lower()
    if 'timeout' in name or isinstance(exc, TimeoutError):
        return 'timeout'
    if isinstance(exc, ParseError):
        return 'parse_error'
    if isinstance(exc, ServiceError):
        return exc.category
    if any(word in name for word in ('connection', 'authentication', 'ratelimit', 'apierror', 'badrequest', 'notfound', 'permissiondenied', 'serviceunavailable', 'contextwindow')) or isinstance(exc, httpx.HTTPError):
        return 'service_error'
    if isinstance(exc, (json.JSONDecodeError, jsonschema.ValidationError)):
        return 'parse_error'
    return 'internal_error'


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ParseError('Duplicate JSON key')
            result[key] = value
        return result

    def constant(_):
        raise ParseError('Non-finite JSON number')

    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ParseError('Non-finite JSON number')
        return parsed

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)
    except json.JSONDecodeError:
        raise ParseError('Action must contain one complete JSON object') from None


def parse_action(text, tools):
    from agent_system.parsers.action_envelope import extract_action
    try:
        _, raw = extract_action(text)
    except ValueError as exc:
        raise ParseError(str(exc)) from exc
    data = strict_json(raw)
    if not isinstance(data, dict) or set(data) != {'name', 'arguments'} or not isinstance(data['name'], str) or not isinstance(data['arguments'], dict):
        raise ParseError('Action requires string name and object arguments')
    if data['name'] == 'respond':
        # respond is our text-to-message transport, not an official domain tool.
        try:
            jsonschema.validate(data['arguments'], RESPOND_SCHEMA)
        except jsonschema.ValidationError:
            raise ParseError('Invalid customer response arguments') from None
        if not data['arguments']['content'].strip():
            raise ParseError('Customer response must not be empty')
        return AssistantMessage(role='assistant', content=data['arguments']['content'])
    # Match official native calls: tool lookup and Python argument binding run
    # inside Environment.get_response, which returns ToolMessage(error=True).
    # Prevalidating schemas here would bypass its feedback and max_errors guard.
    return AssistantMessage(role='assistant', content=None, tool_calls=[ToolCall(id='react_' + uuid4().hex, name=data['name'], arguments=data['arguments'])])


def text_message(message):
    """Never send tool roles or tool_calls fields to the agent endpoint."""
    if isinstance(message, ToolMessage):
        return {'role': 'user', 'content': 'API output: ' + (message.content or '')}
    if isinstance(message, AssistantMessage):
        raw = (message.raw_data or {}).get('react_text')
        if raw is None:
            if message.tool_calls:
                raw = '\n'.join('<think></think><action>' + json.dumps({'name': tc.name, 'arguments': tc.arguments}, ensure_ascii=False) + '</action>' for tc in message.tool_calls)
            else:
                raw = '<think></think><action>' + json.dumps({'name': 'respond', 'arguments': {'content': message.content}}, ensure_ascii=False) + '</action>'
        return {'role': 'assistant', 'content': raw}
    return {'role': message.role, 'content': message.content}






# Only this nonsecret descriptor enters upstream deepcopy, logs and session records.
TRAPI_TRANSPORT = '_environment_trapi'


def environment_completion(*, model, **kwargs):
    descriptor = kwargs.pop(TRAPI_TRANSPORT, None)
    if descriptor is None:
        return litellm.completion(model=model, **kwargs)
    from .trapi import normalize_trapi_request, trapi_client

    # The orchestrator can inject seed after llm_settings; normalize at the actual boundary.
    params = normalize_trapi_request(model, kwargs)
    try:
        with trapi_client(descriptor['base_url'], timeout=params.get('timeout', 120), max_retries=0) as client:
            return litellm.completion(model=model, custom_llm_provider='openai',
                                      api_base=descriptor['base_url'], client=client, **params)
    except Exception as exc:
        # Native generate logs exception text. Never expose credential/SDK error bodies there.
        raise ServiceError(descriptor['role'], 'timeout' if error_category(exc) == 'timeout' else 'service_error',
                           getattr(exc, 'error_type', None) or type(exc).__name__, getattr(exc, 'status_code', None)) from None


def _private_environment_generate():
    original = llm_utils.generate
    namespace = dict(original.__globals__, completion=environment_completion)
    private = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    private.__kwdefaults__ = original.__kwdefaults__

    def generate(model, *args, **kwargs):
        # Native logging should describe the effective nonsecret kwargs, including
        # removal of a seed injected by the orchestrator after llm_settings.
        if TRAPI_TRANSPORT in kwargs:
            from .trapi import normalize_trapi_request
            kwargs = normalize_trapi_request(model, kwargs)
        return private(model, *args, **kwargs)

    return generate


@contextmanager
def user_settings(a):
    previous = user_simulator.generate
    if a.user_provider == 'trapi':
        user_simulator.generate = _private_environment_generate()
    try:
        yield
    finally:
        user_simulator.generate = previous


def llm_settings(a, role, seed):
    import os
    template_kwargs = thinking_settings(a, role)
    prefix = '' if role == 'agent' else f'{role}_'
    if getattr(a, f'{role}_provider') == 'trapi':
        if role not in ('user', 'judge'):
            raise ValueError('TRAPI is supported only for environment user/judge roles')
        from .trapi import normalize_trapi_request, resolve_trapi_config
        config = resolve_trapi_config(model=getattr(a, f'{role}_model'), base_url=getattr(a, f'{role}_base_url'))
        kwargs = normalize_trapi_request(config['model'], {
            'temperature': getattr(a, prefix + 'temperature'),
            'max_tokens': getattr(a, prefix + 'max_tokens'), 'seed': seed,
            'timeout': a.timeout, 'num_retries': a.max_retries})
        kwargs[TRAPI_TRANSPORT] = {'base_url': config['base_url'], 'role': role}
        return config['model'], kwargs
    kwargs = {'custom_llm_provider': getattr(a, f'{role}_provider'),
              'api_base': getattr(a, f'{role}_base_url').rstrip('/'),
              'api_key': os.getenv(getattr(a, f'{role}_api_key_env'), 'local-no-key'),
              'temperature': getattr(a, prefix + 'temperature'),
              'max_tokens': getattr(a, prefix + 'max_tokens'),
              'timeout': a.timeout, 'num_retries': a.max_retries}
    if kwargs['custom_llm_provider'] == 'openai':
        kwargs['seed'] = seed
    if template_kwargs:
        kwargs['extra_body'] = {'chat_template_kwargs': template_kwargs}
    model = a.model if role == 'agent' else getattr(a, f'{role}_model')
    return model, kwargs


def needs_judge(task):
    ec = task.evaluation_criteria
    return bool(ec and RewardType.NL_ASSERTION in ec.reward_basis and ec.nl_assertions)


def inventory(a):
    domains = list(COUNTS) if a.domain in ('all', 'both') else [a.domain]
    tasks, info = {}, {}
    for domain in domains:
        full = get_tasks(domain, task_split_name='base')
        ids = [t.id for t in full]
        if len(full) != COUNTS[domain] or len(set(ids)) != len(ids):
            raise ValueError(f'{domain}/base inventory mismatch: got {len(full)}, expected {COUNTS[domain]} unique IDs')
        selected = full[:]
        if a.task_ids is not None:
            unknown = set(a.task_ids) - set(ids)
            if unknown:
                raise ValueError(f'Unknown {domain}/base task IDs: {sorted(unknown)}')
            by_id = {t.id: t for t in full}
            selected = [by_id[i] for i in a.task_ids]
        else:
            if a.shuffle:
                random.Random(a.seed).shuffle(selected)
            if a.limit is not None or a.debug:
                selected = selected[:a.limit if a.limit is not None else 3]
        env = build_environment(domain)
        tools = env.get_tools()
        for tool in tools:
            jsonschema.Draft202012Validator.check_schema(tool.openai_schema['function']['parameters'])
        tasks[domain] = selected
        info[domain] = {'split': 'base', 'full_count': len(full), 'full_task_ids': ids,
                        'selected_count': len(selected), 'selected_task_ids': [t.id for t in selected],
                        'judge_required_task_ids': [t.id for t in selected if needs_judge(t)],
                        'agent_tool_count': len(tools), 'user_tool_count': len(env.get_user_tools()) if env.user_tools is not None else 0,
                        'tasks_sha256': hashlib.sha256(json.dumps([t.model_dump(mode='json') for t in full], sort_keys=True).encode()).hexdigest()}
    return tasks, info


@contextmanager
def judge_settings(a, seed):
    """Official evaluator has module constants, not dependency injection. Restore on exit.

    Only the evaluator's model/transport defaults change, never its prompt or scoring.
    CLI deliberately executes sequentially and does not expose parallel judge contexts.
    """
    settings = llm_settings(a, 'judge', seed)
    previous = (nl_evaluator.DEFAULT_LLM_NL_ASSERTIONS, nl_evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS, nl_evaluator.generate)
    try:
        nl_evaluator.DEFAULT_LLM_NL_ASSERTIONS, nl_evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS = settings
        if a.judge_provider == 'trapi':
            nl_evaluator.generate = _private_environment_generate()
        yield
    finally:
        nl_evaluator.DEFAULT_LLM_NL_ASSERTIONS, nl_evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS, nl_evaluator.generate = previous


def probe(a, role, weights=None):
    model, kwargs = llm_settings(a, role, a.seed)
    provider = getattr(a, f'{role}_provider')
    identity = {'model': model, 'provider': provider,
                'base_url': kwargs[TRAPI_TRANSPORT]['base_url'] if provider == 'trapi' else kwargs['api_base']}
    try:
        if provider == 'openai':
            with httpx.Client(timeout=a.timeout, trust_env=False) as client:
                response = client.get(kwargs['api_base'] + '/models', headers={'Authorization': 'Bearer ' + kwargs['api_key']})
                response.raise_for_status()
                match = next((m for m in response.json()['data'] if m['id'] == model), None)
            if match is None:
                raise ServiceError(role, 'service_error', 'ModelNotListed')
            identity['server_model'] = {k: match[k] for k in ('id', 'root', 'max_model_len') if k in match}
            if role == 'agent' and match.get('max_model_len') and match['max_model_len'] != a.context_length:
                raise ServiceError(role, 'service_error', 'ContextLengthMismatch')
            if weights and weights.get('local_files_verified'):
                if urlsplit(kwargs['api_base']).hostname not in ('localhost', '127.0.0.1', '::1') or not match.get('root') or Path(match['root']).expanduser().resolve() != Path(weights['source']):
                    raise ServiceError(role, 'service_error', 'CannotAttestLocalWeightsAgainstServerRoot')
                identity['local_server_weights_match'] = True
        elif weights and weights.get('local_files_verified'):
            raise ServiceError(role, 'service_error', 'LocalWeightsRequireOpenAIModelListing')
        # Reasoning deployments can spend the completion budget before emitting text.
        # Preserve the configured TRAPI budget rather than causing false empty probes.
        if provider != 'trapi':
            token_key = 'max_completion_tokens' if 'max_completion_tokens' in kwargs else 'max_tokens'
            kwargs[token_key] = min(32, kwargs[token_key])
        response = environment_completion(model=model, messages=[{'role': 'user', 'content': 'Reply with OK.'}], **kwargs)
        if not response.choices[0].message.content:
            raise ServiceError(role, 'service_error', 'EmptyProbeResponse')
        identity['probe_passed'] = True
        return identity
    except ServiceError:
        raise
    except Exception as exc:
        raise ServiceError(role, 'timeout' if error_category(exc) == 'timeout' else 'service_error', type(exc).__name__, getattr(exc, 'status_code', None)) from None






class TextActionBoundary(LLMAgent):
    """Shared public state and action prompt; policy generation is supplied externally."""
    def __init__(self, *args, history_length=2, **kwargs):
        super().__init__(*args, **kwargs)
        if type(history_length) is not int or history_length < 0:
            raise ValueError('t2bench history_length must be a nonnegative integer')
        self.history_length = history_length
        self.calls = []

    @property
    def system_prompt(self):
        return build_t2bench_prompt(self.domain_policy, [t.openai_schema for t in self.tools])

    def decision_messages(self, state):
        task = next((m.content for m in state.messages if isinstance(m, UserMessage) and m.content),
                    'Help the customer according to the policy; ask what they need if no request has been provided.')
        return build_t2bench_messages(
            self.domain_policy, [t.openai_schema for t in self.tools],
            [text_message(m) for m in state.messages], self.history_length,
            task_description=task)


