"""Explicit reference-agent evaluation using the shared t2bench environment semantics."""
from __future__ import annotations
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import litellm
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, UserMessage
from agent_system.environments.env_package.t2bench import runtime as environment
from agent_system.environments.env_package.t2bench.runtime import (
    EvaluationType, Orchestrator, ParseError, ServiceError, TASK_PROMPT_PROTOCOL, TEXT_PROTOCOL,
    build_environment, build_user, get_tasks, error_category, evaluate_simulation, nl_evaluator,
    inventory, judge_settings, llm_settings, needs_judge, parse_action, text_message, user_settings,
    build_t2bench_messages, build_t2bench_prompt, TextActionBoundary,
)


def probe(a, role, weights=None):
    if role == 'agent' and getattr(a, 'algorithm', 'grpo_react') == 'dyad':
        from experiments.shared.train_eval.reference.t2bench.dyad_client import DyadClient
        return DyadClient(a, a.seed, weights).probe()
    return environment.probe(a, role, weights)


class ReActAgent(TextActionBoundary):
    def _generate_next_message(self, message, state):
        state.messages.extend(message.tool_messages if isinstance(message, MultiToolMessage) else [message])
        messages = self.decision_messages(state)
        request = dict(model=self.llm, messages=messages, **self.llm_args)
        if any(k in request for k in ('tools', 'tool_choice', 'functions', 'function_call')):
            raise ValueError('Function-calling options forbidden for text ReAct agent')
        call = {'messages': copy.deepcopy(messages)}
        self.calls.append(call)
        start = time.monotonic()
        try:
            response = litellm.completion(**request)
        except Exception as exc:
            category = error_category(exc)
            raise ServiceError('agent', 'timeout' if category == 'timeout' else 'service_error', type(exc).__name__, getattr(exc, 'status_code', None)) from None
        finally:
            call['seconds'] = time.monotonic() - start
        choice = response.choices[0]
        text = choice.message.content
        reasoning = getattr(choice.message, 'reasoning_content', None)
        # LiteLLM extracts inline <think> text on OpenAI-compatible HTTP responses.
        # Restore only supplied reasoning; missing reasoning must still fail parsing.
        if isinstance(text, str) and not text.lstrip().startswith('<think>') and isinstance(reasoning, str):
            call.update(transport_content=text, transport_reasoning_content=reasoning)
            text = '<think>' + reasoning + '</think>' + text
        call.update(raw_output=text, finish_reason=choice.finish_reason,
                    usage=response.usage.model_dump() if response.usage else None, returned_model=response.model)
        if choice.message.tool_calls:
            raise ParseError('Agent endpoint returned function calls instead of text ReAct')
        result = parse_action(text, self.tools)
        result.raw_data = {'react_text': text, 'finish_reason': choice.finish_reason}
        result.generation_time_seconds = call['seconds']
        result.usage = ({'prompt_tokens': response.usage.prompt_tokens, 'completion_tokens': response.usage.completion_tokens} if response.usage else None)
        # No fabricated cost for local/unknown models.
        return result


class DyadAgent(ReActAgent):
    """Change only generation transport; keep policy prompt and official orchestration."""
    def __init__(self, *args, dyad_args, dyad_seed, **kwargs):
        super().__init__(*args, **kwargs)
        from experiments.shared.train_eval.reference.t2bench.dyad_client import DyadClient
        self.dyad_client = DyadClient(dyad_args, dyad_seed)
        self.calls = self.dyad_client.calls

    def _generate_next_message(self, message, state):
        state.messages.extend(message.tool_messages if isinstance(message, MultiToolMessage) else [message])
        messages = self.decision_messages(state)
        started = time.monotonic()
        text = self.dyad_client.generate(messages, [t.openai_schema for t in self.tools])
        result = parse_action(text, self.tools)
        name = result.tool_calls[0].name if result.tool_calls else 'respond'
        if name != self.calls[-1]['selection']['name']:
            raise ServiceError('agent', 'service_error', 'DyadSelectionMismatch')
        result.raw_data = {'react_text': text, 'dyad': self.calls[-1]}
        result.generation_time_seconds = time.monotonic() - started
        return result


def solve_episode(a, domain, task, trial):
    seed = (a.seed + trial * 1000003 + int(hashlib.sha256((domain + '/' + task.id).encode()).hexdigest()[:8], 16)) % 2**31
    result = {'domain': domain, 'task_id': task.id, 'trial': trial, 'seed': seed, 'status': 'internal_error',
              'reward': None, 'official_scored': False, 'task_completed': False,
              'agent_protocol': 'text_react', 'text_protocol': TEXT_PROTOCOL,
              'task_prompt_protocol': TASK_PROMPT_PROTOCOL, 'history_length': a.history_length,
              'tool_events': [], 'simulation': None}
    env = agent = orch = None
    phase = 'setup'
    started = time.monotonic()
    try:
        env = build_environment(domain)
        model, kwargs = llm_settings(a, 'agent', seed)
        is_dyad = getattr(a, 'algorithm', 'grpo_react') == 'dyad'
        agent_type = DyadAgent if is_dyad else ReActAgent
        extra = {'dyad_args': a, 'dyad_seed': seed} if is_dyad else {}
        agent = agent_type(tools=env.get_tools(), domain_policy=env.get_policy(), llm=model, llm_args=kwargs,
                           history_length=a.history_length, **extra)
        result['agent_protocol'] = 'dyad' if is_dyad else 'text_react'
        model, kwargs = llm_settings(a, 'user', seed)
        user = build_user('user_simulator', env, task, llm=model, llm_args=kwargs)
        # Observe only live tool execution. Evaluator constructs its own replay environment.
        official_get_response = env.get_response
        def observed_response(call):
            before = (env.get_db_hash(), env.get_user_db_hash())
            response = official_get_response(call)
            after = (env.get_db_hash(), env.get_user_db_hash())
            result['tool_events'].append({'call': call.model_dump(mode='json'), 'response': response.model_dump(mode='json'),
                                          'state_before': before, 'state_after': after, 'state_changed': before != after})
            return response
        env.get_response = observed_response
        orch = Orchestrator(domain=domain, agent=agent, user=user, environment=env, task=task,
                            seed=seed, max_steps=a.max_steps, max_errors=a.max_errors,
                            timeout=a.episode_timeout, validate_communication=True)
        phase = 'simulation'
        with user_settings(a):
            simulation = orch.run()
        simulation.policy = env.get_policy()
        result['simulation'] = simulation.model_dump(mode='json')
        result['termination_reason'] = simulation.termination_reason.value
        result['task_completed'] = result['termination_reason'] in ('user_stop', 'agent_stop')
        phase = 'judge' if needs_judge(task) and result['task_completed'] else 'scoring'
        with judge_settings(a, seed):
            simulation.reward_info = evaluate_simulation(simulation, task, EvaluationType.ALL, solo_mode=False, domain=domain)
        result.update(simulation=simulation.model_dump(mode='json'), reward=simulation.reward_info.reward, official_scored=True)
        if result['task_completed']:
            result['status'] = 'success' if result['reward'] >= 1 - 1e-6 else 'task_failure'
        else:
            reason = result['termination_reason']
            result['status'] = 'timeout' if reason == 'timeout' else 'incomplete'
            if reason in ('user_error', 'infrastructure_error', 'unexpected_error', 'context_window_exceeded'):
                result.update(status='service_error', reward=None)
            if reason == 'user_error':
                message = orch.message
                has_text = message.has_text_content()
                tool_calls = message.tool_calls or []
                result['error'] = {
                    'phase': 'user_protocol', 'type': 'OfficialUserCommunicationError',
                    'category': 'user_protocol_error', 'http_status': None,
                    'violation': 'mixed_text_and_tool_calls' if has_text and tool_calls else 'empty_message',
                    'has_text_content': has_text, 'tool_call_count': len(tool_calls),
                    'tool_names': [call.name for call in tool_calls],
                    'message': 'Official user simulator violated the text-or-tool-call protocol; no output repair or validation bypass was applied',
                }
    except Exception as exc:
        category = error_category(exc)
        result['status'] = category
        # Agent protocol failures are failed attempts, not successful official scoring.
        result['reward'] = 0.0 if category == 'parse_error' and phase == 'simulation' and isinstance(exc, ParseError) else None
        result['error'] = {'phase': phase, 'type': type(exc).__name__, 'http_status': getattr(exc, 'status_code', None)}
        if isinstance(exc, (ParseError, ServiceError)):
            result['error']['message'] = str(exc)
    finally:
        result['seconds'] = time.monotonic() - started
        result['agent_calls'] = agent.calls if agent else []
        result['partial_trajectory'] = [m.model_dump(mode='json') for m in orch.get_trajectory()] if orch else []
        # User state includes official hidden scenario, but never transport credentials.
        result['user_messages'] = [m.model_dump(mode='json') for m in getattr(getattr(orch, 'user_state', None), 'messages', [])]
        result['tool_error_count'] = sum(bool(e['response'].get('error')) for e in result['tool_events'])
        result['state_change_count'] = sum(e['state_changed'] for e in result['tool_events'])
    return result


def summarize(results, task_ids, trials):
    expected = {(task_id, trial) for task_id in task_ids for trial in range(trials)}
    actual = [(r['task_id'], r['trial']) for r in results]
    if len(set(actual)) != len(actual) or not set(actual).issubset(expected):
        raise ValueError('Duplicate or unexpected task/trial records')
    valid = [r for r in results if r['reward'] is not None]
    complete = len(valid) == len(expected)
    successful = sum(r['reward'] >= 1 - 1e-6 for r in valid)
    counts = Counter(r['status'] for r in results)
    return {'planned_tasks': len(task_ids), 'task_ids': task_ids, 'num_trials': trials,
            'planned_episodes': len(expected), 'executed_episodes': len(results),
            'executed_tasks': len({r['task_id'] for r in results}),
            'completed_episodes': sum(r['task_completed'] for r in results),
            'official_scored_episodes': sum(r['official_scored'] for r in results),
            'incomplete_episodes': sum(not r['task_completed'] for r in results) + len(expected - set(actual)),
            'missing_episodes': len(expected - set(actual)), 'status_counts': dict(counts),
            'user_protocol_error_episodes': sum(r.get('termination_reason') == 'user_error' for r in results),
            'error_counts': {name: counts[name] for name in ('timeout', 'parse_error', 'service_error', 'task_failure', 'incomplete', 'internal_error')},
            'metrics_complete': complete, 'metric_denominator': len(expected),
            'average_reward': sum(r['reward'] for r in valid) / len(expected) if complete else None,
            'success_rate': successful / len(expected) if complete else None,
            'pass^1': successful / len(expected) if complete else None,
            'diagnostic_valid_success_rate': successful / len(valid) if valid else None,
            'diagnostic_valid_denominator': len(valid),
            'tool_error_count': sum(r.get('tool_error_count', 0) for r in results),
            'state_change_count': sum(r.get('state_change_count', 0) for r in results),
            'metric_policy': 'Planned task/trial denominator; agent parse failures count as zero; official premature termination uses official zero reward; unscored service/judge/internal errors suppress aggregate metrics. Only mean success/pass^1 is reported.'}

