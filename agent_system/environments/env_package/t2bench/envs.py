"""Synchronous official tau2 orchestration with an external agent decision boundary."""
from __future__ import annotations

import copy
import hashlib
import os
import time

os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'

from tau2.agent.llm_agent import LLMAgent
from agent_system.environments.prompts.t2bench import build_t2bench_prompt
from tau2.data_model.message import MultiToolMessage, UserMessage
from tau2.orchestrator.orchestrator import Role, get_now

from agent_system.environments.env_package.t2bench.protocol import (arguments, failure, finalize_incomplete,
                                 raw_action, response, result_identity, verify_snapshot)
from .runtime import (EvaluationType, Orchestrator, RESPOND_SCHEMA, TASK_PROMPT_PROTOCOL, TEXT_PROTOCOL, TextActionBoundary, build_environment,
                      build_user, evaluate_simulation, get_tasks, judge_settings, llm_settings,
                      needs_judge, parse_action, text_message, user_settings)


class ExternalActionAgent(TextActionBoundary):
    """Consume one supplied action without creating an agent model client or thread."""

    pending = None

    def _generate_next_message(self, message, state):
        if self.pending is None:
            raise RuntimeError('No external action at the agent boundary')
        state.messages.extend(message.tool_messages if isinstance(message, MultiToolMessage) else [message])
        result, self.pending = self.pending, None
        result.timestamp = get_now()
        return result


class T2BenchEnv:
    def __init__(self, **config):
        self.config = config
        self.orch = None
        self.result = None
        self.messages = []
        self.official_message_count = 0
        self.tools = []
        self.steps = 0
        self.closed = False
        self.finalized = False
        self.cleaned = False
        self.tool_events = []
        self.task_description = 'Help the customer according to the policy; ask what they need if no request has been provided.'
        self.has_customer_request = False

    def reset(self, domain, task_id, split='base', trial=0, seed=None, episode_id=None,
              session_config=None, **metadata):
        if self.orch is not None or self.result is not None:
            raise RuntimeError('Close the session before resetting')
        if split != 'base':
            raise ValueError('tau2 supports only the pinned base split')
        self.args = arguments('t2bench', {**self.config, **(session_config or {}), 'task_split': split})
        task_id = str(task_id)
        if seed is None:
            seed = (self.args.seed + trial * 1000003 + int(hashlib.sha256((domain + '/' + task_id).encode()).hexdigest()[:8], 16)) % 2**31
        self.result = result_identity('t2bench', domain, split, task_id, trial, seed, episode_id)
        self.phase = 'setup'
        try:
            verify_snapshot('t2bench', metadata.get('snapshot_ref'))
            self.task = next(task for task in get_tasks(domain, task_split_name=split) if task.id == task_id)
            self.env = build_environment(domain)
            self.agent = ExternalActionAgent(tools=self.env.get_tools(), domain_policy=self.env.get_policy(), llm=self.args.model, llm_args={})
            self.tools = copy.deepcopy([tool.openai_schema for tool in self.agent.tools])
            self.tools.append({'type': 'function', 'function': {'name': 'respond', 'description': 'Send a message to the customer.', 'parameters': copy.deepcopy(RESPOND_SCHEMA)}})
            model, kwargs = llm_settings(self.args, 'user', seed)
            user = build_user('user_simulator', self.env, self.task, llm=model, llm_args=kwargs)
            official_response = self.env.get_response
            def observe(call):
                before = (self.env.get_db_hash(), self.env.get_user_db_hash())
                answer = official_response(call)
                after = (self.env.get_db_hash(), self.env.get_user_db_hash())
                if call.requestor == 'assistant':
                    self.tool_events.append({'call': call.model_dump(mode='json'), 'response': answer.model_dump(mode='json'),
                                             'state_before': before, 'state_after': after, 'state_changed': before != after})
                return answer
            self.env.get_response = observe
            self.orch = Orchestrator(domain=domain, agent=self.agent, user=user, environment=self.env, task=self.task,
                                     seed=seed, max_steps=self.args.max_steps, max_errors=self.args.max_errors,
                                     timeout=self.args.episode_timeout, validate_communication=True)
            self.orch._run_start_time = get_now()
            self.orch._run_start_perf = time.perf_counter()
            self.phase = 'user'
            with user_settings(self.args):
                self.orch.initialize()
            self._pump()
        except Exception as exc:
            self._failure(exc)
        return self._response(initial=True)

    def _visible(self):
        if self.orch is None or not hasattr(self.orch, 'agent_state'):
            return self.messages
        state = self.orch.agent_state
        visible = list(state.system_messages) + list(state.messages)
        if self.orch.to_role == Role.AGENT and self.orch.message is not None:
            incoming = self.orch.message
            visible.extend(incoming.tool_messages if isinstance(incoming, MultiToolMessage) else [incoming])
        if not self.has_customer_request:
            customer = next((m for m in visible if isinstance(m, UserMessage) and m.content), None)
            if customer is not None:
                self.task_description = customer.content
                self.has_customer_request = True
        return [text_message(message) for message in visible]

    def _pump(self):
        while not self.orch.done and self.orch.to_role != Role.AGENT:
            self.phase = 'user' if self.orch.to_role == Role.USER else 'environment'
            with user_settings(self.args):
                self.orch.step()
            self.orch._check_termination()
        # Append only the official history's new suffix, so incoming messages
        # occur once and every tool error stays paired with its actual call.
        visible = self._visible()
        self.messages.extend(visible[self.official_message_count:])
        self.official_message_count = len(visible)
        if self.orch.done:
            self._score()

    def _score(self):
        if self.finalized:
            return
        self.phase = 'finalize'
        simulation = self.orch._finalize()
        self.finalized = True
        simulation.policy = self.env.get_policy()
        reason = simulation.termination_reason.value
        completed = reason in ('user_stop', 'agent_stop')
        self.result.update(termination_reason=reason, task_completed=completed)
        if reason in ('user_error', 'infrastructure_error', 'unexpected_error', 'context_window_exceeded'):
            self.result.update(status='service_error', metric_valid=False)
            if reason == 'user_error':
                message = self.orch.message
                has_text = message.has_text_content()
                calls = message.tool_calls or []
                known_tools = {tool.name for tool in (self.orch.user.tools or [])}
                self.result['error'] = {
                    'phase': 'user_protocol', 'type': 'OfficialUserCommunicationError',
                    'category': 'user_protocol_error', 'http_status': None,
                    'violation': 'mixed_text_and_tool_calls' if has_text and calls else 'empty_message',
                    'has_text_content': has_text, 'tool_call_count': len(calls),
                    'tool_names': [call.name if call.name in known_tools else '<unknown>' for call in calls],
                    'message': 'Official user simulator violated the text-or-tool-call protocol; no output repair or validation bypass was applied',
                }
            return
        self.phase = 'judge' if needs_judge(self.task) and completed else 'scoring'
        with judge_settings(self.args, self.result['seed']):
            reward_info = evaluate_simulation(simulation, self.task, EvaluationType.ALL, solo_mode=False, domain=self.result['domain'])
        reward = reward_info.reward
        self.result.update(status=('success' if reward >= 1 - 1e-6 else 'task_failure') if completed else ('timeout' if reason == 'timeout' else 'incomplete'),
                           official_reward=reward, reward=reward, attempt_reward=reward, official_scored=True,
                           metric_valid=True, reward_info=reward_info.model_dump(mode='json'))

    def _cleanup(self):
        if self.orch is not None and not self.finalized and not self.cleaned:
            self.cleaned = True
            self.orch._cleanup()

    def _failure(self, exc):
        failure(self.result, exc, self.phase)
        self._cleanup()

    def _response(self, *, initial=False, previous=None, format_error=False):
        self.result['tool_events'] = copy.deepcopy(self.tool_events)
        delta = self.messages[len(previous):] if previous is not None else []
        self.result['task_prompt_protocol'] = TASK_PROMPT_PROTOCOL
        self.result['text_protocol'] = TEXT_PROTOCOL
        value = response(self.messages, self.tools, self.result, initial=initial, delta=delta,
                         steps=self.steps, official_steps=self.orch.step_count if self.orch is not None else 0,
                         format_error=format_error)
        value.update(protocol='t2bench', task_description=self.task_description,
                     domain_policy=self.env.get_policy() if getattr(self, 'env', None) is not None else '',
                     generation_seed=self.result['seed'], task_prompt_protocol=TASK_PROMPT_PROTOCOL)
        return value

    def _check_budget(self):
        # Reuse the configured finite session limit for external generations too.
        # Never increment official step_count or score a locally exhausted budget.
        if self.result['status'] == 'running' and self.steps >= self.args.max_steps:
            self.result.update(status='max_steps', reward=0.0, attempt_reward=0.0,
                               metric_valid=True, termination_reason='max_steps')
            self._cleanup()

    def step(self, action):
        if self.closed or self.result is None or self.result['status'] != 'running':
            raise RuntimeError('Session is not active')
        previous = copy.deepcopy(self.messages)
        try:
            self.orch._check_termination()
            if self.orch.done:
                self._score()
                return self._response(previous=previous)
        except Exception as exc:
            self._failure(exc)
            return self._response(previous=previous)
        self.steps += 1
        try:
            text = raw_action(action)
        except (ValueError, TypeError, KeyError) as exc:
            self.phase = 'action_consistency'
            self._failure(exc)
            return self._response(previous=previous)
        try:
            parsed = parse_action(text, self.agent.tools)
        except (ValueError, TypeError, KeyError) as exc:
            # The official generation path aborts when it cannot construct a
            # message. Do not invent a user turn or retry outside orchestration.
            self.messages.append({'role': 'assistant', 'content': text})
            self.result['format_error_count'] += 1
            self.result.update(status='parse_error', termination_reason='parse_error',
                               reward=0.0, attempt_reward=0.0, metric_valid=True,
                               official_reward=None, official_scored=False,
                               error={'phase': 'agent_parse', 'type': type(exc).__name__,
                                      'http_status': None})
            self._cleanup()
            return self._response(previous=previous, format_error=True)
        if isinstance(action, dict) and action.get('missing_expanded_head'):
            self.phase = 'action_consistency'
            self._failure(ValueError('Valid action has no verified expanded-head selection'))
            return self._response(previous=previous)
        name = parsed.tool_calls[0].name if parsed.tool_calls else 'respond'
        arguments_ = parsed.tool_calls[0].arguments if parsed.tool_calls else {'content': parsed.content}
        parsed.raw_data = {'react_text': text}
        if isinstance(action, dict) and (('name' in action and action['name'] != name) or
                                        ('arguments' in action and action['arguments'] != arguments_)):
            self.phase = 'action_consistency'
            self._failure(ValueError('Structured action disagrees with raw text'))
            return self._response(previous=previous)
        if isinstance(action, dict) and action.get('selected_action', name) != name:
            self.phase = 'action_consistency'
            self._failure(ValueError('Expanded-head action differs from parsed action'))
            return self._response(previous=previous)
        try:
            self.phase = 'agent_action'
            self.agent.pending = parsed
            self.orch.step()
            self.orch._check_termination()
            self._pump()
            self._check_budget()
        except Exception as exc:
            self._failure(exc)
        return self._response(previous=previous)

    def finalize(self, reason):
        if self.result is None:
            raise RuntimeError('Session was not initialized')
        if self.result['status'] == 'running':
            finalize_incomplete(self.result, reason)
            self._cleanup()
        return self._response()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._cleanup()
        self.orch = None
        self.env = None
        self.agent = None
