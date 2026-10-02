"""Compile official tau tools with the existing free-argument Dyad router."""
from __future__ import annotations

import copy
import json

RESPOND = {'name': 'respond', 'description': 'Send a message to the customer.',
           'parameters': {'type': 'object', 'properties': {'content': {'type': 'string'}},
                          'required': ['content'], 'additionalProperties': False}}


def compile_tools(tokenizer, vocab_size, tools, capacity, *, respond_definition=None):
    from agent_system.policies.dyad.actions.codegym_schema import build_codegym_schema
    from agent_system.policies.dyad.actions.codegym_tasks import compile_task
    respond = RESPOND if respond_definition is None else respond_definition
    if respond.get('name') != 'respond':
        raise ValueError('Virtual response action must be named respond')
    definitions = [copy.deepcopy(t['function']) for t in tools] + [copy.deepcopy(respond)]
    names = [d['name'] for d in definitions]
    if len(names) != len(set(names)) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError('Official tool names must be nonempty, unique, and must not shadow respond')
    if not tokenizer.eos_token:
        raise ValueError('Native tau schema requires a tokenizer EOS marker')
    raw = build_codegym_schema({'env_name': 'tau_external', 'actions': names, 'params': {}})
    # JSON braces, quoted strings and newlines remain normal LM tokens.
    # Only the real EOS token ends free text; never force a closing brace or repair JSON.
    raw['markers'].update(enter='<action>', exit=tokenizer.eos_token,
                          argument_value_end='', value_end_on_exit=True,
                          max_value_tokens=0, emit_eos=False)
    for definition in definitions:
        name = definition['name']
        raw['actions'][name].update(
            description=definition.get('description', name),
            surface_form='{"name": ' + json.dumps(name) + ', "arguments":',
            mcp={'name': name, 'description': definition.get('description', ''),
                 'inputSchema': definition['parameters']})
    cfg = compile_task(tokenizer, vocab_size, raw, capacity)
    # The compiler's CodeGym compatibility defaults reinsert a newline terminator
    # and prefix-expanded special-token matches. Neither is safe for arbitrary JSON.
    # A token-ID-exact EOS delimiter leaves nested objects and string contents untouched.
    cfg['markers']['value_end_ids'] = []
    cfg['markers']['exit_value_end_ids'] = [int(tokenizer.eos_token_id)]
    cfg['exit_value_end_ids'] = [int(tokenizer.eos_token_id)]
    cfg['value_end_token_ids'] = []
    return cfg


class MissingExpandedHeadSelection(ValueError):
    """A valid generation ended without choosing an expanded action."""


def verify_trace(trace, cfg):
    from agent_system.policies.dyad.actions.action_router import ActionRouter
    if not trace or trace.get('unified') is not True or trace.get('action_config') != cfg:
        raise ValueError('Native action_content/config missing or mismatched')
    router = ActionRouter(cfg)
    selections = []
    for raw in trace['raw_token_ids']:
        while router.decision().kind == 'force':
            router.advance(router.decision().forced_token)
        decision = router.decision()
        if decision.kind == 'expanded':
            if raw not in decision.allowed_ids:
                raise ValueError('Native decision lies outside the router candidate set')
            selections.append({'selected_id': raw, 'allowed_ids': decision.allowed_ids,
                               'name': next(n for n, i in cfg['action_name_ids'].items() if i == raw),
                               'replay_verified': True})
        elif not 0 <= raw < router.V:
            raise ValueError('Expanded decision appeared outside action-selection phase')
        router.advance(raw)
    if not selections:
        raise MissingExpandedHeadSelection('No actual expanded-head selection; text-only output is not Dyad')
    if len(selections) != 1:
        raise ValueError('Expected exactly one actual expanded-head selection')
    return selections[0]
