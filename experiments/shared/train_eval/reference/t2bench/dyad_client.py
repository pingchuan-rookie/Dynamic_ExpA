"""CPU-only transport and provenance contract for the explicit t2bench reference evaluator.

No model/torch imports, no argument repair, no fallback to a text-only endpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

PROTOCOL = 'dyad_tau_json_v1'


from agent_system.policies.dyad.inference.source import digest, file_digest, source_metadata


def add_arguments(parser):
    parser.add_argument('--projector-init', help='Exact Alignment directory/file; never a AgenticRL checkpoint')
    parser.add_argument('--model-config', help='Saved AgenticRL model_config.json (required with Alignment initialization)')


def verify_identity(identity, expected):
    if identity.get('protocol') != PROTOCOL or identity.get('backend') != 'DyadAsyncLLM':
        raise ValueError('Endpoint is not the native Dyad backend')
    if identity.get('source_identity') != expected['identity'] or identity.get('source_identity_sha256') != expected['identity_sha256']:
        raise ValueError('Dyad checkpoint/projector/model_config identity mismatch')
    for field in ('projector_sha256', 'policy_sha256'):
        value = identity.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f'Dyad backend did not attest {field}')
    if identity.get('restoration_complete') is not True:
        raise ValueError('Dyad native restoration has not completed')


class DyadClient:
    def __init__(self, args, seed, expected=None):
        self.args, self.seed = args, seed
        self.expected = expected or source_metadata(args)
        self.base_url = args.agent_base_url.rstrip('/')
        self.calls = []
        self.identity = None

    def _request(self, method, path, payload=None):
        import httpx
        import os
        with httpx.Client(timeout=self.args.timeout, trust_env=False) as client:
            response = client.request(method, self.base_url + path, json=payload,
                                      headers={'Authorization': 'Bearer ' + os.getenv(self.args.agent_api_key_env, 'local-no-key')})
            if response.is_error:
                try:
                    detail = response.json()
                except ValueError:
                    detail = {'error': 'Non-JSON backend error response'}
                self.calls.append({'status': 'backend_error', 'http_status': response.status_code,
                                   'response': detail})
            response.raise_for_status()
            return response.json()

    def probe(self, weights=None):
        identity = self._request('GET', '/dyad/health')
        verify_identity(identity, self.expected)
        if identity.get('context_length') != self.args.context_length:
            raise ValueError('Dyad --context-length differs from the actual backend limit')
        self.identity = identity
        return identity

    def generate(self, messages, tools):
        import sys
        project = Path(__file__).resolve().parents[5]
        if str(project) not in sys.path:
            sys.path.insert(0, str(project))
        from agent_system.utils.thinking import resolve_chat_template_kwargs

        thinking = self.args.thinking
        sources = [getattr(self.args, key, None) for key in ('model', 'model_path', 'checkpoint')]
        template_kwargs = resolve_chat_template_kwargs(
            {} if thinking == 'default' else {'enable_thinking': thinking == 'on'}, model=sources)
        if 'enable_thinking' in template_kwargs:
            thinking = 'on' if template_kwargs['enable_thinking'] else 'off'
        if self.identity is None:
            self.probe()
        response = self._request('POST', '/dyad/act', {
            'messages': messages, 'tools': tools, 'schema_sha256': digest(tools),
            'source_identity_sha256': self.expected['identity_sha256'],
            'temperature': self.args.temperature, 'max_tokens': self.args.max_tokens,
            'seed': self.seed, 'thinking': thinking})
        verify_identity(response['identity'], self.expected)
        if response['identity'] != self.identity:
            raise ValueError('Dyad backend identity changed during an episode')
        if response.get('schema_sha256') != digest(tools):
            raise ValueError('Dyad backend used another tool schema')
        evidence = response.get('selection') or {}
        cfg = (response.get('action_content') or {}).get('action_config') or {}
        names = cfg.get('action_name_ids') or {}
        want_names = [t['function']['name'] for t in tools] + ['respond']
        if set(names) != set(want_names) or len(names) != len(want_names):
            raise ValueError('Dyad compiled schema candidate names differ from official tools')
        for tool in tools:
            definition = tool['function']
            mcp = cfg.get('actions', {}).get(definition['name'], {}).get('mcp') or {}
            if mcp.get('inputSchema') != definition['parameters'] or mcp.get('description', '') != definition.get('description', ''):
                raise ValueError('Dyad compiled schema did not preserve official tool definitions')
        head = response.get('head') or {}
        if head.get('projector_sha256') != self.identity['projector_sha256'] or head.get('shape', [None])[0] != cfg.get('total_size'):
            raise ValueError('Dyad dynamic head projector or capacity identity mismatch')
        if not isinstance(head.get('head_sha256'), str) or len(head['head_sha256']) != 64:
            raise ValueError('Dyad dynamic target head attestation is missing')
        allowed = list(names.values())
        selected = evidence.get('selected_id')
        if evidence.get('allowed_ids') != allowed or selected not in allowed or selected < cfg['num_embeddings_size']:
            raise ValueError('Dyad selection is not a legal expanded-head candidate')
        if names.get(evidence.get('name')) != selected or evidence.get('replay_verified') is not True:
            raise ValueError('Dyad selection evidence is inconsistent')
        if selected not in response['action_content'].get('raw_token_ids', []):
            raise ValueError('Dyad action_content is missing the expanded action decision')
        if response.get('compiled_schema_sha256') != digest(cfg):
            raise ValueError('Dyad compiled schema fingerprint mismatch')
        self.calls.append(response)
        return response['text']
