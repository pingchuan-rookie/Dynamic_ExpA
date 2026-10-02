"""Environment user/judge settings and the fixed t2bench task inventory."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
COMMIT = 'a2c024725189473d2d7cea3a5cfdbcc67478e41f'
COUNTS = {'retail': 114, 'airline': 50, 'telecom': 114}
VERSION = '1.1.0'

def resolve_environment_roles(args, env=None, default_base_url='https://api.openai.com/v1'):
    """Resolve only environment roles; generic OpenAI variables never select policy routing."""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from .trapi import resolve_trapi_config

    if getattr(args, 'agent_provider', 'openai') == 'trapi':
        raise ValueError('TRAPI is supported only for environment user/judge roles')
    for role in ('user', 'judge'):
        provider = getattr(args, f'{role}_provider') or 'trapi'
        setattr(args, f'{role}_provider', provider)
        if provider == 'trapi':
            config = resolve_trapi_config(model=getattr(args, f'{role}_model'),
                                          base_url=getattr(args, f'{role}_base_url'), env=env)
            for key, value in config.items():
                setattr(args, f'{role}_{key}', value)
        elif not getattr(args, f'{role}_base_url'):
            setattr(args, f'{role}_base_url', default_base_url)
    return args

def thinking_settings(a, role):
    """Resolve role policy before reading credentials or constructing requests."""
    import sys
    project = ROOT
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from agent_system.utils.thinking import resolve_chat_template_kwargs

    prefix = '' if role == 'agent' else f'{role}_'
    thinking = getattr(a, prefix + 'thinking')
    if thinking not in ('default', 'on', 'off'):
        raise ValueError(f'Invalid {role} thinking setting')
    model = a.model if role == 'agent' else getattr(a, f'{role}_model')
    sources = [model]
    if role == 'agent':
        sources.extend(getattr(a, key, None) for key in ('model_path', 'checkpoint', 'tokenizer'))
    settings = resolve_chat_template_kwargs(
        {} if thinking == 'default' else {'enable_thinking': thinking == 'on'}, model=sources)
    if settings and getattr(a, f'{role}_provider') != 'openai':
        raise ValueError(f'{role} thinking override requires provider openai')
    return settings
