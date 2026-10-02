"""Environment adapters, pools and import-light execution workers.

Keep exports lazy: importing a worker also imports this parent package, and must
not initialize Ray, verl or model backends in the environment process.
"""

from importlib import import_module

_EXPORTS = {
    'BaseEnvPool': 'core.pool',
    'AlfworldEnvPool': 'alfworld.pool',
    'AlfworldEnvWorker': 'alfworld.worker',
    'load_alfworld_games': 'alfworld.pool',
    'AlfworldLocalEnvTool': 'alfworld.tool',
    'CodeGymEnvPool': 'codegym.pool',
    'CodeGymEnvWorker': 'codegym.worker',
    'parse_codegym_env_str': 'codegym.pool',
    'CodeGymLocalEnvTool': 'codegym.tool',
    'CalcEnvPool': 'calc.pool',
    'CalcEnvWorker': 'calc.worker',
    'CalcLocalEnvTool': 'calc.tool',
    'WebShopEnvPool': 'webshop.pool',
    'WebShopEnvWorker': 'webshop.worker',
    'WebShopLocalEnvTool': 'webshop.tool',
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    if not module.startswith('core.'):
        module = 'backends.' + module
    value = getattr(import_module(f'{__name__}.{module}'), name)
    globals()[name] = value
    return value
