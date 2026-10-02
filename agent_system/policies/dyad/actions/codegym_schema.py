"""Build native CodeGym action schemas from environment action metadata.

Shared by runtime task adaptation and offline schema export; performs no I/O.
"""

MAX_FREE_VALUE_TOKENS = 128


def build_codegym_schema(env: dict) -> dict:
    """Dyad free-argument yaml, emitted in the **new unified format** (not the legacy actions_schema one).

    The legacy format has to go through _normalize_legacy in agent_system/policies/dyad/actions/schema_compiler.py, which rebuilds the
    unified raw from actions_schema/routing and drops any extra top-level key -- so argument_order/
    env_serialize/markers set alongside it would be silently ignored and the schema would compile as a
    plain fixed-order codegym one. The free-argument schema declares no params at all, so the unified format is both shorter
    and the only one that carries these fields; it also lines it up with calc_base.yaml and
    alfworld_base.yaml, which are hand-written in the same shape.

    Markers are the paper's B2 pair. The raw-command schema's premise is that what lands between them is byte-for-byte the
    grpo_react baseline's call, so they must be the exact markers grpo_react's parser looks for.
    """
    actions = {}
    for action_name in env["actions"]:
        n_params = len(env["params"].get(action_name, []))
        actions[action_name] = {
            "description": f"call {action_name} with {n_params} param(s)" if n_params
                           else f"call {action_name}",
            # No trailing space: Qwen2.5's canonical split puts the space with the next token
            # (': ' + '{' would write a standalone space token 220 and push the model's '{' off
            # distribution). The model writes the leading space itself; _clean_open_value strips it and
            # unified_action_to_raw_command re-joins with exactly one space, which JSON ignores.
            "surface_form": '[{"name": "%s", "parameters":' % action_name,
            "init_weight_with": 0,
            "params": {},          # argument_order=free requires this to be empty
        }
    return {
        "env_name": f"{env['env_name']}_base",
        "source": env.get("source"),
        "router": "unified",
        "mode": "codegym",
        "env_serialize": "codegym_raw",
        "template_style": "base",
        "markers": {
            "enter": "<|FunctionCallBegin|>",
            "exit": "<|FunctionCallEnd|>",
            "argument_value_end": "\n",
            "value_end_on_suffix": False,
            "value_end_on_exit": True,        # the model writes '<|' to close the call
            "max_value_tokens": MAX_FREE_VALUE_TOKENS,
            "emit_eos": True,
        },
        "argument_order": "free",
        "head": {
            "action_name": True,
            "argument_key": False,
            "closed_value": False,
            "init_from": "mean_pool",
        },
        "actions": actions,
        "value_sets": {},
    }

