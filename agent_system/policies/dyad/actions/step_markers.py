"""Lightweight identities for the shared version-2 action protocol.

Callers must first establish that shared step protocol v2 is active.
Legacy evaluation/checkpoints must not be labeled with these identities.
"""


def shared_action_marker_protocol(environment: str, action_interface: str) -> str:
    """Return a manifest identity, not a tokenizer-dependent compiled schema."""
    protocols = {
        "alfworld": "react_lowercase_v2",
        "webshop": "react_lowercase_v2",
        "dive": "dive_native_tools_v2",
        "codegym": "codegym_native_v2",
    }
    if environment not in protocols:
        raise ValueError(f"Unsupported shared step environment: {environment!r}")
    if action_interface not in {"text", "dyad"}:
        raise ValueError(f"Unsupported shared step action interface: {action_interface!r}")
    return "native_text_v2" if action_interface == "text" else protocols[environment]
