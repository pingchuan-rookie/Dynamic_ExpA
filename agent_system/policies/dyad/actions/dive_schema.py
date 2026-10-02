"""Preserve DIVE schema identity while sharing the native tools compiler."""
from agent_system.policies.dyad.actions.native_tools import compile_tools as _compile_tools, verify_trace


def compile_tools(tokenizer, vocab_size, tools, capacity):
    return _compile_tools(tokenizer, vocab_size, tools, capacity, environment="dive")
