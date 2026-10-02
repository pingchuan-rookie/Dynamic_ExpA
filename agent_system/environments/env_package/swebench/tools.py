"""Versioned public SWE-bench scaffold shared by baseline and Dyad."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

SCAFFOLD_VERSION = "swebench-tools-v1"
SYSTEM_PROMPT = """You are fixing a software issue in an isolated, offline repository at /testbed.
Inspect the repository, implement the requested fix, and run relevant local tests.
Use the provided bash, search, and editor tools. Dependencies are already installed;
network access is unavailable. Shell commands start in /testbed and do not persist
shell state between calls. The testbed conda environment is activated when available.
Your final submission is the actual working-tree diff, including new files, not text
in your response. Do not edit Git metadata or rely on repository history. Hidden
evaluation tests and reference solutions are unavailable. Call finish when ready.
"""


def _tool(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required, "additionalProperties": False}}}


ACTION_TOOLS = [
    _tool("bash", "Run a shell command in the offline /testbed repository. Output and execution time are bounded.",
          {"command": {"type": "string", "minLength": 1}}, ["command"]),
    _tool("search", "Search literal text recursively within the repository, returning path, line number and matching line.",
          {"query": {"type": "string", "minLength": 1}, "path": {"type": "string", "default": "."}}, ["query"]),
    _tool("editor", "Read, create, or replace exact text in a repository file. Replace requires exactly one matching old_text. Create refuses existing files.",
          {"operation": {"type": "string", "enum": ["read", "create", "replace"]},
           "path": {"type": "string", "minLength": 1}, "text": {"type": "string"},
           "old_text": {"type": "string"}}, ["operation", "path"]),
    _tool("finish", "Finish the attempt and submit the current repository changes for separate official evaluation.",
          {"summary": {"type": "string"}}, []),
]
SCHEMA_HASH = hashlib.sha256(json.dumps(ACTION_TOOLS, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


MINI_TOOL = _tool(
    "mini_swe_agent",
    "Delegate repository inspection, implementation and local testing to a complete mini-swe-agent. "
    "It modifies the current workspace and returns its status and actual patch; call finish for official scoring.",
    {"instruction": {"type": "string", "minLength": 1, "maxLength": 16000}}, ["instruction"])


def action_tools(mini_agent=False):
    return deepcopy([MINI_TOOL, ACTION_TOOLS[-1]] if mini_agent else ACTION_TOOLS)


def scaffold_identity(mini_agent=False):
    if not mini_agent:
        return SCAFFOLD_VERSION, SCHEMA_HASH
    schema = json.dumps(action_tools(True), sort_keys=True, separators=(",", ":"))
    return "swebench-mini-swe-agent-2.4.6-v1", hashlib.sha256(schema.encode()).hexdigest()


def build_initial_messages(instance, mini_agent=False):
    """Explicit allowlist: no interpolation of the raw evaluator task object."""
    from agent_system.environments.prompts.swebench import build_swebench_messages
    task_description = (f"Repository: {instance['repo']}\n"
                        f"Task: {instance['instance_id']}\n\n{instance['problem_statement']}")
    system = SYSTEM_PROMPT
    if mini_agent:
        system = system.replace("Use the provided bash, search, and editor tools.",
                                "Use mini_swe_agent to delegate implementation and testing to a complete coding agent.")
    return build_swebench_messages(task_description, action_tools(mini_agent), system_prompt=system)


# This program only runs inside an unprivileged, network-isolated Docker container.
# JSON is passed as a separate argv entry, never interpolated into a shell program.
FILE_TOOL_PROGRAM = r'''
import json, os, pathlib, sys
args = json.loads(sys.argv[2])
def within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
root = pathlib.Path('/testbed').resolve()
p = (root / args.get('path', '.')).resolve()
if not within(p, root) or '.git' in p.relative_to(root).parts:
    raise ValueError('Path must stay inside /testbed and outside Git metadata')
if sys.argv[1] == 'search':
    hits = 0
    paths = [p] if p.is_file() else p.rglob('*')
    for f in paths:
        if '.git' in f.parts or not f.is_file() or f.is_symlink():
            continue
        if not within(f.resolve(), root):
            continue
        try:
            with f.open(errors='replace') as stream:
                for n, line in enumerate(stream, 1):
                    if args['query'] in line:
                        print(str(f.relative_to(root)) + ':' + str(n) + ':' + line.rstrip()[:4096])
                        hits += 1
                        if hits >= 200:
                            print('[search limited to 200 matches]')
                            sys.exit(0)
        except (OSError, UnicodeError):
            continue
else:
    operation = args['operation']
    if operation == 'read':
        with p.open(errors='replace') as stream:
            print(stream.read(65536), end='')
    elif operation == 'create':
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('x') as stream:
            stream.write(args['text'])
        print('Created ' + str(p.relative_to(root)))
    elif operation == 'replace':
        old = args['old_text']
        if not old:
            raise ValueError('old_text must be nonempty')
        content = p.read_text()
        if content.count(old) != 1:
            raise ValueError('old_text must match exactly once')
        p.write_text(content.replace(old, args['text'], 1))
        print('Updated ' + str(p.relative_to(root)))
'''
