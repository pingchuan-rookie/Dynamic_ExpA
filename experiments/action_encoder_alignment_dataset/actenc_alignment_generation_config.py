"""Generation inputs, intermediate artifacts, and JSONL I/O for Alignment.

Hand-maintained config and prompts live beside this module. Generated catalogues,
case plans, teacher responses, cache and manifest live under data/actenc_alignment/intermediate.
The shared runtime path resolver preserves DYAD_ALIGNMENT_DATA and checkpoint defaults.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

from agent_system.policies.dyad.training.action_encoder_alignment import actenc_alignment_config as runtime_config

PROJECT = runtime_config.PROJECT
data_root = runtime_config.data_root
final_dir = runtime_config.final_dir
dataset_path = runtime_config.dataset_path
runs_dir = runtime_config.runs_dir
ALIGNMENT_DIR = PROJECT / "experiments" / "dyad_training" / "action_encoder_alignment"
DATASET_SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = DATASET_SCRIPT_DIR / "config"
PROMPT_DIR = DATASET_SCRIPT_DIR / "prompts"


def input_path(name: str) -> Path:
    """Resolve stable manifest input identities after source-file renaming."""
    source_names = {
        "config/action_names.yaml", "config/unseen_action_names.yaml",
        "config/entry_marker.yaml", "config/generation.yaml",
        "prompts/prompt_a.txt", "prompts/prompt_b.txt", "prompts/prompt_c.txt",
    }
    if name in source_names:
        relative = Path(name)
        return DATASET_SCRIPT_DIR / relative.parent / f"actenc_alignment_{relative.name}"
    return data_root() / name

#: The five domains, in the order the strategy document lists them. Order matters: it seeds the
#: per-domain random streams, so shuffling this list silently changes every sampled case.
DOMAIN_ORDER = (
    "spatial_manipulation",
    "state_change",
    "information",
    "workflow_communication",
    "transformation",
)

MCP_SIZES = tuple(range(4, 11))

#: `action_set_form` values, as spelled in the dataset schema (strategy document section 1.10).
FORM_MCP = "tool-specification"
FORM_NL = "natural-language"
FORMS = (FORM_MCP, FORM_NL)




def catalogue_dir() -> Path:
    """Strong-LLM output, so it lives with the data rather than with the scripts.

    A function rather than a module constant because it follows `DYAD_ALIGNMENT_DATA`: the dataset
    gate builds a whole tree in a temp directory, and a catalogue path frozen at import time would
    make it read the shipped catalogues while writing everything else to the temp one.
    """
    return intermediate_dir()


def intermediate_dir() -> Path:
    return data_root() / "intermediate"






def manifest_path() -> Path:
    return intermediate_dir() / "manifest.yaml"


def cache_dir() -> Path:
    return intermediate_dir() / "llm_cache"






# --------------------------------------------------------------------------- reading the inputs


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; Alignment cannot run without it")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_generation() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "actenc_alignment_generation.yaml")


def load_action_names(*, unseen: bool = False) -> dict[str, list[str]]:
    name = "actenc_alignment_unseen_action_names.yaml" if unseen else "actenc_alignment_action_names.yaml"
    domains = _load_yaml(CONFIG_DIR / name)["domains"]
    missing = [d for d in DOMAIN_ORDER if d not in domains]
    if missing:
        raise ValueError(f"{name} is missing domains {missing}")
    extra = [d for d in domains if d not in DOMAIN_ORDER]
    if extra:
        raise ValueError(f"{name} has unknown domains {extra}")
    return {d: list(domains[d]) for d in DOMAIN_ORDER}


def load_entry_markers() -> tuple[list[str], list[str]]:
    raw = _load_yaml(CONFIG_DIR / "actenc_alignment_entry_marker.yaml")
    return list(raw["entry_marker_rule"]), list(raw["entry_marker"])


def load_mcp_entry_marker_rules() -> list[str]:
    raw = _load_yaml(CONFIG_DIR / "actenc_alignment_entry_marker.yaml")
    rules = raw["mcp_entry_marker_rule"]
    if not isinstance(rules, list) or len(rules) != len(raw["entry_marker_rule"]) or any(
        not isinstance(rule, str) or "{entry_marker}" not in rule for rule in rules
    ):
        raise ValueError("MCP rules must cover the same rule IDs and include {entry_marker}")
    return list(rules)


def policy_prompt_inputs(generation: dict[str, Any]) -> tuple[dict[str, list[str]], list[str], dict[str, str]]:
    """Resolve format-specific syntax once for the builder and offline reconstruction."""
    nl_rules, markers = load_entry_markers()
    policy = generation.get("policy_prompt") or {}
    examples = {FORM_NL: policy.get("format_example"), FORM_MCP: policy.get("mcp_format_example")}
    for form, example in examples.items():
        action_slot = "{action_json}" if form == FORM_MCP else "{action}"
        if not isinstance(example, str) or "{entry_marker}" not in example or action_slot not in example:
            raise ValueError(f"policy_prompt format_example for {form} needs {{entry_marker}} and {action_slot}")
    return {FORM_NL: nl_rules, FORM_MCP: load_mcp_entry_marker_rules()}, markers, examples


def load_prompt(letter: str) -> str:
    path = PROMPT_DIR / f"actenc_alignment_prompt_{letter}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    return path.read_text(encoding="utf-8")


def catalogue_path(kind: str, *, unseen: bool = False) -> Path:
    if kind not in ("mcp", "nl"):
        raise ValueError(f"catalogue kind must be mcp or nl, got {kind!r}")
    suffix = "_unseen" if unseen else ""
    return catalogue_dir() / f"{kind}{suffix}.yaml"


def load_mcp_catalogue(*, unseen: bool = False) -> dict[str, dict[str, Any]]:
    """`name -> MCP Tool object`. Flat across domains: names are unique repository-wide."""
    raw = _load_yaml(catalogue_path("mcp", unseen=unseen))
    return {tool["name"]: tool for tool in raw["tools"]}


def load_nl_catalogue(*, unseen: bool = False) -> dict[str, str]:
    """`name -> natural-language description`, one entry per tool in the MCP catalogue."""
    raw = _load_yaml(catalogue_path("nl", unseen=unseen))
    return {name: str(text) for name, text in raw["tools"].items()}


def load_both_catalogues() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Seen and unseen merged. The builder needs one lookup: an unseen case's admissible set
    mixes both pools, and two dictionaries at that point is two chances to consult the wrong one."""
    mcp = dict(load_mcp_catalogue())
    mcp.update(load_mcp_catalogue(unseen=True))
    nl = dict(load_nl_catalogue())
    nl.update(load_nl_catalogue(unseen=True))
    return mcp, nl


def domain_of(action: str) -> str:
    """Which domain an action belongs to, across both pools."""
    for unseen in (False, True):
        for domain, actions in load_action_names(unseen=unseen).items():
            if action in actions:
                return domain
    raise KeyError(
        f"{action!r} is in neither actenc_alignment_action_names.yaml "
        "nor actenc_alignment_unseen_action_names.yaml"
    )


# --------------------------------------------------------------------------- jsonl


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    out = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid json: {exc}") from exc
    return out


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Resolve Alignment storage paths")
    parser.add_argument("path", choices=("dataset", "runs", "intermediate"))
    args = parser.parse_args()
    print({"dataset": dataset_path, "runs": runs_dir, "intermediate": intermediate_dir}[args.path]())


if __name__ == "__main__":
    main()
