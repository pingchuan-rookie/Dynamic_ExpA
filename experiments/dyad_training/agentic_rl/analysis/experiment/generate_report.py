#!/usr/bin/env python3
# Copyright 2025 ExpA_sys
"""Build one analysis report per (experiment, env), from the code and from a real run.

Why a generator rather than hand-written markdown: every number and every quoted prompt in these
reports is something the code can produce on demand, and a hand-written copy of it starts drifting
the moment a template changes. The drift is invisible: a report that quotes last week's encoder
input reads exactly like one that quotes this week's. So the prompts are rebuilt here, the
parameter counts are counted here, and the "who trained" numbers are parsed out of the run's own
diagnostics rather than retyped.

Reports print to stdout; --out-dir writes generated reports to a chosen scratch directory.
Do not check generated reports into the source tree.

Usage (cwd=Dynamic_ExpA):
    .venvs/expa-verl/bin/python experiments/dyad_training/agentic_rl/analysis/experiment/generate_report.py --experiment train --env gsm8k
    .venvs/expa-verl/bin/python experiments/dyad_training/agentic_rl/analysis/experiment/generate_report.py --all --out-dir /tmp/reports
"""

from __future__ import annotations

import argparse
import json
import re
import os
import yaml
import sys
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT))

from agent_system.utils.artifact_paths import artifact_root

OUTPUTS = artifact_root(PROJECT) / "outputs"

# Stage -> title, entrypoint, description form, and action space.
@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    entrypoint: str
    description: str
    surface_form: str
    values: str
    question: str


def load_stages():
    config = yaml.safe_load((PROJECT / 'experiments/dyad_training/agentic_rl/train_eval/config/experiments.yaml').read_text())
    entries = {'train': {'overrides': {}, 'benchmarks': config['main']['benchmarks']}}
    entries.update({path.stem: yaml.safe_load(path.read_text()) for path in
                    (PROJECT / 'experiments/dyad_training/agentic_rl/train_eval/ablation').glob('*.yaml')})
    stages = {}
    for key, entry in entries.items():
        settings = {**config['main']['defaults'], **entry.get('overrides', {})}
        stages[key] = Stage(key, '主实验' if key == 'train' else entry.get('title', key),
            f'train.sh {{env}} --experiment {key}', settings['encoder_description'],
            settings['surface_form'], settings['values'],
            'encoder 输入、projector 动作权重及实际训练范围')
    return stages, {key: entry['benchmarks'] for key, entry in entries.items()}


STAGES, BENCHMARKS = load_stages()
MODEL_OVERRIDE = False
MODEL = os.environ.get('MODEL_PATH', 'Qwen/Qwen2.5-0.5B-Instruct')



def form_module_path(form: str) -> str:
    """Resolve a description-form implementation from the registry."""
    from agent_system.policies.dyad.models.action_descriptions import _MODULES

    return _MODULES[form].replace(".", "/") + ".py"


def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


def encoder_input(env: str, surface_form: str, values: str, form: str) -> dict:
    """Regenerate the exact encoder input to include in the report."""
    from agent_system.policies.dyad.models.action_descriptions import build_action_prompt, build_catalogue
    from agent_system.policies.dyad.models.action_descriptions import describe_action_config
    from agent_system.policies.dyad.actions.schema_config import compile_schema_file

    if env == "codegym":
        rel = next(iter(sorted(
            (PROJECT / "agent_system/policies/dyad/actions/schemas/codegym_generated" / surface_form).glob("*.yaml"))))
        rel = rel.relative_to(PROJECT / "agent_system/policies/dyad/actions/schemas").as_posix()
    else:
        rel = f"{env}/{surface_form}{'_closed' if values == 'closed' else ''}.yaml"
    tok = _tokenizer()
    cfg = compile_schema_file(tok, len(tok), PROJECT / "agent_system/policies/dyad/actions/schemas" / rel)
    docs = describe_action_config(cfg)
    catalogue = build_catalogue(docs, form=form)
    first = build_action_prompt(docs[0], form=form, catalogue=catalogue)
    per_action_only = build_action_prompt(docs[0], form=form)
    prompts = [build_action_prompt(d, form=form, catalogue=catalogue) for d in docs]
    return {
        "schema": rel,
        "env_name": cfg["env_name"],
        "rows": len(docs),
        "kinds": {k: sum(1 for d in docs if d.kind == k) for k in ("tool", "parameter", "value")},
        "catalogue": catalogue,
        "first_full": first,
        "first_per_action": per_action_only,
        "first_name": docs[0].name,
        "chars": sum(len(p) for p in prompts),
        "fingerprint": _fingerprint(prompts),
    }


def _fingerprint(prompts: list[str], representation: str = "final_layer_hidden_states") -> str:
    """Fingerprint prompts, tokenizer, model, and representation using the encoder contract."""
    from agent_system.policies.dyad.models.encoder_cache import prompt_fingerprint
    return prompt_fingerprint(prompts, MODEL, f"{MODEL}#{representation}")


def projector_facts(hidden: int = 896) -> dict:
    from agent_system.policies.dyad.models.action_projector import build_projector
    from agent_system.policies.dyad.models.action_head import DirectActionHead

    out = {}
    for name in ("mean", "mlp", "attention"):
        projector = build_projector(name, hidden, **({"num_heads": 8} if name == "attention" else {}))
        head = DirectActionHead(name, hidden, hidden)
        out[name] = {
            "projector_total": sum(p.numel() for p in projector.parameters()),
            "projector_breakdown": {n: p.numel() for n, p in projector.named_parameters()},
            "head_total": sum(p.numel() for p in head.parameters()),
        }
    return out


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_RAY = re.compile(r"^\([^)]*pid=\d+[^)]*\)\s+")


def _clean(line: str) -> str:
    return _RAY.sub("", _ANSI.sub("", line.rstrip("\n")))


def run_facts(log: Path) -> dict:
    """Read training configuration and trainable-parameter evidence from a run log."""
    facts = {"log": str(log), "setting": None, "prompts": None, "trainable": None,
             "phase": None, "steps": []}
    if not log.exists():
        return facts
    for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _clean(raw)
        if "[Dyad] encoder setting:" in line and facts["setting"] is None:
            facts["setting"] = line.split("encoder setting:", 1)[1].strip()
        if "[Dyad] encoder prompts:" in line and facts["prompts"] is None:
            facts["prompts"] = line.split("encoder prompts:", 1)[1].strip()
        m = re.search(r"\[Dyad\] training_schedule (\w+): .*trainable=(\{.*\})", line)
        if m and facts["trainable"] is None:
            facts["phase"] = m.group(1)
            facts["trainable"] = m.group(2)
    return facts


def grad_facts(run_dir: Path) -> dict:
    """Read gradient norms and head-weight signatures from the run JSONL events."""
    out = {"grad_norms": [], "head_sigs": [], "run_dir": str(run_dir)}
    if not run_dir.exists():
        return out
    for f in sorted(run_dir.glob("dyad_actor_pid*.jsonl")):
        for line in f.read_text(errors="replace").splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") == "optimizer_step":
                out["grad_norms"].append(
                    round((e["grad_norm"].get("mean", 0) if isinstance(e.get("grad_norm"), dict) else e.get("grad_norm")) or 0, 5))
    for f in sorted(run_dir.glob("dyad_vllm_model_runner_pid*.jsonl")):
        sigs = []
        for line in f.read_text(errors="replace").splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") == "action_head_initialized":
                w = (((e.get("action_head") or {}).get("parameters") or {})
                     .get("weight") or {}).get("value") or {}
                sigs.append(w.get("mean"))
        if sigs:
            out["head_sigs"].append(sigs)
    return out


def latest_run(experiment: str, env: str) -> Path | None:
    candidates = []
    for meta in OUTPUTS.glob(f'*/agentic_rl/*/{env}/*/resolved_config.json'):
        config = json.loads(meta.read_text())
        if config.get('experiment') == experiment and config.get('algorithm') == 'dyad':
            candidates.append(meta.parent)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None



HEADER = """<!-- GENERATED，不签入。重新生成：
     .venvs/expa-verl/bin/python experiments/dyad_training/agentic_rl/analysis/experiment/generate_report.py --experiment {key} --env {env}
     贴出来的 prompt 与参数量都是**当场从代码重算**的，不是抄的；抄的会悄悄过期。 -->

# {key} {title} · {env}

**这份报告回答**：{question}

| | |
|---|---|
| 入口 | `bash experiments/shared/train_eval/{entrypoint}` |
| 模型 | `{model}`（hidden={hidden}） |
| 动作空间 | surface_form=`{surface_form}` × values=`{values}` → `{schema}` |
| 描述形式 | `{description}` |
| head 行数 | {rows}（{kinds}） |
"""


def _fence(text: str, lang: str = "text") -> str:
    fence = "`" * max(3, max((len(m) for m in re.findall(r"`+", text)), default=0) + 1)
    return f"{fence}{lang}\n{text}\n{fence}"


def section_encoder_input(enc: dict, stage: Stage) -> str:
    return f"""
---

## 1. action encoder 的输入：原文

encoder 读的每一行都是 **`catalogue` + `这一个动作`** 两段拼起来的。
catalogue 对该 env 的每一行都相同（离线生成一次），per-action 段每行不同。

代码路径：

| 环节 | 路径 |
|---|---|
| action_config → 每个扩展 id 一份 `ActionDoc` | `agent_system/policies/dyad/models/action_descriptions.py::describe_action_config` |
| `ActionDoc` → 文本（一种描述形式一个模块） | `{form_module_path(stage.description)}` |
| 两段拼起来 | `agent_system/policies/dyad/models/action_descriptions.py::build_action_prompt` |
| 训练时谁来调它 | `agent_system/policies/dyad/models/action_head_factory.py::encoder_prompts` |
| 拼好的文本 → head 的每一行 | `agent_system/policies/dyad/models/action_head_factory.py::build_action_head` |

### 1.1 catalogue（该 env 每一行都读到的前缀）

{_fence(enc['catalogue'].rstrip())}

### 1.2 第一个动作 `{enc['first_name']}` 的 per-action 段

{_fence(enc['first_per_action'])}

### 1.3 这一行的完整输入（= 上面两段拼接，逐字）

{_fence(enc['first_full'])}

**数字**：{enc['rows']} 行 head，全部 prompt 合计 {enc['chars']:,} 字符，
指纹 `{enc['fingerprint'][:32]}...`。

指纹由按 head 行序排列的 prompt、tokenizer、模型和 representation 共同确定。
只有这些输入全部一致时，才应与对应 encoder 缓存指纹比较。
"""


def section_projector(facts: dict) -> str:
    rows = "\n".join(
        f"| `{name}` | {info['projector_total']:,} | {info['head_total']:,} |"
        for name, info in facts.items()
    )
    return f"""
---

## 2. projector 与 DirectActionHead

projector 将一段变长的 encoder token 表示压成一个向量。
DirectActionHead 将该向量映射到 policy hidden 维度，再按 scale 设置处理范数，直接作为动作头的一行权重。
两侧维度相同时，维度映射是 Identity。

| 环节 | 路径 |
|---|---|
| `MeanProjector` / `MlpProjector` / `AttentionProjector` | `agent_system/policies/dyad/models/action_projector.py` |
| 动作权重生成 | `agent_system/policies/dyad/models/action_head.py::DirectActionHead` |

以下参数量由当前模块现场统计，encoder 与 policy hidden 均使用本报告模型维度。

| projector | projector 参数量 | DirectActionHead 参数量 |
|---|---:|---:|
{rows}

`mean` 对有效 token 表示求平均，不含可训练参数。
`mlp` 学习 token 分数，以 softmax 权重汇总原始表示。
`attention` 使用可学习 query 与 key/value/output 投影汇总表示。
动作权重来自 encoder 表示，不是从词表输出权重中直接平均得到。
"""


def section_training(stage: Stage, rf: dict, gf: dict, projector: dict) -> str:
    trainable = rf.get("trainable") or "（日志里没有这一行）"
    grads = gf.get("grad_norms") or []
    nonzero = sum(1 for g in grads if g and g > 0)
    sigs = gf.get("head_sigs") or []
    changed = [len(set(s)) for s in sigs]

    verdict = _training_verdict(stage.key, trainable, nonzero, len(grads), changed)
    return f"""
---

## 3. 到底是谁在训

「冻结有没有生效」在一次 run 浪费掉之前是不可见的，而且弄错了**不会抛异常** --
它只会训错东西，同时每个指标都很正常。所以这里用三层口径，缺一层都可能把「没训」看成「训了」。

### 第 1 层 · 构造期数出来的可训练参数

```
[Dyad] encoder setting: {rf.get('setting') or '（无）'}
schedule={rf.get('phase') or '?'} trainable={trainable}
```

可训练参数由 `models/action_head_factory.py::apply_training_schedule` 统计。
训练范围应结合本次运行的 schedule 和 encoder setting 解读，不由报告编号推断。

### 第 2 层 · 运行期真的有梯度

```
grad_norm 逐 optimizer step = {grads}
其中 > 0 的：{nonzero}/{len(grads)}
```

非单样本 GRPO 组内奖励相同时，组内中心化优势为零，因此策略 surrogate 可能没有梯度。
总梯度还取决于 KL、entropy 等其他目标项。
是否发生参数更新需要结合 optimizer 状态与实际权重判断，不能仅看梯度范数。

### 第 3 层 · head 的权重真的动了

```
每个 rank 的 action_head_initialized 权重签名（mean），去重后的个数 = {changed}
```

签名来自日志记录的权重摘要，不是完整的权重版本校验。
签名相同不能证明权重未更新，policy 梯度非零也不要求动作头发生变化。

### 结论

{verdict}
"""


def _training_verdict(key: str, trainable: str, nonzero: int, total: int, changed: list) -> str:
    m = re.search(r"'actor':\s*(\d+).*?'projector':\s*(\d+)", trainable or "")
    if not m:
        return ("**无法判定**：这次 run 的日志里没有 `[Dyad] training_schedule ... trainable=` 那一行，"
                "也就是说构造期的冻结报告没打出来。先把 run 跑出来再看。")
    actor, projector = int(m.group(1)), int(m.group(2))
    lines = [f"- 第 1 层：记录的可训练参数为 actor **{actor:,}**、projector **{projector:,}**。"
             "是否符合预期需要与本次运行的 schedule 对照。"]
    if total == 0:
        lines.append("- 第 2 层：这次 run 没有 optimizer_step 事件，训练没走到那一步。")
    elif nonzero == 0:
        lines.append(f"- 第 2 层：{total} 个 step 没有记录到非零 grad_norm。"
                     "需要结合优势、其他目标项及 optimizer 状态判断原因。")
    else:
        lines.append(f"- 第 2 层：{nonzero}/{total} 个 step 有非零梯度 → 梯度确实流到了可训练的那一侧。")
    if not changed:
        lines.append("- 第 3 层：没有 action_head_initialized 事件，拿不到权重签名。")
    elif all(c <= 1 for c in changed):
        lines.append("- 第 3 层：head 权重签名没有观察到变化。"
                     "冻结生成侧或仅更新 policy 时可以如此，签名相同不能证明未重建。")
    else:
        lines.append(f"- 第 3 层：观察到 head 权重签名变化（去重后 {changed}）。"
                     "这不能单独证明每次同步都正确。")
    return "\n".join(lines)


def build_report(stage: Stage, env: str, run_dir: Path | None = None) -> str:
    run_dir = run_dir or latest_run(stage.key, env)
    if run_dir is None or not run_dir.is_dir():
        raise ValueError('No matching run found; pass --run-dir explicitly')
    metadata_path = run_dir / 'resolved_config.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if metadata and (metadata.get('experiment') != stage.key or metadata.get('benchmark') != env):
        raise ValueError('Run metadata does not match the requested experiment / benchmark')
    model = metadata.get('hydra', {}).get('actor_rollout_ref.model.path')
    if model and not MODEL_OVERRIDE:
        globals()['MODEL'] = model
    model_config = run_dir / 'model_config.json'
    if model_config.exists():
        saved = json.loads(model_config.read_text()).get('model', {})
        stage = replace(stage, description=saved.get('DYAD_ENCODER_DESCRIPTION', stage.description),
                        surface_form=saved.get('DYAD_SURFACE_FORM', stage.surface_form),
                        values=saved.get('DYAD_VALUES', stage.values))
    enc = encoder_input(env, stage.surface_form, stage.values, stage.description)
    from transformers import AutoConfig
    from agent_system.utils.hf_config import text_hidden_size
    import torch
    hidden = text_hidden_size(AutoConfig.from_pretrained(MODEL))
    with torch.device('meta'):
        projectors = projector_facts(hidden)
    log = next((run_dir / name for name in ('trainer.log', 'dyad.log', 'main_dyad.log')
                if (run_dir / name).is_file()), run_dir / 'trainer.log')
    rf = run_facts(log)
    gf = grad_facts(run_dir)

    body = HEADER.format(
        key=stage.key, title=stage.title, env=env, question=stage.question, model=MODEL, hidden=hidden,
        entrypoint=stage.entrypoint.format(env=env), surface_form=stage.surface_form, values=stage.values,
        schema=enc["schema"], description=stage.description, rows=enc["rows"],
        kinds="、".join(f"{k} {v}" for k, v in enc["kinds"].items() if v),
    )
    body += section_encoder_input(enc, stage)
    body += section_projector(projectors)
    body += section_training(stage, rf, gf, projectors)
    body += f"""
---

## 附：这份报告是从哪来的

| | |
|---|---|
| 训练日志 | `{log}` |
| run 目录 | `{run_dir.relative_to(PROJECT) if run_dir.is_relative_to(PROJECT) else run_dir}` |
| 自动分析 | 同目录下 `analysis/verify_dyad.md`、`analysis/decode_trajectories.md` |

prompt 原文与参数量是本脚本**当场重算**的，不是从别处抄的；
「谁在训」的三层数字是从上面那个 run 自己的日志与 jsonl 里读出来的。
"""
    return body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiment", choices=sorted(STAGES))
    ap.add_argument("--model", default=None, help="tokenizer model id or local snapshot")
    ap.add_argument("--run-dir", type=Path, help="specific run to analyze (single experiment only)")
    ap.add_argument("--env", choices=sorted({env for envs in BENCHMARKS.values() for env in envs}))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-dir", type=Path,
                    help="写到这个目录的 <experiment>/<env>.md；不给就打到 stdout")
    args = ap.parse_args()
    globals()["MODEL_OVERRIDE"] = args.model is not None
    if args.model:
        globals()["MODEL"] = args.model
    if args.all and args.run_dir:
        ap.error("--run-dir cannot be combined with --all")

    todo = []
    if args.all:
        todo = [(s, e) for s in STAGES.values() for e in BENCHMARKS[s.key]]
    elif args.experiment and args.env:
        if args.env not in BENCHMARKS[args.experiment]:
            ap.error("This experiment does not support the selected benchmark")
        todo = [(STAGES[args.experiment], args.env)]
    else:
        ap.error("要么 --all，要么 --experiment 与 --env 同时给")

    # Keep stdout as report content; send progress and failures to stderr for safe redirection.
    written = 0
    for stage, env in todo:
        try:
            text = build_report(stage, env, args.run_dir)
        except Exception as exc:  # Continue generating reports for environments with available data.
            print(f"[gen-reports] SKIP {stage.key}/{env}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        if args.out_dir:
            path = args.out_dir / stage.key / f"{env}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"[gen-reports] wrote {path} ({len(text):,} chars)", file=sys.stderr)
        else:
            if written:
                print()
            print(text)
        written += 1

    if not written:
        print(f"[gen-reports] FAIL {len(todo)} 份一份都没生成出来", file=sys.stderr)
        return 1
    return 0 if written == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())
