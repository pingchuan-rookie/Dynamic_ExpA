"""Evaluate an ALFWorld ReAct baseline with online interaction and action matching.

Read task identities and ground truth from data/alfworld/dyad_stmt JSONL files.
Build native ReAct prompts and use the environment's available actions at each
step. Match Action: text by exact, case-insensitive, and normalized forms; fuzzy
matching is optional. The HTTP lifecycle is create, reset(game), step, close.

The default policy uses an OpenAI-compatible chat-completions endpoint.
See README.md for commands and evaluation conditions.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Standalone evaluator environments do not install Dyad; its policy helper is stdlib-only.
_PROJECT = Path(__file__).resolve().parents[6]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))
from agent_system.utils.thinking import resolve_chat_template_kwargs

SYSTEM_PROMPT = """You are an intelligent agent solving tasks in a simulated text-based household (ALFWorld).

At every step you receive:
- Observation: a description of the current situation.
- Available actions: the complete list of actions you may take right now.

Reason briefly, then choose exactly ONE action from the Available actions list.

Always reply in EXACTLY this format (two lines):
Thought: <one or two short sentences reasoning about the current state and your next subgoal>
Action: <one action copied verbatim from the Available actions list>

Rules:
- The "Action:" line MUST be the final line of your reply.
- Copy the chosen action EXACTLY as it appears in Available actions (identical words, labels and numbers).
- Choose only from the Available actions. Never invent objects, receptacles, labels or actions.
- Write nothing after the Action line.
- Track the objects you carry and the subgoals you have completed across turns.
"""

# Optional ReAct demonstration, disabled by --no-fewshot.
FEWSHOT_EXAMPLE = """Here is one short example of the interaction format (a different task):

Observation:
You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a countertop 1, a coffeemachine 1, and a sinkbasin 1.
Your task is to: put a clean mug in coffeemachine.
Available actions:
- go to cabinet 1
- go to coffeemachine 1
- go to countertop 1
- go to sinkbasin 1
- inventory
- look
Thought: To put a clean mug in the coffeemachine, I first need to find a mug. I will check the countertop.
Action: go to countertop 1

Observation:
On the countertop 1, you see a mug 1.
Available actions:
- go to coffeemachine 1
- go to sinkbasin 1
- take mug 1 from countertop 1
Thought: I found mug 1. I should pick it up so I can clean it.
Action: take mug 1 from countertop 1

(End of example. Now solve the real task below.)
"""


@dataclass
class Task:
    """One ALFWorld task corresponding to a row in alfworld/dyad_stmt."""

    index: int
    game: int
    task_id: str
    task_type: str
    ground_truth: List[str]
    query: str


def _coerce_ground_truth(gt: Any) -> List[str]:
    """Normalize ground_truth to a list of action strings.

    Accept lists and string representations of NumPy arrays from parquet previews.
    """
    if isinstance(gt, list):
        return [str(x) for x in gt]
    if isinstance(gt, str):
        s = gt.strip()
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(x) for x in v]
        except Exception:
            pass
        # Accept NumPy repr strings that omit commas between quoted elements.
        toks = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
        out = [a or b for a, b in toks]
        if out:
            return out
    return []


def load_tasks(path: Path, limit: Optional[int] = None) -> List[Task]:
    """Read ALFWorld task identities from JSONL records."""
    tasks: List[Task] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ei = rec.get("extra_info", {})
            create_kwargs = (
                ei.get("tools_kwargs", {})
                .get("alfworld_action", {})
                .get("create_kwargs", {})
            )
            game = create_kwargs.get("game")
            if game is None:
                # A game index is required for the online reset call.
                continue
            tasks.append(
                Task(
                    index=int(rec.get("index", len(tasks))),
                    game=int(game),
                    task_id=str(create_kwargs.get("task_id", "")),
                    task_type=str(create_kwargs.get("task_type", "")),
                    ground_truth=_coerce_ground_truth(
                        rec.get("reward_model", {}).get("ground_truth", [])
                    ),
                    query=str(ei.get("interaction_kwargs", {}).get("query", "")),
                )
            )
            if limit is not None and len(tasks) >= limit:
                break
    return tasks


def format_available_actions(actions: List[str]) -> str:
    return "\n".join(f"- {a}" for a in actions)


def build_system_prompt() -> str:
    """Task-level ReAct instructions are independent of native model thinking mode."""
    return SYSTEM_PROMPT


def build_first_user_prompt(observation: str, actions: List[str], use_fewshot: bool) -> str:
    """Build the first user turn from the reset observation and available actions."""
    body = (
        f"Observation:\n{observation.strip()}\n"
        f"Available actions:\n{format_available_actions(actions)}"
    )
    if use_fewshot:
        return f"{FEWSHOT_EXAMPLE}\n{body}"
    return body


def build_step_user_prompt(observation: str, actions: List[str]) -> str:
    """Build a subsequent user turn from action feedback and available actions."""
    return (
        f"Observation:\n{observation.strip()}\n"
        f"Available actions:\n{format_available_actions(actions)}"
    )


def extract_thought(response: str) -> str:
    m = re.search(r"(?i)thought\s*:\s*(.+)", response)
    if not m:
        return ""
    return m.group(1).splitlines()[0].strip()


def extract_action_text(response: str) -> str:
    """Extract action text after the last Action: marker.

    Fall back to the last nonempty line when no marker is present.
    """
    raw = ""
    line_matches = list(re.finditer(r"(?im)^\s*action\s*:\s*(.+?)\s*$", response))
    if line_matches:
        raw = line_matches[-1].group(1)
    else:
        inline = list(re.finditer(r"(?is)action\s*:\s*(.+)", response))
        if inline:
            raw = inline[-1].group(1)
        else:
            lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
            raw = lines[-1] if lines else ""

    raw = raw.strip()
    if raw:
        raw = raw.splitlines()[0].strip()
    raw = raw.strip("`*_ ").strip()
    raw = raw.strip("\"'").strip()
    raw = re.sub(r"[.\s]+$", "", raw)
    return raw


def _canonical(s: str) -> str:
    """Normalize case, whitespace, and in/on variants for ALFWorld action matching.

    The in/on preposition occurs in put actions and is normalized to one placeholder.
    """
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("in/on", "\x00")
    s = re.sub(r"\b(?:in|on)\b", "\x00", s)
    return s.strip()


def match_action(
    parsed: str,
    available: List[str],
    *,
    fuzzy: bool = False,
    fuzzy_cutoff: float = 0.82,
) -> Optional[str]:
    """Match parsed text to an available action, returning None if unmatched.

    Default matching normalizes case, whitespace, punctuation, and in/on without
    correcting object labels. fuzzy=True additionally permits containment and
    difflib matching, which may map an incorrect label to a different action.
    """
    if not parsed:
        return None

    for a in available:
        if parsed == a:
            return a
    pl = parsed.lower()
    for a in available:
        if pl == a.lower():
            return a
    pc = _canonical(parsed)
    canon_map: Dict[str, str] = {}
    for a in available:
        canon_map.setdefault(_canonical(a), a)
    if pc in canon_map:
        return canon_map[pc]

    if not fuzzy:
        # Strict matching leaves unmatched text unchanged for the environment.
        return None

    # For prose-wrapped actions, select the longest available action matching whole words.
    contained: List[Tuple[int, str]] = []
    for canon_a, orig_a in canon_map.items():
        if not canon_a:
            continue
        if re.search(r"(?:^|\s)" + re.escape(canon_a) + r"(?:$|\s)", pc):
            contained.append((len(canon_a), orig_a))
    if contained:
        contained.sort(reverse=True)
        return contained[0][1]
    best = difflib.get_close_matches(pc, list(canon_map.keys()), n=1, cutoff=fuzzy_cutoff)
    if best:
        return canon_map[best[0]]
    return None


class BaseModel:
    """Model policy interface."""

    def reset_episode(self, task: "Task") -> None:  # noqa: D401 - Default no-op policy hook.
        """Reset per-episode policy state before an interaction starts."""

    def generate(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class OpenAIChatModel(BaseModel):
    """OpenAI-compatible chat-completions client."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 180.0,
        retries: int = 3,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries

    def generate(self, messages: List[Dict[str, str]]) -> str:
        import requests

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        template_kwargs = resolve_chat_template_kwargs(model=self.model)
        if template_kwargs:
            # requests sends the JSON body directly, unlike the OpenAI SDK's extra_body.
            payload["chat_template_kwargs"] = template_kwargs
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.post(
                    self.url, json=payload, headers=headers, timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"] or ""
            except Exception as exc:  # noqa: BLE001 - Retry network and response-decoding failures.
                last_err = exc
                if attempt < self.retries:
                    time.sleep(min(2.0 * attempt, 8.0))
        raise RuntimeError(f"model request failed after {self.retries} tries: {last_err!r}")




class AlfWorldHTTPClient:
    """HTTP client for the ALFWorld FastAPI server."""

    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, body: Optional[dict] = None) -> Any:
        import requests

        resp = requests.post(f"{self.base}{path}", json=body, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def create(self) -> int:
        out = self._post("/create")
        if isinstance(out, dict) and "error" in out:
            raise RuntimeError(f"create error: {out}")
        return int(out["id"])

    def reset(self, env_id: int, game: int, world_type: str = "Text") -> Dict[str, Any]:
        out = self._post(
            "/reset", {"id": env_id, "game": game, "world_type": world_type}
        )
        if isinstance(out, dict) and "error" in out:
            raise RuntimeError(f"reset error: {out}")
        return out

    def step(self, env_id: int, action: str) -> Dict[str, Any]:
        out = self._post("/step", {"id": env_id, "action": action})
        if isinstance(out, dict) and "error" in out:
            raise RuntimeError(f"step error: {out}")
        return out

    def close(self, env_id: int) -> Any:
        try:
            return self._post("/close", {"id": env_id})
        except Exception:
            return None




def _trim_messages(
    messages: List[Dict[str, str]], window: int
) -> List[Dict[str, str]]:
    """Keep the system message, first user turn, and the most recent window of turns."""
    if window is None or window <= 0:
        return messages
    head = messages[:2]  # System message and initial user turn.
    tail = messages[2:]
    keep = window * 2
    if len(tail) <= keep:
        return messages
    return head + tail[-keep:]


def run_episode(
    env: Any,
    model: BaseModel,
    task: Task,
    *,
    max_steps: int,
    world_type: str,
    use_fewshot: bool,
    history_window: int,
    fuzzy: bool,
    verbose: bool,
) -> Dict[str, Any]:
    """Run a complete ReAct interaction for one task."""
    model.reset_episode(task)

    result: Dict[str, Any] = {
        "index": task.index,
        "game": task.game,
        "task_id": task.task_id,
        "task_type": task.task_type,
        "won": False,
        "reward": 0.0,
        "steps": 0,
        "invalid_actions": 0,
        "error": None,
        "transcript": [],
    }

    env_id: Optional[int] = None
    try:
        env_id = env.create()
        reset_out = env.reset(env_id, task.game, world_type)
        observation = reset_out["observation"]
        available = list(reset_out.get("available_actions", []))

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": build_system_prompt()},
            {
                "role": "user",
                "content": build_first_user_prompt(observation, available, use_fewshot),
            },
        ]

        for step_idx in range(1, max_steps + 1):
            response = model.generate(_trim_messages(messages, history_window))
            messages.append({"role": "assistant", "content": response})

            thought = extract_thought(response)
            parsed = extract_action_text(response)
            matched = match_action(parsed, available, fuzzy=fuzzy)
            action_to_send = matched if matched is not None else parsed
            if matched is None:
                result["invalid_actions"] += 1

            step_out = env.step(env_id, action_to_send)
            observation = step_out.get("observation", "")
            available = list(step_out.get("available_actions", []))
            reward = float(step_out.get("reward", 0.0))
            done = bool(step_out.get("done", False))

            result["transcript"].append(
                {
                    "step": step_idx,
                    "thought": thought,
                    "parsed_action": parsed,
                    "matched_action": matched,
                    "sent_action": action_to_send,
                    "observation": observation,
                    "reward": reward,
                    "done": done,
                }
            )
            if verbose:
                tag = "OK " if matched is not None else "RAW"
                print(
                    f"    [step {step_idx:>2}] ({tag}) {action_to_send!r} -> "
                    f"reward={reward} done={done}"
                )

            result["steps"] = step_idx
            if reward >= 1.0:
                result["won"] = True
                result["reward"] = reward
            if done:
                break

            messages.append(
                {"role": "user", "content": build_step_user_prompt(observation, available)}
            )

        result["reward"] = max(result["reward"], 0.0)
    except Exception as exc:  # noqa: BLE001 - Preserve other task results on failure.
        result["error"] = repr(exc)
    finally:
        if env_id is not None:
            try:
                env.close(env_id)
            except Exception:
                pass
    return result


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    won = sum(1 for r in results if r["won"])
    errors = sum(1 for r in results if r["error"])
    steps_won = [r["steps"] for r in results if r["won"]]

    by_type: Dict[str, Dict[str, int]] = {}
    for r in results:
        bucket = by_type.setdefault(r["task_type"], {"total": 0, "won": 0})
        bucket["total"] += 1
        bucket["won"] += 1 if r["won"] else 0

    return {
        "total": total,
        "won": won,
        "success_rate": (won / total) if total else 0.0,
        "errors": errors,
        "avg_steps_success": (sum(steps_won) / len(steps_won)) if steps_won else 0.0,
        "by_task_type": {
            k: {
                **v,
                "success_rate": (v["won"] / v["total"]) if v["total"] else 0.0,
            }
            for k, v in sorted(by_type.items())
        },
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("ALFWorld ReAct baseline 结果")
    print("=" * 60)
    print(f"任务总数      : {summary['total']}")
    print(f"成功数        : {summary['won']}")
    print(f"成功率        : {summary['success_rate'] * 100:.2f}%")
    print(f"报错任务数    : {summary['errors']}")
    print(f"成功任务平均步数: {summary['avg_steps_success']:.2f}")
    print("-" * 60)
    print("按 task_type 细分:")
    for task_type, stats in summary["by_task_type"].items():
        print(
            f"  {task_type:<28} {stats['won']:>3}/{stats['total']:<3} "
            f"({stats['success_rate'] * 100:5.1f}%)"
        )
    print("=" * 60)


def find_repo_root() -> Path:
    """Find the repository by pyproject.toml and agent_system/ without assuming directory depth."""
    for d in Path(__file__).resolve().parents:
        if (d / "pyproject.toml").is_file() and (d / "agent_system").is_dir():
            return d
    raise RuntimeError(
        f"从 {Path(__file__).resolve()} 向上找不到 Dynamic_ExpA 根目录"
        "（标记：同级存在 pyproject.toml 和 agent_system/）。请用 --data 显式给出 jsonl 路径。"
    )


# Match the dataset variant name in experiments/shared/dataset/alfworld.py.
# Use task metadata only; prompts are constructed for this baseline.
DATASET_VARIANT = "dyad_stmt"


def default_data_path(split: str) -> Path:
    data_root = find_repo_root() / "data" / "alfworld"
    path = data_root / DATASET_VARIANT / f"{split}.jsonl"
    if path.exists():
        return path
    # Include available paths in the error to diagnose moved datasets.
    available = sorted(d.name for d in data_root.iterdir() if d.is_dir()) if data_root.is_dir() else []
    raise FileNotFoundError(
        f"找不到 {path}\n"
        f"{data_root} 下现有：{available or '(目录不存在)'}\n"
        f"数据集目录名改过就更新 DATASET_VARIANT，或用 --data 显式指定 jsonl 路径。"
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ALFWorld ReAct baseline（在线交互 + 正则匹配）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--split", choices=["test", "train"], default="test", help="使用哪个划分")
    p.add_argument("--data", type=str, default=None, help="jsonl 路径，覆盖 --split 默认值")
    p.add_argument("--limit", type=int, default=None, help="只评测前 N 个任务，默认全部")
    p.add_argument(
        "--env-server", type=str, default="http://127.0.0.1:36001", help="ALFWorld 服务端地址"
    )
    p.add_argument(
        "--world-type", choices=["Text", "Embody", "Hybrid"], default="Text", help="环境类型"
    )
    p.add_argument(
        "--model-base-url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="OpenAI 兼容接口 base url（vLLM 默认 /v1）",
    )
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="模型名")
    p.add_argument("--api-key", type=str, default="EMPTY", help="API key（vLLM 任意）")
    p.add_argument("--temperature", type=float, default=0.0, help="采样温度")
    p.add_argument("--max-tokens", type=int, default=512, help="单步生成上限")
    p.add_argument("--max-steps", type=int, default=40, help="单任务最大交互步数")
    p.add_argument(
        "--history-window",
        type=int,
        default=0,
        help="对话历史截断窗口（0 = 保留全部历史）",
    )
    p.add_argument("--no-fewshot", action="store_true", help="不在首轮加入 in-context 示例")
    p.add_argument(
        "--fuzzy-match",
        action="store_true",
        help="开启宽松动作匹配（包含+difflib）；默认严格匹配更忠实地衡量能力",
    )
    p.add_argument("--output", type=str, default=None, help="结果 json 输出路径")
    p.add_argument("--verbose", action="store_true", help="打印每一步交互")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)


    data_path = Path(args.data) if args.data else default_data_path(args.split)
    if not data_path.exists():
        print(f"[ERROR] 找不到数据文件: {data_path}", file=sys.stderr)
        return 2
    tasks = load_tasks(data_path, limit=args.limit)
    if not tasks:
        print(f"[ERROR] 未从 {data_path} 读到任何任务", file=sys.stderr)
        return 2
    print(f"[INFO] 数据: {data_path}")
    print(f"[INFO] 任务数: {len(tasks)}（split={args.split}）")

    env = AlfWorldHTTPClient(args.env_server)
    print(f"[INFO] 环境: {args.env_server}")

    model = OpenAIChatModel(
        base_url=args.model_base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(f"[INFO] 模型: {args.model} @ {args.model_base_url}")


    use_fewshot = not args.no_fewshot
    results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, task in enumerate(tasks, 1):
        res = run_episode(
            env,
            model,
            task,
            max_steps=args.max_steps,
            world_type=args.world_type,
            use_fewshot=use_fewshot,
            history_window=args.history_window,
            fuzzy=args.fuzzy_match,
            verbose=args.verbose,
        )
        results.append(res)
        status = "WON " if res["won"] else "lose"
        if res["error"]:
            status = "ERR "
        print(
            f"[{i:>3}/{len(tasks)}] game={task.game:<5} {task.task_type:<26} "
            f"{status} steps={res['steps']:<2} invalid={res['invalid_actions']}"
            + (f"  err={res['error']}" if res["error"] else "")
        )
    elapsed = time.perf_counter() - t0

    summary = summarize(results)
    summary["elapsed_sec"] = round(elapsed, 2)
    summary["model"] = args.model
    summary["split"] = args.split
    print_summary(summary)

    out_path = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent
        / "outputs"
        / f"react_{args.split}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "results": results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[INFO] 结果已保存: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
