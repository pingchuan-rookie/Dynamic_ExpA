"""Evaluate ALFWorld using the official ReAct prompt and task protocol.

Use valid_unseen, official two-shot examples, and task-type prefix routing from
https://github.com/ysymyth/ReAct. Send the next action line to the environment
without candidate correction; think: turns receive the observation "OK.".
Report overall and per-task-type success within max_steps.

The API adaptation places examples and history in one user message and requests
the next action. Endpoint and credentials come from explicit arguments or the
configured environment; no credential-file path is assumed.

The reference HTTP server must load unseen games with ALFWORLD_INCLUDE_UNSEEN=1.
See the adjacent README.md for usage and protocol details.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Keep standalone environments independent of the training dependencies.
_PROJECT = Path(__file__).resolve().parents[6]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))
from agent_system.utils.thinking import is_qwen35_model, resolve_chat_template_kwargs

HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = HERE / "prompts" / "alfworld_3prompts.json"


def find_repo_root() -> Path:
    """Find the repository by pyproject.toml and agent_system/ without assuming directory depth."""
    for d in HERE.parents:
        if (d / "pyproject.toml").is_file() and (d / "agent_system").is_dir():
            return d
    raise RuntimeError(
        f"从 {HERE} 向上找不到 Dynamic_ExpA 根目录（标记：同级存在 pyproject.toml 和 agent_system/）。"
        "请用 ALFWORLD_CONFIGS_DIR 指向 server 的 configs/ 目录。"
    )


def configs_dir() -> Path:
    """Return the directory containing train, test, and unseen mappings.

    Evaluation and the server must read the same mappings: global unseen indices
    start after the train and test lists. ALFWORLD_CONFIGS_DIR overrides the default.
    """
    override = os.environ.get("ALFWORLD_CONFIGS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return find_repo_root() / "experiments/shared/train_eval/reference/alfworld/alfworld_server/configs"


# Official task-prefix order maps gamefile directories to react_<type>_0/1 prompts.
OFFICIAL_PREFIXES: "OrderedDict[str, str]" = OrderedDict(
    [
        ("pick_and_place", "put"),
        ("pick_clean_then_place_in_recep", "clean"),
        ("pick_heat_then_place_in_recep", "heat"),
        ("pick_cool_then_place_in_recep", "cool"),
        ("look_at_obj_in_light", "examine"),
        ("pick_two_obj_and_place", "puttwo"),
    ]
)

PROMPT_HEADER = "Interact with a household to solve a task. Here are two examples.\n"
PROMPT_TASK_SEP = "\nHere is the task.\n"

# Official-style chat adaptation without candidate actions.
DEFAULT_SYSTEM_PROMPT = (
    "You are an agent solving tasks in a text-based household environment (ALFWorld). "
    "You are shown two example interactions, then a new task to solve. "
    "Continue the interaction by producing the NEXT single line only: either exactly one "
    "admissible action (e.g. 'go to cabinet 1', 'open fridge 1', "
    "'take mug 1 from countertop 1', 'put mug 1 in/on coffeemachine 1', 'look', 'inventory'), "
    "or a reasoning line starting with 'think:'. "
    "Output exactly one line and nothing else. Do not repeat the leading '>' prompt."
)

# Optional candidate-constrained prompt.
ADMISSIBLE_SYSTEM_PROMPT = (
    "You are an agent solving tasks in a text-based household environment (ALFWorld). "
    "You are shown two example interactions, then a new task to solve. "
    "For the task to solve, after each observation you are given a line "
    "'Admissible actions: [...]' that lists every valid action in the current state "
    "(the two examples do NOT include this line). "
    "Continue the interaction by producing the NEXT single line only: either exactly one "
    "action copied VERBATIM from the current 'Admissible actions' list, "
    "or a reasoning line starting with 'think:'. "
    "Output exactly one line and nothing else. Do not repeat the leading '>' prompt."
)

ADMISSIBLE_PREFIX = "Admissible actions: "


def format_admissible(actions: Optional[List[str]]) -> str:
    """Render candidate actions on one conversation line, or return an empty string."""
    if not actions:
        return ""
    return ADMISSIBLE_PREFIX + "[" + ", ".join(str(a) for a in actions) + "]"



def load_official_prompts(path: Path) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_two_shot_prompt(prompts: Dict[str, str], short: str) -> str:
    """Preserve official reasoning steps independently of native model thinking mode."""
    examples = prompts[f"react_{short}_1"] + prompts[f"react_{short}_0"]
    return PROMPT_HEADER + examples + PROMPT_TASK_SEP


def detect_prefix(task_type_name: str) -> Optional[str]:
    """Map the official gamefile prefix to a prompt task type such as put or clean."""
    for prefix, short in OFFICIAL_PREFIXES.items():
        if task_type_name.startswith(prefix):
            return short
    return None


def parse_action(raw: str) -> str:
    """Extract the first nonempty action line from a model response.

    Remove an echoed prompt marker and Markdown backticks, preserving think:
    prefixes, case, and punctuation for the environment.
    """
    if not raw:
        return ""
    line = ""
    for ln in raw.splitlines():
        s = ln.strip()
        if s:
            line = s
            break
    else:
        line = raw.strip()
    line = line.lstrip(">").strip()
    if line.startswith("`") or line.endswith("`"):
        line = line.strip("`").strip()
    return line


def load_env_file(env_file: Optional[Path]) -> None:
    """Load KEY=VALUE entries without overwriting existing environment variables.

    Use python-dotenv when available and the local parser otherwise.
    """
    if env_file is None or not Path(env_file).exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=str(env_file), override=False)
        return
    except Exception:
        pass
    with open(env_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


class BaseModel:
    def generate(self, system_prompt: str, user_content: str) -> str:
        raise NotImplementedError


class _RateLimiter:
    """Space requests within this process by at least min_interval seconds."""

    def __init__(self, min_interval: float = 0.0) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._last_ts = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        dt = time.time() - self._last_ts
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last_ts = time.time()


class OpenAIChatModel(BaseModel):
    """OpenAI-compatible chat client with parameter compatibility retries.

    On explicit unsupported-parameter errors, remove the rejected option or switch
    max_tokens to max_completion_tokens before retrying.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None,
        timeout: float = 180.0,
        retries: int = 4,
        min_interval: float = 0.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature: Optional[float] = temperature
        self.max_tokens: Optional[int] = max_tokens
        self.max_tokens_param = "max_tokens"
        self.stop = list(stop) if stop else None
        self.timeout = timeout
        self.retries = retries
        self._client = None
        self._limiter = _RateLimiter(min_interval)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
            )
        return self._client

    def _build_kwargs(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs[self.max_tokens_param] = self.max_tokens
        if self.stop:
            kwargs["stop"] = self.stop
        template_kwargs = resolve_chat_template_kwargs(model=self.model)
        if template_kwargs:
            kwargs["extra_body"] = {"chat_template_kwargs": template_kwargs}
        return kwargs

    def _adapt_on_error(self, msg: str) -> bool:
        """Adapt unsupported request parameters; return True when a retry is possible."""
        low = msg.lower()
        if "temperature" in low and self.temperature is not None:
            self.temperature = None
            return True
        if ("max_tokens" in low or "max_completion_tokens" in low) and self.max_tokens_param == "max_tokens":
            self.max_tokens_param = "max_completion_tokens"
            return True
        if "stop" in low and self.stop:
            self.stop = None
            return True
        return False

    def generate(self, system_prompt: str, user_content: str) -> str:
        client = self._get_client()
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                self._limiter.wait()
                resp = client.chat.completions.create(**self._build_kwargs(messages))
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if self._adapt_on_error(str(exc)):
                    continue  # Retry corrected parameters immediately without network backoff.
                if attempt < self.retries:
                    time.sleep(min(2.0 * attempt, 8.0))
        raise RuntimeError(
            f"model request failed after {self.retries} tries: {last_err!r}"
        )


class OpenAIResponsesModel(BaseModel):
    """Responses API client for models served through that endpoint.

    Pass system text as instructions and interaction history as input. The output
    budget includes reasoning tokens. Omit temperature and stop for compatibility.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        max_output_tokens: int = 2048,
        timeout: float = 180.0,
        retries: int = 4,
        min_interval: float = 0.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        if is_qwen35_model(model):
            raise ValueError("Qwen3.5 no-thinking requires --api chat, not Responses API")
        self.max_output_tokens: Optional[int] = max_output_tokens
        self.timeout = timeout
        self.retries = retries
        self._client = None
        self._limiter = _RateLimiter(min_interval)

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
            )
        return self._client

    @staticmethod
    def _extract_text(resp: Any) -> str:
        text = getattr(resp, "output_text", None)
        if text:
            return str(text).strip()
        parts: List[str] = []
        for item in getattr(resp, "output", None) or []:
            if getattr(item, "type", None) != "message":
                continue
            for c in getattr(item, "content", None) or []:
                t = getattr(c, "text", None)
                if t:
                    parts.append(str(t))
        return "".join(parts).strip()

    def generate(self, system_prompt: str, user_content: str) -> str:
        client = self._get_client()
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                self._limiter.wait()
                kwargs: Dict[str, Any] = {"model": self.model, "input": user_content}
                if system_prompt:
                    kwargs["instructions"] = system_prompt
                if self.max_output_tokens is not None:
                    kwargs["max_output_tokens"] = self.max_output_tokens
                resp = client.responses.create(**kwargs)
                return self._extract_text(resp)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.retries:
                    time.sleep(min(2.0 * attempt, 8.0))
        raise RuntimeError(
            f"responses request failed after {self.retries} tries: {last_err!r}"
        )




class AlfWorldHTTPClient:
    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, body: Optional[dict] = None) -> Any:
        import requests

        resp = requests.post(f"{self.base}{path}", json=body, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str) -> Any:
        import requests

        resp = requests.get(f"{self.base}{path}", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def stats(self) -> Dict[str, Any]:
        return self._get("/stats")

    def create(self) -> int:
        out = self._post("/create")
        if isinstance(out, dict) and "error" in out:
            raise RuntimeError(f"create error: {out}")
        return int(out["id"])

    def reset(self, env_id: int, game: int, world_type: str = "Text") -> Dict[str, Any]:
        out = self._post("/reset", {"id": env_id, "game": game, "world_type": world_type})
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




@dataclass
class Task:
    index: int  # Zero-based index within the unseen mapping.
    game_index: int  # Global server index: unseen_start + index.
    task_type: str  # Mapping task directory, used for logging and fallback routing.


def load_unseen_tasks(
    configs_dir: Path, unseen_start_override: Optional[int] = None
) -> Tuple[List[Task], int]:
    """Compute the global unseen-game index range from the three server mappings."""
    train_path = configs_dir / "mappings_train.json"
    test_path = configs_dir / "mappings_test.json"
    unseen_path = configs_dir / "mappings_unseen.json"

    if not unseen_path.exists():
        raise FileNotFoundError(
            f"missing {unseen_path}\n请先运行：python3 make_unseen_mappings.py"
        )

    def _len(p: Path) -> int:
        if not p.exists():
            return 0
        with open(p, "r", encoding="utf-8") as f:
            return len(json.load(f))

    with open(unseen_path, "r", encoding="utf-8") as f:
        unseen = json.load(f)

    if unseen_start_override is not None:
        start = unseen_start_override
    else:
        start = _len(train_path) + _len(test_path)

    tasks = [
        Task(index=i, game_index=start + i, task_type=str(m.get("task_type", "")))
        for i, m in enumerate(unseen)
    ]
    return tasks, start


def run_episode(
    env: Any,
    model: BaseModel,
    task: Task,
    prompts: Dict[str, str],
    *,
    max_steps: int,
    system_prompt: str,
    verbose: bool,
    show_admissible: bool = True,
    record_raw: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "index": task.index,
        "game_index": task.game_index,
        "task_type": task.task_type,
        "prefix": None,
        "loaded": False,
        "won": False,
        "reward": 0.0,
        "steps": 0,
        "error": None,
        "transcript": [],
    }
    if record_raw:
        result["system_prompt"] = system_prompt
        result["reset_reply"] = None
        result["initial_prompt"] = None

    env_id: Optional[int] = None
    try:
        env_id = env.create()
        reset_out = env.reset(env_id, task.game_index, "Text")
        result["loaded"] = True
        task_type_name = str(reset_out.get("task_type", task.task_type))
        reset_ob = str(reset_out.get("observation", ""))
        if record_raw:
            result["reset_reply"] = reset_out

        short = detect_prefix(task_type_name)
        if short is None:
            result["error"] = f"unknown task type prefix: {task_type_name!r}"
            return result
        result["prefix"] = short


        two_shot = build_two_shot_prompt(prompts, short)
        reset_block = reset_ob
        if show_admissible:
            adm_line = format_admissible(reset_out.get("available_actions"))
            if adm_line:
                reset_block = f"{reset_ob}\n{adm_line}"
        init_prompt = two_shot + reset_block + "\n>"
        running = ""
        if record_raw:
            result["initial_prompt"] = init_prompt

        if verbose:
            print(f"\n[game {task.index} idx={task.game_index} type={short}]")
            print(f"  OBS: {reset_ob.splitlines()[-1] if reset_ob else ''}")

        for step_idx in range(1, max_steps + 1):
            prompt_sent = init_prompt + running
            raw = model.generate(system_prompt, prompt_sent)
            action = parse_action(raw)

            step_out = env.step(env_id, action)
            observation = str(step_out.get("observation", ""))
            reward = float(step_out.get("reward", 0.0))
            done = bool(step_out.get("done", False))
            available = step_out.get("available_actions")

            is_think = action.startswith("think:")
            if is_think:
                observation = "OK."

            obs_block = observation
            if show_admissible:
                adm_line = format_admissible(available)
                if adm_line:
                    obs_block = f"{observation}\n{adm_line}"
            running += f" {action}\n{obs_block}\n>"
            result["steps"] = step_idx
            entry: Dict[str, Any] = {
                "step": step_idx,
                "action": action,
                "is_think": is_think,
                "observation": observation,
                "reward": reward,
                "done": done,
            }
            if show_admissible:
                entry["admissible"] = available
            if record_raw:
                entry["prompt"] = prompt_sent
                entry["raw_response"] = raw
                entry["env_reply"] = step_out
            result["transcript"].append(entry)
            if verbose:
                print(f"  Act {step_idx:>2}: {action}\n           Obs: {observation}")

            if reward >= 1.0:
                result["won"] = True
                result["reward"] = reward
            if done:
                break

    except Exception as exc:  # noqa: BLE001 - Preserve other episode results on failure.
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
    loaded = sum(1 for r in results if r.get("loaded"))
    load_errors = sum(1 for r in results if not r.get("loaded"))
    errors = sum(1 for r in results if r["error"])
    steps_won = [r["steps"] for r in results if r["won"]]

    order = list(OFFICIAL_PREFIXES.values())
    by_type: "OrderedDict[str, Dict[str, int]]" = OrderedDict(
        (k, {"total": 0, "won": 0}) for k in order
    )
    for r in results:
        short = r.get("prefix")
        if short not in by_type:
            by_type[short if short else "unknown"] = {"total": 0, "won": 0}
        bucket = by_type[short if short else "unknown"]
        bucket["total"] += 1
        bucket["won"] += 1 if r["won"] else 0

    return {
        "total": total,
        "loaded": loaded,
        "load_errors": load_errors,
        "won": won,
        # Compute success over loaded episodes; report load errors separately.
        "success_rate": (won / loaded) if loaded else 0.0,
        "success_rate_over_total": (won / total) if total else 0.0,
        "errors": errors,
        "avg_steps_success": (sum(steps_won) / len(steps_won)) if steps_won else 0.0,
        "by_task_type": {
            k: {**v, "success_rate": (v["won"] / v["total"]) if v["total"] else 0.0}
            for k, v in by_type.items()
        },
    }


def print_summary(summary: Dict[str, Any], model_name: str) -> None:
    print("\n" + "=" * 62)
    print("ALFWorld 官方 ReAct 评测结果")
    print("=" * 62)
    print(f"模型            : {model_name}")
    print(f"任务总数(OOD)   : {summary['total']}")
    print(f"可加载局数      : {summary['loaded']}（坏局/load error: {summary['load_errors']}）")
    print(f"成功数          : {summary['won']}")
    print(f"成功率(按可加载): {summary['success_rate'] * 100:.2f}%")
    print(f"成功率(按总数)  : {summary['success_rate_over_total'] * 100:.2f}%")
    print(f"报错局数        : {summary['errors']}")
    print(f"成功局平均步数  : {summary['avg_steps_success']:.2f}")
    print("-" * 62)
    print("按官方 6 类任务分桶:")
    for short, stats in summary["by_task_type"].items():
        if stats["total"] == 0:
            continue
        print(
            f"  {short:<10} {stats['won']:>3}/{stats['total']:<3} "
            f"({stats['success_rate'] * 100:5.1f}%)"
        )
    print("=" * 62)


def write_readable_transcript(
    results: List[Dict[str, Any]], path: Path, model_name: str
) -> None:
    """Write conversation fragments in their original append order.

    Separate appends with star lines. Removing those separators reconstructs the
    initial prompt followed by each action and observation.
    """
    SEP = "*" * 60

    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            if len(results) > 1:
                f.write(
                    f"======== EPISODE {r['index']} game={r['game_index']} "
                    f"{r.get('prefix')} won={r['won']} steps={r['steps']} ========\n"
                )
            sysp = r.get("system_prompt") or ""
            initp = r.get("initial_prompt") or ""
            f.write(sysp)
            if sysp and initp:
                f.write("\n\n")
            f.write(initp)
            for t in r.get("transcript", []):
                f.write("\n" + SEP + "\n")
                f.write(f" {t['action']}")
                f.write("\n" + SEP + "\n")
                obs_block = t["observation"]
                if t.get("admissible"):
                    adm_line = format_admissible(t["admissible"])
                    if adm_line:
                        obs_block = f"{t['observation']}\n{adm_line}"
                f.write(f"{obs_block}\n>")
            f.write("\n\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ALFWorld 官方 ReAct 评测（API 模型，默认 GPT-5.5）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--env-server", default="http://127.0.0.1:36001", help="ALFWorld HTTP server 地址")
    p.add_argument("--prompts", default=str(DEFAULT_PROMPT_FILE), help="官方 alfworld_3prompts.json 路径")
    p.add_argument("--limit", type=int, default=None, help="只评测前 N 局（官方 134，可用来对齐官方数量）")
    p.add_argument("--max-steps", type=int, default=50, help="单局最大步数（官方 ReAct 循环上限 49）")
    p.add_argument(
        "--unseen-start", type=int, default=None,
        help="OOD 游戏在 server 里的起始全局索引（默认由 mappings_train+test 长度推算 / 或从 /stats 读）",
    )
    p.add_argument("--no-verify-server", action="store_true", help="跳过启动前 /stats 校验（不检查 unseen 是否加载）")
    p.add_argument("--model", default=os.environ.get("REACT_EVAL_MODEL", "openai/gpt-5.5"), help="模型名")
    p.add_argument(
        "--api", choices=["chat", "responses"], default="chat",
        help="调用哪种 OpenAI 接口：chat=/chat/completions；responses=/responses（gpt-5.x 推理模型需用此项）",
    )
    p.add_argument("--base-url", default=None, help="OpenAI 兼容 base_url（默认取 GITHUB_MODELS_BASE_URL）")
    p.add_argument("--api-key-env", default="GITHUB_TOKEN", help="从哪个环境变量读 api_key")
    p.add_argument("--env-file", default=None, help="加载的 .env 文件，不给则只读环境变量")
    p.add_argument("--temperature", type=float, default=0.0, help="采样温度（官方贪心 = 0.0；responses 推理模型忽略）")
    p.add_argument("--max-tokens", type=int, default=256, help="chat 接口单次回复最大 token")
    p.add_argument("--max-output-tokens", type=int, default=2048, help="responses 接口单次输出最大 token（含推理，建议给足）")
    p.add_argument(
        "--request-interval", type=float, default=0.0,
        help="相邻两次模型调用最小间隔秒（走本地 copilot-api 代理时务必设 >0，如 3，防触发 GitHub 滥用检测）",
    )
    p.add_argument(
        "--admissible", action=argparse.BooleanOptionalAction, default=True,
        help="把当前可执行动作(admissible actions)附在每步观测后喂给模型；--no-admissible 回到官方严格模式(不给候选)",
    )
    p.add_argument(
        "--system-prompt", default=None,
        help="chat 适配用的 system 指令（默认按是否给 admissible 自动选：给候选用 ADMISSIBLE 版，否则官方版）",
    )
    p.add_argument("--verbose", action="store_true", help="打印每步交互")
    p.add_argument(
        "--record-raw", action="store_true",
        help="记录最原始数据（每步发给模型的完整 prompt / 模型未解析原始回复 / env 完整回复），并额外写一份可读 .txt",
    )
    p.add_argument("--output", default=None, help="结果 json 路径（默认写 outputs/）")
    return p.parse_args(argv)




def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    system_prompt = args.system_prompt
    if system_prompt is None:
        system_prompt = ADMISSIBLE_SYSTEM_PROMPT if args.admissible else DEFAULT_SYSTEM_PROMPT

    if is_qwen35_model(args.model):
        if args.api != "chat":
            raise ValueError("Qwen3.5 no-thinking requires --api chat, not Responses API")

    prompts = load_official_prompts(Path(args.prompts))
    tasks, computed_start = load_unseen_tasks(configs_dir(), args.unseen_start)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        print("[ERR] 没有可评测的 OOD 任务（mappings_unseen.json 为空？）", file=sys.stderr)
        return 2

    load_env_file(Path(args.env_file) if args.env_file else None)
    base_url = args.base_url or os.environ.get(
        "GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference"
    )
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(
            f"[ERR] 未从环境变量 {args.api_key_env} 读到 api_key。"
            f"请在 {args.env_file} 里配置，或 export {args.api_key_env}=...",
            file=sys.stderr,
        )
        return 2
    env = AlfWorldHTTPClient(args.env_server)
    if args.api == "responses":
        model = OpenAIResponsesModel(
            base_url=base_url,
            model=args.model,
            api_key=api_key,
            max_output_tokens=args.max_output_tokens,
            min_interval=args.request_interval,
        )
    else:
        model = OpenAIChatModel(
            base_url=base_url,
            model=args.model,
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stop=["\n"],
            min_interval=args.request_interval,
        )
    model_name = args.model

    if not args.no_verify_server:
        try:
            st = env.stats()
            num_unseen = int(st.get("num_unseen_games", 0))
            srv_start = st.get("unseen_start_index")
            if num_unseen <= 0:
                print(
                    "[ERR] server 未加载 valid_unseen 游戏。请以 "
                    "`ALFWORLD_INCLUDE_UNSEEN=1` 重启 server（见 README）。",
                    file=sys.stderr,
                )
                return 2
            if args.unseen_start is None and isinstance(srv_start, int):
                # Use the server-reported offset if it differs from locally loaded mappings.
                if srv_start != computed_start:
                    print(
                        f"[WARN] 本地推算 unseen_start={computed_start} 与 server 上报 "
                        f"{srv_start} 不一致，改用 server 值。"
                    )
                    for t in tasks:
                        t.game_index = srv_start + t.index
            if num_unseen < len(tasks):
                print(
                    f"[WARN] server 只加载了 {num_unseen} 局 unseen，"
                    f"但请求评测 {len(tasks)} 局；将按 server 数量截断。"
                )
                tasks = tasks[:num_unseen]
        except Exception as exc:  # noqa: BLE001
            print(
                f"[ERR] 无法连接 server /stats：{exc!r}\n"
                f"请确认 server 已在 {args.env_server} 启动。",
                file=sys.stderr,
            )
            return 2

    print(
        f"[INFO] 模型={model_name} | OOD 局数={len(tasks)} | max_steps={args.max_steps} | "
        f"env={args.env_server}"
    )
    t0 = time.time()
    results: List[Dict[str, Any]] = []
    for i, task in enumerate(tasks, 1):
        res = run_episode(
            env, model, task, prompts,
            max_steps=args.max_steps,
            system_prompt=system_prompt,
            verbose=args.verbose,
            show_admissible=args.admissible,
            record_raw=args.record_raw,
        )
        results.append(res)
        flag = "WON " if res["won"] else ("ERR " if res["error"] else "lose")
        print(
            f"[{i:>3}/{len(tasks)}] {flag} idx={task.game_index} "
            f"type={res['prefix']} steps={res['steps']}"
            + (f" err={res['error']}" if res["error"] else "")
        )

    summary = summarize(results)
    summary["model"] = model_name
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    print_summary(summary, model_name)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        Path(args.output)
        if args.output
        else HERE / "outputs" / f"react_official_unseen_{ts}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "config": {
                    "model": model_name,
                    "api": args.api,
                    "admissible": args.admissible,
                    "env_server": args.env_server,
                    "max_steps": args.max_steps,
                    "temperature": args.temperature,
                    "request_interval": args.request_interval,
                    "limit": args.limit,
                    "prompts": str(args.prompts),
                },
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[INFO] 结果已保存: {out_path}")
    if args.record_raw:
        txt_path = out_path.with_suffix(".txt")
        write_readable_transcript(results, txt_path, model_name)
        print(f"[INFO] 可读原始记录已保存: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
