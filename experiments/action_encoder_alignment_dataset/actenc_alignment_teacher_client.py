"""The strong LLM behind Prompts A, B and C, and the retry policy around it.

Reached over the OpenAI chat-completions API, which is what the local gateway speaks. Only the
standard library is used: this module runs from the dataset builder and from the gate, neither of
which should need a http client installed to answer "is the shipped dataset well formed".

Three things here are not incidental.

**Validation is the caller's, and a failure is a retry, not a crash.** Every prompt in the strategy
document ends with a list of requirements, and a model violates one every few hundred calls. The
document says so for Prompt C -- "if validation fails, regenerate only this case_id's context and
reasoning" -- and the same policy applies to A and B. So `generate` takes a validator and re-samples
until it passes.

**Retries raise the temperature.** Re-sampling at the same temperature after a validation failure
is most of the way to an infinite loop when the failure is systematic for that input: the model has
a mode, and the mode is what just got rejected. The prompt is never touched, because the prompt is
the artefact under version control and a retry that edits it is a different experiment.

**Responses are cached on disk, keyed by attempt.** The pipeline is re-run whenever anything
downstream changes, and re-paying for 1400 identical calls to find out that the builder had a typo
is the kind of cost that makes people stop re-running it. Keying on the attempt index keeps retries
distinct, so a cached run replays exactly the sequence that happened.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TypeVar

T = TypeVar("T")


class ValidationError(ValueError):
    """A response that reached us intact but broke one of the prompt's stated requirements.

    `payload` and `penalty` exist for the one requirement that cannot be met every time. Prompt C
    forbids naming an action outside the case's catalogue, and half of these action names are
    ordinary English verbs -- `send`, `lift`, `notify`. When the natural way to describe an
    intention uses one of them, re-sampling does not help: measured over six attempts at
    temperature 1.0, the same word came back every time.

    A validator can therefore say "this response is usable but violates requirement 6 in these two
    places", carry the value in `payload`, and rank the attempt by `penalty`. `generate(...,
    best_effort=True)` then keeps the least-bad attempt instead of failing the case, and the
    violation travels with the record so the gate can hold the *rate* to a ceiling. A hard zero
    would be a threshold nobody can meet; an unmeasured fallback would be one nobody can see.
    """

    def __init__(self, message: str, payload: Any = None, penalty: int = 0):
        super().__init__(message)
        self.payload = payload
        self.penalty = penalty



class StrongLlmError(RuntimeError):
    """Every attempt failed. Carries the last error so the caller can print something useful."""


@dataclass(frozen=True)
class StrongLlmConfig:
    model: str
    base_url: str
    api_key_env: str = "ANTHROPIC_AUTH_TOKEN"
    concurrency: int = 8
    max_attempts: int = 6
    timeout_s: int = 180
    max_tokens: int = 8192

    @classmethod
    def from_generation(cls, generation: dict[str, Any]) -> "StrongLlmConfig":
        raw = dict(generation.get("strong_llm") or {})
        raw.pop("temperature", None)
        return cls(**raw)


def temperature_for(generation: dict[str, Any], letter: str) -> float:
    table = (generation.get("strong_llm") or {}).get("temperature") or {}
    return float(table.get(f"prompt_{letter}", 0.7))


class StrongLlm:
    def __init__(self, config: StrongLlmConfig, cache_dir: Optional[Path] = None):
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._api_key = os.environ.get(config.api_key_env, "")
        if not self._api_key:
            raise RuntimeError(
                f"{config.api_key_env} is empty. The gateway rejects unauthenticated requests with "
                "an opaque 500, which reads as 'the model is down' rather than 'no credential'."
            )
        self.calls = 0
        self.cache_hits = 0

    # ---- one request ----------------------------------------------------------------

    def _cache_key(self, prompt: str, temperature: float, attempt: int) -> str:
        digest = hashlib.sha256()
        for part in (self.config.model, prompt, f"{temperature:.4f}", str(attempt)):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def _cached(self, key: str) -> Optional[str]:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["content"]
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def _store(self, key: str, content: str) -> None:
        if self.cache_dir is None:
            return
        path = self.cache_dir / f"{key}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def complete(self, prompt: str, temperature: float, attempt: int = 0) -> str:
        key = self._cache_key(prompt, temperature, attempt)
        hit = self._cached(key)
        if hit is not None:
            self.cache_hits += 1
            return hit

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": self.config.max_tokens,
        }
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        last: Optional[Exception] = None
        # Transport retries are separate from validation retries: a 502 is not the model's fault
        # and must not consume a validation attempt, or a flaky gateway looks like a model that
        # cannot follow the prompt.
        for transport_attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise StrongLlmError(f"empty completion from {self.config.model}")
                self.calls += 1
                self._store(key, content)
                return content
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, KeyError,
                    json.JSONDecodeError, StrongLlmError) as exc:
                last = exc
                time.sleep(1.5 * (2 ** transport_attempt))
        raise StrongLlmError(f"{self.config.model} unreachable after 4 transport attempts: {last}")

    # ---- one validated result -------------------------------------------------------

    def generate(
        self,
        prompt: str,
        validate: Callable[[str], T],
        temperature: float,
        *,
        label: str = "",
        attempt_offset: int = 0,
        best_effort: bool = False,
    ) -> T:
        """Re-sample until `validate` accepts, then return whatever it returned.

        `attempt_offset` shifts the cache keys. A repair pass over cases that were individually
        valid but collectively duplicated has to draw *new* samples; without the offset it would
        replay the cached first attempt and the duplicate would survive every repair round.

        `best_effort` accepts the least-bad attempt when every attempt failed *and* at least one
        carried a payload. Failures with no payload are hard by construction -- malformed json, an
        empty field, a leaked action-entry marker -- and no amount of best-effort makes them usable.
        """
        errors: list[str] = []
        fallback: Optional[tuple[int, Any, str]] = None
        for step in range(self.config.max_attempts):
            attempt = attempt_offset + step
            # Escalate, but stay inside the API's range. Same prompt, different sample.
            hot = min(1.0, temperature + 0.2 * attempt)
            text = self.complete(prompt, hot, attempt)
            try:
                return validate(text)
            except ValidationError as exc:
                errors.append(f"attempt {attempt} (t={hot:.2f}): {exc}")
                if exc.payload is not None and (fallback is None or exc.penalty < fallback[0]):
                    fallback = (exc.penalty, exc.payload, str(exc))
        if best_effort and fallback is not None:
            return fallback[1]
        joined = "\n  ".join(errors)
        raise StrongLlmError(
            f"{label or 'request'} failed {self.config.max_attempts} validation attempts:\n  {joined}"
        )




def run_parallel(
    items: Sequence[T],
    worker: Callable[[T], Any],
    concurrency: int,
    *,
    on_done: Optional[Callable[[int, int], None]] = None,
    max_failures: int = 25,
    describe: Optional[Callable[[T], str]] = None,
) -> list[Any]:
    """`worker` over `items`, concurrently, results in the input order.

    Order-preserving on purpose. The products are jsonl files that get diffed between runs, and a
    completion-order write turns "two cases changed" into "every line moved".

    Failures are **collected**, not propagated on sight. Letting the first exception out of the
    `as_completed` loop is what a plain implementation does, and on a 1050-item job it produces a
    process that appears to hang: the `with` block exits, `ThreadPoolExecutor.__exit__` calls
    `shutdown(wait=True)`, and the remaining thousand queued tasks run to completion with nobody
    reading their results and no progress line being printed. Measured once, on this pipeline: the
    job sat at "29/1050" for four minutes while quietly finishing the other 1021.

    Collecting them also answers a better question. One exception says one case failed; the summary
    says whether 1 case failed or 400 did, and those call for opposite responses -- fix that case,
    or fix the validator.
    """
    results: list[Any] = [None] * len(items)
    failures: list[str] = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(worker, item): i for i, item in enumerate(items)}
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            done += 1
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001  every failure is reported, none is swallowed
                label = describe(items[index]) if describe else f"item {index}"
                failures.append(f"{label}: {exc}")
                if len(failures) > max_failures:
                    for pending in futures:
                        pending.cancel()
                    break
            if on_done is not None:
                on_done(done, len(items))
    if failures:
        joined = "\n  ".join(failures[:20])
        more = f"\n  ... and {len(failures) - 20} more" if len(failures) > 20 else ""
        raise StrongLlmError(f"{len(failures)} of {len(items)} items failed:\n  {joined}{more}")
    return results



# --------------------------------------------------------------------------- json helpers


def extract_json(text: str) -> Any:
    """The first json value in a response, tolerating a fenced block or a sentence around it.

    The prompts all say "return only json". Models mostly comply, and wrapping the payload in
    ```json is the one deviation frequent enough that rejecting it would burn attempts on a
    difference that carries no information.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines[1:]).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValidationError(f"no json value in response: {text[:200]!r}")
