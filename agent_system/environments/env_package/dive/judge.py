"""Strict terminal scoring using the pinned DIVE verifier prompt and labels."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

# The isolated DIVE environment need not install the training package.
_PROJECT = Path(__file__).resolve().parents[4]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))
from .model_api import GenerateOptions, Message, ModelClient

from .runtime import DiveInfrastructureError, deadline, load_upstream
from .judge_config import resolve_judge_config


class DiveJudge:
    def __init__(self, config=None, repo=None, *, model_client: ModelClient):
        self.config = resolve_judge_config(config)
        self.model = self.config["model"]
        self.base_url = self.config["base_url"]
        self.provider = self.config["provider"]
        self.model_client = model_client
        self.options = GenerateOptions(
            max_output_tokens=self.config.get("max_completion_tokens", self.config.get("max_tokens", 2048)),
            temperature=0,
        )
        self.timeout_s = self.config["timeout_s"]
        load_upstream(repo)
        from dive.verifier import TaskVerifier
        self.verifier = object.__new__(TaskVerifier)

    def score(self, query, reference, answer):
        prompt = self.verifier._build_verify_prompt(query, reference, answer)
        try:
            with deadline(self.timeout_s):
                response = self.model_client.generate(
                    [Message(role="user", content=prompt)], options=self.options,
                )
                text = response.text
        except DiveInfrastructureError:
            raise
        except Exception as exc:
            raise DiveInfrastructureError(f"DIVE judge request failed ({type(exc).__name__})") from None
        result = self.verifier._parse_verification_response(text or "")
        label = result.get("correct")
        if label not in ("correct", "partial", "incorrect"):
            raise DiveInfrastructureError("DIVE judge returned an invalid judgement label")
        return {"reward": float(label == "correct"), "label": label,
                "verification": result, "judge_model": self.model, "judge_provider": self.provider,
                **({"judge_request_params": response.request_params} if self.provider == "trapi" else {}),
                "judge_prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(), "reward_rule": "strict_correct"}
