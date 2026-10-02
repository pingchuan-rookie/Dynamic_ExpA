"""Isolated official WebShop backend, driven by a standard-library JSONL protocol.

Run this file with .venvs/webshop/bin/python, never import upstream dependencies
in the training interpreter. One process owns one full catalog/Lucene SimServer.
Requests: {id, op, ...}; replies: {id, ok, result} or {id, ok:false, error}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ASSETS = ROOT / "data/webshop/assets"
DEFAULT_SOURCE = ROOT / "agent_system/environments/env_package/webshop/source"
SEED = 233
FORMAT_VERSION = 1
ASSET_HASHES = {
    "items_shuffle.json": "2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8",
    "items_ins_v2.json": "1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e",
    "items_human_ins.json": "cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def task_split(task_id):
    return "test" if task_id < 500 else "dev" if task_id < 1500 else "train"


def configure_official(source_dir, assets_dir, java_home=None):
    """Redirect only data/index locations; retain upstream algorithms/templates."""
    source_dir, assets_dir = Path(source_dir).resolve(), Path(assets_dir).resolve()
    if not (source_dir / "web_agent_site/envs/web_agent_text_env.py").is_file():
        raise FileNotFoundError(f"Official WebShop source missing: {source_dir}")
    java_home = Path(java_home or os.environ.get("WEBSHOP_JAVA_HOME") or
                     ROOT / ".venvs/webshop/java").resolve()
    if not (java_home / "bin/java").is_file():
        raise FileNotFoundError(f"Java 11 missing: {java_home}; run prepare_webshop_assets.py --install")
    import subprocess
    java_version = subprocess.run([str(java_home / "bin/java"), "-version"],
                                  capture_output=True, text=True, check=True).stderr
    if not re.search(r'version "11\.', java_version):
        raise RuntimeError(f"WebShop requires Java 11: {java_version.strip()}")
    os.environ["JAVA_HOME"] = str(java_home)
    os.environ["PATH"] = str(java_home / "bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(source_dir))
    import web_agent_site.utils as utils
    utils.DEFAULT_FILE_PATH = str(assets_dir / "items_shuffle.json")
    utils.DEFAULT_ATTR_PATH = str(assets_dir / "items_ins_v2.json")
    utils.HUMAN_ATTR_PATH = str(assets_dir / "items_human_ins.json")
    import web_agent_site.engine.engine as engine
    engine.DEFAULT_FILE_PATH = utils.DEFAULT_FILE_PATH
    engine.DEFAULT_ATTR_PATH = utils.DEFAULT_ATTR_PATH
    engine.HUMAN_ATTR_PATH = utils.HUMAN_ATTR_PATH

    def full_search_engine(num_products=None):
        if num_products is not None:
            raise ValueError("WebShop full requires num_products=None")
        return engine.LuceneSearcher(str(assets_dir / "search_engine/indexes"))

    engine.init_search_engine = full_search_engine
    from web_agent_site.envs import web_agent_text_env as official
    official.init_search_engine = full_search_engine
    official.DEFAULT_FILE_PATH = utils.DEFAULT_FILE_PATH
    return official


class WebShopEnv:
    def __init__(self, config=None, *, verify_manifest=True):
        config = dict(config or {})
        if config.get("seed", SEED) != SEED:
            raise ValueError(f"Official full assets use fixed construction seed {SEED}")
        if config.get("num_products") is not None or config.get("human_goals", True) is not True:
            raise ValueError("Only full catalog with human_goals=true is supported")
        if config.get("observation_mode", "text_rich") != "text_rich":
            raise ValueError("WebShop dataset protocol requires observation_mode=text_rich")
        self.assets_dir = Path(config.get("assets_dir", DEFAULT_ASSETS)).resolve()
        self.source_dir = Path(config.get("source_dir", config.get("webshop_root", DEFAULT_SOURCE))).resolve()
        self.sessions = {}
        self.max_sessions = int(config.get("max_sessions", 4096))
        if self.max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        manifest_path = self.assets_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        if verify_manifest:
            if (self.manifest.get("variant") != "full" or
                    self.manifest.get("seed") != SEED or
                    self.manifest.get("human_goals") is not True):
                raise ValueError(f"Missing/incompatible full asset manifest: {manifest_path}")
            if self.manifest.get("format_version") != FORMAT_VERSION or self.manifest.get("observation_mode") != "text_rich":
                raise ValueError("Unsupported WebShop manifest protocol/observation mode")
            if not self.manifest.get("source_files"):
                raise ValueError("Manifest must pin official source and templates")
            for name, metadata in self.manifest["source_files"].items():
                if sha256_file(self.source_dir / name) != metadata["sha256"]:
                    raise ValueError(f"Official source differs from prepared assets: {name}")
            for name, expected in ASSET_HASHES.items():
                path = self.assets_dir / name
                if sha256_file(path) != expected:
                    raise ValueError(f"Full asset checksum mismatch: {path}")
            for name, metadata in self.manifest["index"]["files"].items():
                if sha256_file(self.assets_dir / "search_engine/indexes" / name) != metadata["sha256"]:
                    raise ValueError(f"Lucene index checksum mismatch: {name}")
        self.official = configure_official(self.source_dir, self.assets_dir, config.get("java_home"))
        from thefuzz import fuzz
        if fuzz.SequenceMatcher.__module__ != "difflib":
            raise RuntimeError("Optional Levenshtein changes official default scoring; use the locked WebShop environment")
        if verify_manifest:
            from importlib.metadata import version
            packages = self.manifest["runtime"]["packages"]
            normalized = {key.lower().replace("_", "-"): val for key, val in packages.items()}
            for name in ("spacy", "en-core-web-sm", "thefuzz", "pyserini", "numpy", "beautifulsoup4", "Flask"):
                if version(name) != normalized.get(name.lower()):
                    raise ValueError(f"Runtime dependency differs from manifest: {name}")
        # Upstream seeds only the shuffle, AFTER random prices and price constraints.
        # Seed construction too, without changing upstream's operation ordering.
        random.seed(SEED)
        self.server = self.official.SimServer(
            "http://127.0.0.1:3000", str(self.assets_dir / "items_shuffle.json"),
            num_products=None, human_goals=True, show_attrs=False,
        )
        self.goals_sha256 = json_digest(self.server.goals)
        self.product_prices_sha256 = json_digest(self.server.product_prices)
        self.counts = {"products": len(self.server.all_products), "goals": len(self.server.goals),
                       "test": 500, "dev": 1000, "train": len(self.server.goals) - 1500}
        if self.counts["products"] < 1_000_000 or self.counts["train"] <= 0:
            raise ValueError(f"Not the official full human benchmark: {self.counts}")
        index_manifest = json.loads((self.assets_dir / "search_engine/index_manifest.json").read_text())
        # Official DefaultLuceneDocumentGenerator skips empty contents (60 in
        # this pinned full catalog). Preserve and account for that behavior.
        if verify_manifest and index_manifest != self.manifest["index"]:
            raise ValueError("Index receipt differs from the prepared asset manifest")
        expected_docs = self.counts["products"] - len(index_manifest["empty_document_ids"])
        if (index_manifest["catalog_products"] != self.counts["products"] or
                self.server.search_engine.num_docs != expected_docs or
                index_manifest["documents"] != expected_docs):
            raise ValueError("Full catalog and Lucene nonempty document counts disagree")
        if verify_manifest:
            for key in ("goals_sha256", "product_prices_sha256"):
                if self.manifest.get(key) != getattr(self, key):
                    raise ValueError(f"Reconstructed benchmark differs from manifest: {key}")
            if self.manifest.get("counts") != self.counts:
                raise ValueError("Reconstructed counts differ from manifest")
        base = self.official.WebAgentTextEnv

        class GoalEnv(base):
            def reset(env, session=None, instruction_text=None):
                return super(GoalEnv, env).reset(
                    session=env.kwargs["goal_index"] if session is None else session,
                    instruction_text=instruction_text,
                )

        self.env_class = GoalEnv

    def health(self):
        return {"ready": True, "protocol_version": FORMAT_VERSION, "variant": "full",
                "human_goals": True, "seed": SEED, "counts": self.counts,
                "active_sessions": len(self.sessions),
                "server_sessions": len(self.server.user_sessions), "server_instances": 1,
                "goals_sha256": self.goals_sha256,
                "product_prices_sha256": self.product_prices_sha256,
                "assets_dir": str(self.assets_dir)}

    def _task_id(self, task_id, split):
        if isinstance(task_id, bool):
            raise ValueError("task_id must be a global integer goal index")
        if not isinstance(task_id, int):
            if not isinstance(task_id, str) or not re.fullmatch(r"0|[1-9][0-9]*", task_id):
                raise ValueError("task_id must be a global integer goal index")
            task_id = int(task_id)
        if not 0 <= task_id < len(self.server.goals):
            raise ValueError(f"task_id outside full goal list: {task_id}")
        if split != task_split(task_id):
            raise ValueError(f"task_id {task_id} belongs to {task_split(task_id)}, not {split}")
        return task_id

    def _result(self, session_id, observation, reward=0.0, done=False):
        entry = self.sessions[session_id]
        reward = float(reward)
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError(f"Invalid official reward: {reward}")
        return {"session_id": session_id, "task_id": entry["task_id"],
                "split": task_split(entry["task_id"]), "observation": observation,
                "instruction": self.server.goals[entry["task_id"]]["instruction_text"],
                "reward": reward, "done": bool(done), "task_score": reward if done else 0.0,
                "won": bool(done and reward == 1.0), "step_count": entry["step_count"],
                "available_actions": entry["env"].get_available_actions()}

    def create(self, session_id, task_id, split):
        if not isinstance(session_id, str) or not session_id or len(session_id) > 256:
            raise ValueError("session_id must be a nonempty string of at most 256 characters")
        if session_id in self.sessions:
            raise ValueError(f"Session already exists: {session_id}")
        if len(self.sessions) >= self.max_sessions:
            raise RuntimeError("Maximum live WebShop sessions exceeded; close unused sessions")
        task_id = self._task_id(task_id, split)
        # Prefix internal identifiers, not user identifiers; same goal can run concurrently.
        prefix = uuid.uuid4().hex + "_"
        internal_id = prefix + str(task_id)
        state = random.getstate()
        try:
            random.seed(SEED + task_id)
            env = self.env_class(observation_mode="text_rich", server=self.server,
                                 file_path=str(self.assets_dir / "items_shuffle.json"),
                                 goal_index=task_id, session_prefix=prefix, get_image=0)
            self.sessions[session_id] = {"env": env, "task_id": task_id, "done": False,
                                         "step_count": 0, "rng_state": random.getstate()}
            return self._result(session_id, env.observation)
        except BaseException:
            self.sessions.pop(session_id, None)
            self.server.user_sessions.pop(internal_id, None)
            raise
        finally:
            random.setstate(state)

    def step(self, session_id, action):
        entry = self.sessions[session_id]
        if entry["done"]:
            raise RuntimeError("Session is terminal; close it and create a new session")
        if not isinstance(action, str):
            raise TypeError("action must be an official search[...] or click[...] string")
        state = random.getstate()
        try:
            random.setstate(entry["rng_state"])
            observation, reward, done, _ = entry["env"].step(action)
            entry["rng_state"] = random.getstate()
            entry["done"] = bool(done)
            entry["step_count"] += 1
            return self._result(session_id, observation, reward, done)
        except BaseException:
            # Official actions may mutate state before raising. Never allow reuse of
            # a partially executed transition or turn infrastructure failure into reward=0.
            self.close(session_id)
            raise
        finally:
            random.setstate(state)

    def close(self, session_id):
        entry = self.sessions.pop(session_id, None)
        if entry is not None:
            env = entry["env"]
            self.server.user_sessions.pop(env.session, None)
            env.prev_obs.clear()
            env.prev_actions.clear()
            env.close()
        return {"session_id": session_id, "closed": entry is not None}

    def export_tasks(self, split=None, offset=0, limit=None, output_path=None):
        if split is not None and split not in ("test", "dev", "train"):
            raise ValueError(f"Unknown split: {split}")
        if type(offset) is not int or offset < 0 or (limit is not None and (type(limit) is not int or limit < 0)):
            raise ValueError("offset/limit must be nonnegative integers")
        ids = [i for i in range(len(self.server.goals)) if split is None or task_split(i) == split]
        ids = ids[offset: None if limit is None else offset + limit]
        tasks = []
        destination = Path(output_path).resolve() if output_path else None
        temp = destination.with_name(destination.name + ".part") if destination else None
        if destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
        stream = open(temp, "w", encoding="utf-8") if temp else None
        try:
            for task_id in ids:
                session_id = "export_" + uuid.uuid4().hex
                try:
                    initial = self.create(session_id, task_id, task_split(task_id))
                    goal = self.server.goals[task_id]
                    task = {"task_id": task_id, "split": task_split(task_id),
                            "instruction": goal["instruction_text"],
                            "initial_observation": initial["observation"],
                            "available_actions": initial["available_actions"],
                            "metadata": {"goal_sha256": json_digest(goal), "goal": goal}}
                    if stream:
                        stream.write(json.dumps(task, ensure_ascii=False, allow_nan=False) + "\n")
                    else:
                        tasks.append(task)
                finally:
                    self.close(session_id)
            if stream:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
                temp.replace(destination)
        finally:
            if stream and not stream.closed:
                stream.close()
            if temp and temp.exists():
                temp.unlink()
        result = {"count": len(ids), "manifest": self.manifest}
        if destination:
            result["output_path"] = str(destination)
            result["sha256"] = sha256_file(destination)
        else:
            result["tasks"] = tasks
        return result

    def shutdown(self):
        for session_id in list(self.sessions):
            self.close(session_id)
        self.server.search_engine.close()


