#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified Dyad sampling state machine (the single source of truth for how sampling is constrained).

One ActionRouter is called by both rollout (dyad_gpu_model_runner) and training (policy_trace
replay), which guarantees "per-decision mask consistency" (the top invariant). It reads the
structured fields produced by compile_action_schema (markers / actions / value_sets /
argument_order / head), generalizing the original FlatActionRouter:

  Phases: NONE -> ACTION_NAME -> ARGUMENT_VALUE_{OPEN,CLOSED} -> (loop) -> exit -> NONE
  Three mask kinds, i.e. three shapes the admissible action set C_t can take:
    force        -- exactly 1 token is allowed (template / surface_form_suffix / exit / eos)
    base_vocab   -- C_t is the base vocabulary [0,V) (NONE / FREE_TEXT / open param values)
    expanded     -- C_t is a subset of the expanded action space E (extended ids in [V,V+|E|)) (ACTION_NAME /
                    closed param values)

Native open arguments enter FREE_TEXT after action selection. Enumerated values use
fixed slots and ARGUMENT_VALUE_CLOSED; parameter names are not sampled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class Phase(Enum):
    NONE = auto()
    ACTION_NAME = auto()
    ARGUMENT_VALUE_OPEN = auto()
    ARGUMENT_VALUE_CLOSED = auto()
    # Dyad free-argument (argument_order="free"): one free-text segment, no param slot. Kept apart from
    # ARGUMENT_VALUE_OPEN on purpose -- free arguments have no slot to fill (the synthesized slot carries
    # value_end_ids=[] / surface_form_suffix_seq=[]), and carrying slot state it never
    # reads is surplus state the rollout mask and the training replay can drift apart on.
    FREE_TEXT = auto()


@dataclass
class Decision:
    """How this step constrains sampling (typed; read both when rollout applies the mask and when training rebuilds it)."""
    kind: str                              # "force" | "base_vocab" | "expanded"
    phase: str = ""                        # diagnostics only
    forced_token: int | None = None        # kind==force
    allowed_ids: list[int] = field(default_factory=list)  # kind==expanded: the allowed **extended ids**
    is_tool: bool = False                  # True when kind==expanded and the action head is used


def _get(mapping: dict, key: int):
    if key in mapping:
        return mapping[key]
    return mapping[str(key)]


class ActionRouter:
    def __init__(self, action_config: dict[str, Any]):
        self.cfg = action_config
        self.V = int(action_config["num_embeddings_size"])
        markers = action_config.get("markers", {}) or {}
        self._enter_seq = list(markers.get("enter_seq", []))
        matcher = markers.get("enter_matcher") if not action_config.get("native_tool_protocol") else None
        self._enter_transitions = ([{int(k): int(v) for k, v in row.items()}
                                    for row in matcher["transitions"]] if matcher else None)
        self._enter_length = int(matcher["length"]) if matcher else 0
        self._enter_completion = ({int(k): list(v) for k, v in matcher["completion"].items()}
                                  if matcher else {})
        self._enter_state = 0
        self._native_action_prefix = list(markers.get("native_action_prefix_seq", []))
        self._exit_seq = list(markers.get("exit_seq", []))
        self._value_end_set = set(int(x) for x in markers.get("value_end_ids", []))
        self._exit_value_end_set = set(int(x) for x in markers.get("exit_value_end_ids", []))
        self._exit_value_prefix_ids = {
            int(k): list(v) for k, v in (markers.get("exit_value_prefix_ids") or {}).items()
        }
        self._turn_end_ids = list(markers.get("turn_end_ids", []))
        self._max_value_tokens = int(markers.get("max_value_tokens", 0) or 0)
        self._native_exit_sequences = [list(sequence) for sequence in markers.get("native_exit_sequences", [])]
        if any(not sequence for sequence in self._native_exit_sequences):
            raise ValueError("Native action exit sequences must not be empty")
        self._native_recent: list[int] = []
        self._action_name_ids = list(action_config["action_name_ids"].values())
        self._actions = action_config["actions"]
        self._value_sets = action_config.get("value_sets", {}) or {}
        self._argument_order = action_config.get("argument_order", "fixed")
        if self._argument_order not in ("free", "fixed") or action_config.get("head", {}).get("argument_key"):
            raise ValueError("Only free arguments or fixed value slots are supported")

        # per-sequence state
        self.phase = Phase.NONE
        self.action_name: str | None = None
        self.remaining_params: list[dict] = []   # unfilled parameters in schema order
        self.current_param: dict | None = None
        self.pending: list[int] = []
        self.after: tuple | None = None
        self.none_recent: list[int] = []

        # open-value state: how many tokens have been written + this param's suffix terminator set.
        self.value_tokens = 0
        self._cur_value_end: set[int] = set()
        # FREE_TEXT runaway guard. Not slot state: free arguments have no slot. Without it a missed terminator
        # generates forever.
        self.free_tokens = 0
        # Free arguments only: how many times the action head was consumed this tool call (guard, see advance).
        self.head_samples = 0

    # ---- how this step must be masked ----
    @property
    def in_enter_prefix(self) -> bool:
        """True when the tokens seen so far in NONE could still complete the enter marker.

        `DyadGPUModelRunner.add_action_content` transports every decode step, but
        resets the decision record only after a completed action. A marker prefix
        is not a completion boundary even when phase is NONE and pending is empty.
        Its sampled prefix tokens must remain available to reconstruct the router.
        Marker length depends on the schema and tokenizer, not just the environment.
        """
        if self._enter_transitions is not None:
            return self._enter_state > 0
        es = self._enter_seq
        if not es or not self.none_recent:
            return False
        # A suffix of what we have seen that is also a proper prefix of the marker means the marker
        # may still complete. Longest first: the longest match is the one that matters.
        for k in range(min(len(self.none_recent), len(es) - 1), 0, -1):
            if self.none_recent[-k:] == es[:k]:
                return True
        return False

    def decision(self) -> Decision:
        if self.pending:
            return Decision(kind="force", phase=self.phase.name, forced_token=int(self.pending[0]))
        if self.phase is Phase.NONE:
            return Decision(kind="base_vocab", phase="NONE")
        if self.phase is Phase.ARGUMENT_VALUE_OPEN:
            return Decision(kind="base_vocab", phase="ARGUMENT_VALUE_OPEN")
        if self.phase is Phase.FREE_TEXT:
            return Decision(kind="base_vocab", phase="FREE_TEXT")
        if self.phase is Phase.ACTION_NAME:
            return Decision(kind="expanded", phase="ACTION_NAME",
                            allowed_ids=list(self._action_name_ids), is_tool=True)
        if self.phase is Phase.ARGUMENT_VALUE_CLOSED:
            allowed = list(self._value_sets.get(self.current_param["value_set"], []))
            return Decision(kind="expanded", phase="ARGUMENT_VALUE_CLOSED", allowed_ids=allowed, is_tool=True)
        raise ValueError(self.phase)

    # ---- consume one token, advance the state, return the plain tokens to write into the context ----
    def advance(self, raw_id: int) -> list[int]:
        raw_id = int(raw_id)

        # (a) forced token
        if self.pending:
            tok = self.pending.pop(0)
            if not self.pending and self.after is not None:
                self._run_after()
            return [tok]

        # (b) NONE: base vocabulary; watch for the enter marker
        if self.phase is Phase.NONE:
            es = self._enter_seq
            self.none_recent.append(raw_id)
            cap = max(len(es) + 2, 2)
            if len(self.none_recent) > cap:
                del self.none_recent[:-cap]
            if self._enter_transitions is not None:
                self._enter_state = self._enter_transitions[self._enter_state].get(raw_id, 0)
                if self._enter_state < 0:
                    raise ValueError("Sampled token crosses the action marker/name boundary; "
                                     "cannot replace an already sampled base-vocabulary action")
                if self._enter_state == self._enter_length:
                    self._enter_state = 0
                    self.none_recent = []
                    self.phase = Phase.ACTION_NAME
                elif self._enter_state in self._enter_completion:
                    self.pending = list(self._enter_completion[self._enter_state])
                    self._enter_state = 0
                    self.none_recent = []
                    self.after = ("enter_action",)
            elif len(es) == 1:
                if raw_id == es[0]:
                    self.none_recent = []
                    self.phase = Phase.ACTION_NAME
            elif es:
                if len(self.none_recent) >= len(es) and self.none_recent[-len(es):] == es:
                    self.none_recent = []
                    self.phase = Phase.ACTION_NAME
                elif (not self.cfg.get("native_tool_protocol") and len(self.none_recent) >= len(es) - 1
                      and self.none_recent[-(len(es) - 1):] == es[:-1]):
                    # Prefix trigger + tail completion (a multi-token marker's tail char may merge
                    # with the next char).
                    self.none_recent = []
                    self.pending = [es[-1]]
                    self.after = ("enter_action",)
            if self.phase is Phase.ACTION_NAME and self._native_action_prefix:
                # Force syntax before asking the expanded head for the tool name.
                # Triggering on the native special token avoids BPE marker/name merging.
                self.pending = list(self._native_action_prefix)
            return [] if raw_id >= self.V else [raw_id]

        # (c) ACTION_NAME: sample the action-name extended id -> write surface_form_seq + fix the param list
        if self.phase is Phase.ACTION_NAME:
            self.head_samples += 1
            if self._argument_order == "free" and self.head_samples > 1:
                # Dyad free-argument's defining property: the expanded action space holds only action names, so one tool
                # call consumes the head exactly once. A second sample means the state machine looped
                # back through a param path free arguments do not have -- fail loudly, because the quiet
                # failure mode is a training replay whose decision_mask no longer lines up with what
                # was actually sampled, which does not raise, it just turns training into noise.
                raise ValueError(
                    f"Dyad free-argument sampled the action head {self.head_samples} times in one tool call; "
                    "argument_order='free' must sample it exactly once"
                )
            name = _get(self.cfg["id_to_str"], raw_id)
            self.action_name = name
            action = self._actions[name]
            seq = list(action["surface_form_seq"])
            first = seq[0] if seq else None
            self.pending = seq[1:]
            self.remaining_params = [dict(p) for p in action["params"]]
            self.after = ("after_action_name",)
            if not self.pending:
                self._run_after()
            return [first] if first is not None else []

        # (e) ARGUMENT_VALUE_OPEN: free value over the base vocabulary; three terminations + a length fallback
        if self.phase is Phase.ARGUMENT_VALUE_OPEN:
            # (e1) Suffix termination: the model has already written a prefix of surface_form_suffix (e.g.
            #      ' and' after 'mug', or the '.' inside the BPE-merged '1.</'). This token is
            #      **rewritten** into surface_form_suffix's first token and the remainder goes into pending
            #      for force-writeing, so the context always ends up as the canonical phrasing.
            #      Requires a non-empty value (value_tokens>0), so the very first token cannot
            #      mis-trigger it.
            if self.value_tokens > 0 and raw_id in self._cur_value_end:
                seq = list(self.current_param.get("surface_form_suffix_seq", []))
                first = seq[0] if seq else None
                self.pending = seq[1:]
                self._set_after_for_next()
                if not self.pending and self.after is not None:
                    self._run_after()
                return [first] if first is not None else []
            if raw_id in self._exit_value_end_set:
                # The model wrote the exit marker itself ('</', or the BPE-merged '.</'): this signals
                # the **end of the action**, not merely the end of the current param -- finish right
                # away and leave the remaining unfilled params empty (tool_parser skips them).
                # Rewrite that token into the exit marker's first token and force-write the rest, so
                # the context stays canonical.
                # Scenario: the model does not split params the way the schema does, e.g.
                # 'receptacle is sofa 1.</Action>' (it folded the id into the value and skipped
                # ' from ') -- a legal env command is still obtained in that case.
                self._finish()
                first = self.pending.pop(0) if self.pending else None
                return [first] if first is not None else []
            if raw_id in self._value_end_set:
                # Got '\n': write it into the context, the value ends, write surface_form_suffix, then advance.
                self.pending = list(self.current_param.get("surface_form_suffix_seq", []))
                self._set_after_for_next()
                if not self.pending and self.after is not None:
                    self._run_after()
                return [raw_id]
            self.value_tokens += 1
            if self._max_value_tokens and self.value_tokens >= self._max_value_tokens:
                # Fallback: force termination when the terminator was missed, so the state machine is
                # guaranteed to reach finish.
                self.pending = list(self.current_param.get("surface_form_suffix_seq", []))
                self._set_after_for_next()
                if not self.pending and self.after is not None:
                    self._run_after()
            return [raw_id]

        # (e') FREE_TEXT (Dyad free-argument): one free-text segment straight through to the action close.
        # Same three exits ARGUMENT_VALUE_OPEN offers, minus the per-param suffix rewriting -- free text has
        # no suffix to rewrite, so none of that code path is reachable for it anyway.
        if self.phase is Phase.FREE_TEXT:
            if self._native_exit_sequences:
                self._native_recent.append(raw_id)
                width = max(map(len, self._native_exit_sequences))
                self._native_recent = self._native_recent[-width:]
                if any(self._native_recent[-len(sequence):] == sequence for sequence in self._native_exit_sequences):
                    self._finish()
                    self._native_recent = []
                return [raw_id]
            if raw_id in self._exit_value_end_set:
                # Opt-in native text keeps any value prefix merged with the exit.
                # Still emit exactly one token now and force the remainder, so raw
                # decisions and policy replay retain the runner's one-token cadence.
                prefix = self._exit_value_prefix_ids.get(raw_id, [])
                self._finish()
                self.pending = list(prefix) + self.pending
                first = self.pending.pop(0) if self.pending else None
                return [first] if first is not None else []
            if raw_id in self._value_end_set:
                # '\n': write it, then close (free arguments have no next slot to advance to).
                self._finish()
                return [raw_id]
            self.free_tokens += 1
            if self._max_value_tokens and self.free_tokens >= self._max_value_tokens:
                # Terminator missed -- close anyway so the state machine always reaches finish.
                self._finish()
            return [raw_id]

        # (f) ARGUMENT_VALUE_CLOSED: pick 1 extended id from value_set (a single pick ends it, no value_end)
        if self.phase is Phase.ARGUMENT_VALUE_CLOSED:
            seq = list(_get(self.cfg["id_to_seq"], raw_id))
            first = seq[0] if seq else None
            self.pending = seq[1:] + list(self.current_param.get("surface_form_suffix_seq", []))
            self._set_after_for_next()
            if not self.pending and self.after is not None:
                self._run_after()
            return [first] if first is not None else []

        raise ValueError(self.phase)

    # ---- internal transitions ----
    def _set_after_for_next(self):
        if self.remaining_params:
            self.after = ("next_param",)
        else:
            self.after = ("finish",)

    def _enter_argument_value(self):
        vk = self.current_param.get("value_kind", "open")
        self.phase = Phase.ARGUMENT_VALUE_CLOSED if vk == "closed" else Phase.ARGUMENT_VALUE_OPEN
        self.value_tokens = 0
        self._cur_value_end = set(int(i) for i in (self.current_param.get("value_end_ids") or []))

    def _begin_next_param(self):
        self.current_param = self.remaining_params.pop(0)
        self._enter_argument_value()

    def _run_after(self):
        act = self.after
        self.after = None
        if act[0] == "enter_action":
            self.phase = Phase.ACTION_NAME
            self.none_recent = []
        elif act[0] == "after_action_name":
            if self._argument_order == "free":
                # Dyad free-argument: the head fixed the action name; everything after it is free text. No
                # slot is taken, so remaining_params stays as compiled and is simply never read.
                self.phase = Phase.FREE_TEXT
                self.free_tokens = 0
            elif not self.remaining_params:
                self._finish()
            else:
                self._begin_next_param()
        elif act[0] == "next_param":
            if not self.remaining_params:
                self._finish()
            else:
                self._begin_next_param()
        elif act[0] == "finish":
            self._finish()

    def _finish(self):
        self.phase = Phase.NONE
        self.action_name = None
        self.current_param = None
        self.remaining_params = []
        self.none_recent = []
        self.value_tokens = 0
        self._cur_value_end = set()
        self.free_tokens = 0
        self.head_samples = 0
        self.pending = list(self._exit_seq) + list(self._turn_end_ids)


# ======================================================================
# Training side: replay ActionRouter to rebuild the masks (per-decision identical to rollout)
# ======================================================================
def build_unified_policy_trace(
    action_config: dict[str, Any],
    raw_token_ids: list[int],
    emitted_token_ids: list[int],
) -> dict[str, Any]:
    """Replay the raw decision sequence recorded by rollout with ActionRouter, rebuilding the mask at each emitted position.

    The output matches policy_trace.build_policy_trace:
      seq_mask[i]           whether the i-th emitted position is a real policy decision
      tool_mask[i]          whether that decision uses the action head (an extended-id decision)
      allowed_action_ids[i] extended ids allowed at that decision point (already -vocab_size; empty for vocab decisions)
      response_dyad[i]      decision positions carry the raw decision id back, the rest match emitted
    """
    V = int(action_config["num_embeddings_size"])
    n = len(emitted_token_ids)
    seq_mask = [False] * n
    tool_mask = [False] * n
    allowed = [[] for _ in range(n)]
    response_dyad = list(emitted_token_ids)

    router = ActionRouter(action_config)
    raw_idx = 0
    pos = 0
    guard = 0

    def check_emitted(raw: int, d: Decision) -> None:
        # Match the runner's one-token write-back, including rewritten value terminators.
        # A suffix-only decision payload must never label positions in preceding Think text.
        written = router.advance(raw)
        if not written and raw >= V:
            raise ValueError(
                f"Dyad replay at emitted position {pos}: phase={d.phase} consumed "
                f"expanded id {raw} but produced no token; rollout requires one base token per step."
            )
        expected = int(written[0]) if written else raw
        if expected != int(emitted_token_ids[pos]):
            raise ValueError(
                f"Dyad replay token mismatch at emitted position {pos}: "
                f"expected={expected}, emitted={emitted_token_ids[pos]}, raw={raw}, "
                f"phase={d.phase}, kind={d.kind}, raw_idx={raw_idx}/{len(raw_token_ids)}."
            )

    while pos < n:
        guard += 1
        if guard > 10 * (n + 8):
            raise RuntimeError("unified policy_trace replay loop guard")
        d = router.decision()
        if d.kind == "force":
            check_emitted(d.forced_token, d)
            pos += 1
            continue
        if raw_idx >= len(raw_token_ids):
            raise ValueError(
                f"Dyad replay exhausted raw decisions at emitted position {pos}/{n}: "
                f"phase={d.phase}, kind={d.kind}."
            )
        raw = int(raw_token_ids[raw_idx]); raw_idx += 1
        is_tool = raw >= V
        seq_mask[pos] = True
        tool_mask[pos] = is_tool
        # Vocab terminators can also be rewritten into canonical surface tokens.
        # Score the sampled decision, not its rendered replacement, for either head.
        response_dyad[pos] = raw
        if d.kind == "expanded":
            allowed[pos] = [aid - V for aid in d.allowed_ids]
        else:  # base_vocab
            allowed[pos] = []
        # `is_tool` comes from what rollout actually sampled; `allowed` comes from what the replay
        # thinks this position offers. They are two independent readings of the same decision, and
        # AGENTS.md section 1 requires them to agree: the admissible action set applied while sampling
        # and the one replayed for training must be the same one.
        #
        # Without this check a disagreement produces tool_mask=True with an empty admissible list,
        # which is not detected here at all -- it surfaces two layers later as
        # `RuntimeError: Dyad tool position has no allowed actions` inside compute_split_policy_outputs,
        # with no phase, no position and no schema in the message. That is the shape the codegym
        # blocker (B1, filed 2026-07-28) has had all along, which is why it stayed undiagnosed.
        #
        # Raising here instead is not a fix for B1; it is what makes B1 diagnosable, because this is
        # the frame that still holds the router state that caused it.
        if d.kind == "expanded" and not is_tool:
            raise RuntimeError(
                f"Dyad replay expected an expanded action at emitted position {pos}, "
                f"but rollout sampled base-vocabulary id {raw}; phase={d.phase}."
            )
        if is_tool and d.kind != "expanded":
            raise RuntimeError(
                "Dyad replay disagrees with rollout: rollout sampled expanded id "
                f"{raw} (action_id={raw - V}) at emitted position {pos}, but the replayed router is "
                f"in phase {getattr(d, 'phase', '?')} with kind={d.kind!r}, which admits no expanded "
                f"action. schema={action_config.get('env_name')!r} "
                f"raw_idx={raw_idx - 1}/{len(raw_token_ids)} "
                f"action_size={action_config.get('total_size')}. "
                "The admissible action set applied while sampling and the one replayed for training "
                "must be the same one (AGENTS.md section 1)."
            )
        if is_tool and (raw - V) not in allowed[pos]:
            raise RuntimeError(
                "Dyad replay disagrees with rollout: rollout sampled expanded id "
                f"{raw} (action_id={raw - V}) at emitted position {pos}, but the replayed admissible "
                f"action set is {allowed[pos]}. schema={action_config.get('env_name')!r} "
                f"phase={getattr(d, 'phase', '?')} raw_idx={raw_idx - 1}/{len(raw_token_ids)}."
            )
        check_emitted(raw, d)
        pos += 1

    if raw_idx != len(raw_token_ids):
        raise ValueError(
            f"Dyad replay has {len(raw_token_ids) - raw_idx} unconsumed raw decisions "
            f"after {n} emitted tokens."
        )

    return {
        "emitted_token_ids": list(emitted_token_ids),
        "response_dyad": response_dyad,
        "seq_mask": seq_mask,
        "tool_mask": tool_mask,
        "allowed_action_ids": allowed,
        "action_size": int(action_config["total_size"]),
    }


def _clean_open_value(text: str, surface_form_suffix: str) -> str:
    """Final cleanup of an open param value: strip whitespace + drop any surface_form_suffix fragment stuck to the tail.

    When the terminator was missed (BPE merged the value's last char with the boundary, e.g. '1.')
    or the length fallback fired, the value text may carry the leading fragment of surface_form_suffix;
    it is stripped here by **longest match**, so the env is always fed a clean value.
    """
    text = text.strip()
    for boundary in (surface_form_suffix, surface_form_suffix.lstrip()):
        if not boundary:
            continue
        for k in range(len(boundary), 0, -1):
            if text.endswith(boundary[:k]) and len(text) > k:
                text = text[: -k].strip()
                break
    return text.strip()


def replay_unified_actions(
    action_config: dict[str, Any],
    raw_token_ids: list[int],
    tokenizer=None,
) -> list[dict]:
    """Replay the decision sequence with ActionRouter -> [{"name", "params": {pname: value_str}}] (in order of appearance).

    Used by env-serialize (tool_parser) to restore decisions into environment commands (e.g.
    alfworld). Aligned with rollout decision by decision:
      - force decisions (forced tokens): advance the router, consume no raw token, no semantics.
      - real decisions: consume one raw token and route it to an action name / param value by phase.
    NONE decisions (thinking text + the enter trigger) are ignored. A param value's owner is
    resolved by reading current_param before advance in fixed slot order.

    Two kinds of param values:
      - closed: one extended id maps directly to a value string (id_to_str);
      - open  : the model writes it out token by token over the base vocabulary; the tokens are
        accumulated here and decoded with the tokenizer (terminators excluded).
        An open param therefore **must** be given a tokenizer, otherwise its value is an empty string.
    When raw_token_ids is truncated, the actions collected so far are still returned, without error.
    """
    id_to_str = action_config["id_to_str"]
    router = ActionRouter(action_config)
    actions: list[dict] = []
    cur: dict | None = None
    open_buf: list[int] = []
    raw_idx = 0
    n = len(raw_token_ids)
    guard = 0

    def _flush_open(param: dict) -> None:
        nonlocal open_buf
        if cur is None or param is None:
            open_buf = []
            return
        if tokenizer is not None and open_buf:
            try:
                text = tokenizer.decode(open_buf)
            except Exception:
                text = ""
        else:
            text = ""
        cur["params"][param["name"]] = _clean_open_value(text, param.get("surface_form_suffix", "") or "")
        open_buf = []

    while raw_idx < n:
        guard += 1
        if guard > 10 * (n + 8):
            raise RuntimeError("replay_unified_actions loop guard")
        d = router.decision()
        if d.kind == "force":
            router.advance(d.forced_token)
            continue
        raw = int(raw_token_ids[raw_idx]); raw_idx += 1
        phase = d.phase
        if phase == "ACTION_NAME":
            if cur is not None:
                actions.append(cur)
            cur = {"name": _get(id_to_str, raw), "params": {}}
            open_buf = []
            router.advance(raw)
            continue
        if phase == "ARGUMENT_VALUE_CLOSED" and cur is not None and router.current_param is not None:
            cur["params"][router.current_param["name"]] = _get(id_to_str, raw)
            router.advance(raw)
            continue
        if phase == "FREE_TEXT" and cur is not None:
            # Dyad free-argument: one free-text segment per action, so there is no current_param to read --
            # the value belongs to the action's single synthesized slot.
            is_end = raw in router._exit_value_end_set or raw in router._value_end_set
            if not is_end:
                open_buf.append(raw)
            router.advance(raw)
            if router.phase is not Phase.FREE_TEXT:
                slots = router._actions[cur["name"]]["params"]
                _flush_open(slots[0] if slots else None)
            continue
        if phase == "ARGUMENT_VALUE_OPEN" and cur is not None and router.current_param is not None:
            param = router.current_param
            is_end = (
                (router.value_tokens > 0 and raw in router._cur_value_end)
                or raw in router._exit_value_end_set
                or raw in router._value_end_set
            )
            if not is_end:
                open_buf.append(raw)
            router.advance(raw)
            # The value has ended once the router set `after` (surface_form_suffix pending) or already
            # left the OPEN phase.
            if router.after is not None or router.phase is not Phase.ARGUMENT_VALUE_OPEN:
                _flush_open(param)
            continue
        # NONE: no param-value semantics.
        router.advance(raw)

    if cur is not None:
        actions.append(cur)
    return actions

