"""Route legacy flat action decisions with one open argument phase."""
from __future__ import annotations

from enum import Enum, auto
from typing import Any


def enter_decisions(action_config: dict[str, Any]) -> list[int]:
    """The token sequence the model must sample in the NONE phase in order to "trigger entering an action".

    - Multi-token marker (e.g. <|FunctionCallBegin|>): returns the prefix enter_seq[:-1]; the last
      token is force-written by the router once it detects the prefix (tail completion), so the model
      only has to emit the prefix.
    - Single-token marker (e.g. <tool_call>): returns [enter_seq[0]].
    Mainly for CPU tests/tools to build decision sequences correctly for the current marker, instead
    of hard-coding the marker's token count.
    """
    es = list(action_config.get("enter_seq", []))
    if not es:
        return list(action_config.get("enter_trigger_ids", []))[:1]
    return list(es) if len(es) == 1 else list(es[:-1])



class Phase(Enum):
    # Not the same enum as agent_system/policies/dyad/actions/action_router.py's Phase, and the two must not be merged: codegym
    # has one param phase where the router has three (ARGUMENT_KEY / ARGUMENT_VALUE_OPEN /
    # ARGUMENT_VALUE_CLOSED), and this ARGUMENT_VALUE collides on auto() value 3 with the router's
    # ARGUMENT_KEY. Nothing compares them across modules today -- keep it that way.
    NONE = auto()
    ACTION_NAME = auto()
    ARGUMENT_VALUE = auto()   # param value: free generation over the full vocabulary, '\n' terminated (the only param phase)



def allowed_action_ids_for(action_config, phase, action_name, argument_index):
    """Decision point -> the allowed extended action ids (None = plain vocabulary). The single source of truth shared by rollout and trace."""
    if phase is Phase.NONE:
        return None
    if phase is Phase.ACTION_NAME:
        return list(action_config["action_name_ids"].values())
    if phase is Phase.ARGUMENT_VALUE:
        return None  # the full vocabulary is open; '\n'(value_end_token_ids) lives in it and acts as the terminator
    raise ValueError(phase)



class FlatActionRouter:
    def __init__(self, action_config: dict[str, Any]):
        self.cfg = action_config
        self.phase = Phase.NONE
        self.action_name = None
        self.argument_index = 0
        self.n_params = 0
        self.pending: list[int] = []
        self.after = None  # ("begin_param", i) | ("finish",) | ("enter_action",) | None
        # ARGUMENT_VALUE terminators: the set of token ids in the vocabulary that decode to '\n'.
        self._value_end_set = set(int(x) for x in action_config.get("value_end_token_ids", []))
        # Optional: the model emitting the exit marker's first token also ends the value (calc uses
        # '</'; codegym's empty set disables it).
        self._exit_value_end_set = set(int(x) for x in action_config.get("exit_value_end_ids", []))
        # The canonical id sequence of a multi-token enter marker (e.g. <|FunctionCallBegin|>) + the
        # exit surface-form sequence.
        self._enter_seq = list(action_config.get("enter_seq", []))
        self._exit_surface_form_seq = list(action_config.get("exit_surface_form_seq", []))
        # A rolling buffer of the most recent tokens SAMPLEd in the NONE phase; decoding-independent --
        # multi-token markers are detected by endswith on the token-id sequence. Cleared on entry/exit.
        self.none_recent: list[int] = []

    # ---- how this step must be masked ----
    def decision(self) -> dict:
        if self.pending:
            return {"kind": "force", "forced_token": self.pending[0]}
        if self.phase is Phase.ARGUMENT_VALUE:
            return {"kind": "sample", "phase": "ARGUMENT_VALUE", "vocab_open": True,
                    "value_end_allowed": False, "allowed_action_ids": []}
        allowed = allowed_action_ids_for(self.cfg, self.phase, self.action_name, self.argument_index)
        if allowed is None:  # NONE
            return {"kind": "sample", "phase": "NONE", "vocab_open": True,
                    "value_end_allowed": False, "allowed_action_ids": []}
        return {"kind": "sample", "phase": self.phase.name, "vocab_open": False,
                "value_end_allowed": False, "allowed_action_ids": list(allowed)}

    # ---- consume one token, return the plain tokens to write into the context ----
    def advance(self, raw_id: int) -> list[int]:
        V = self.cfg["num_embeddings_size"]

        # (a) forced token
        if self.pending:
            tok = self.pending.pop(0)
            if not self.pending and self.after is not None:
                self._run_after()
            return [tok]

        # (b) NONE
        if self.phase is Phase.NONE:
            es = self._enter_seq
            # Record NONE/SAMPLE tokens into the rolling buffer (force-write takes branch (a) and
            # never reaches here).
            self.none_recent.append(raw_id)
            cap = max(len(es) + 2, 2)
            if len(self.none_recent) > cap:
                del self.none_recent[:-cap]
            if len(es) == 1:
                # Single-token marker (e.g. <tool_call>=151657): a hit enters immediately (backward compatible).
                if raw_id == es[0]:
                    self.none_recent = []
                    self.phase = Phase.ACTION_NAME
            elif es:
                # Multi-token marker: the tail character (e.g. '>') may merge with the next character,
                # so "prefix trigger + tail completion" is preferred.
                if len(self.none_recent) >= len(es) and self.none_recent[-len(es):] == es:
                    # Canonical: the complete marker is already in the context ('>' did not merge), enter directly.
                    self.none_recent = []
                    self.phase = Phase.ACTION_NAME
                elif len(self.none_recent) >= len(es) - 1 and self.none_recent[-(len(es) - 1):] == es[:-1]:
                    # Only the marker prefix appeared (missing the last token): enter early -- force-write
                    # the missing tail token, and after it drains enter ACTION_NAME officially via
                    # after=("enter_action",). The tail token the model was supposed to generate ('>' or a
                    # merged '>x') gets blocked and discarded by the ACTION_NAME mask.
                    # Force-written tail tokens do NOT go into raw_token_ids.
                    self.none_recent = []
                    self.pending = [es[-1]]
                    self.after = ("enter_action",)
            return [] if raw_id >= V else [raw_id]

        # (c) ACTION_NAME
        if self.phase is Phase.ACTION_NAME:
            name = self.cfg["id_to_str"][raw_id]
            self.action_name = name
            self.n_params = len(self.cfg["surface_form"][name]["params"])
            seq = list(self.cfg["id_to_seq"][raw_id])
            first = seq[0] if seq else None
            rest = seq[1:]
            if self.n_params == 0:
                self.pending = rest + self.cfg["surface_form"][name]["closing_ids"]
                self.after = ("finish",)
            else:
                self.pending = rest
                self.after = ("begin_param", 0)
            if not self.pending and self.after is not None:
                self._run_after()
            return [first] if first is not None else []

        # (d) ARGUMENT_VALUE: free value over the full vocabulary, '\n' terminated (no more PARAM_PICK / VALUE_END extended ids)
        if self.phase is Phase.ARGUMENT_VALUE:
            if raw_id in self._exit_value_end_set:
                # The model emitted the exit marker's first token directly (e.g. the '</' of </Action>):
                # treat it as the end of the value, but **do not write** that token (finish will
                # force-write the complete exit </Action>, which would otherwise produce '</</Action>').
                # Write surface_form_suffix, then advance/finish. A cold-start model often emits </Action>
                # rather than '\n', and this branch is the fallback for that.
                self.pending = list(self._suffix_ids())
                self._set_after_for_next()
                if not self.pending and self.after is not None:
                    self._run_after()
                return []
            if raw_id in self._value_end_set:
                # Got '\n': **write it into the context** (to stay aligned with the decisions -- the
                # training-side trace picks up '\n' at that position, otherwise it would be mismatched
                # against surface_form_suffix's first token). Then force-write surface_form_suffix (the separator
                # between params / the terminator of the last one) and advance to the next param / finish.
                self.pending = list(self._suffix_ids())
                self._set_after_for_next()
                if not self.pending and self.after is not None:
                    self._run_after()
            return [raw_id]  # a value content token, or the terminating '\n' (both are written into the context)

        raise ValueError(self.phase)

    # ---- internals ----
    def _suffix_ids(self):
        return self.cfg["surface_form"][self.action_name]["params"][self.argument_index]["surface_form_suffix_ids"]

    def _set_after_for_next(self):
        nxt = self.argument_index + 1
        self.after = ("begin_param", nxt) if nxt < self.n_params else ("finish",)

    def _run_after(self):
        act = self.after
        self.after = None
        if act[0] == "finish":
            self.phase = Phase.NONE
            self.action_name = None
            self.argument_index = 0
            self.n_params = 0
            self.none_recent = []
            # Matches ALFWorld's `</Action>` + eos closing semantics: once the action surface form is
            # complete, first force-write the exit marker (<|FunctionCallEnd|>; skipped when
            # exit_surface_form_seq is empty), then force eos, so the assistant turn is explicitly closed.
            # These positions are forced tokens and do NOT enter raw_token_ids; the training-side
            # trace replays with the same router, so their seq_mask=False.
            self.pending = list(self._exit_surface_form_seq) + list(self.cfg.get("turn_end_ids", []))
        elif act[0] == "enter_action":
            # Prefix-trigger tail completion is done: officially enter ACTION_NAME (the next step is
            # masked to the action-name extended ids).
            self.phase = Phase.ACTION_NAME
            self.none_recent = []
        elif act[0] == "begin_param":
            self._begin_param(act[1])

    def _begin_param(self, i):
        self.argument_index = i
        # Every param uniformly takes ARGUMENT_VALUE: free generation over the full vocabulary, '\n'
        # terminated. Value opening delimiters are no longer pre-written.
        self.phase = Phase.ARGUMENT_VALUE
        self.pending = []
        self.after = None

