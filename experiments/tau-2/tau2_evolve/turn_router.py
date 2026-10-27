"""Per-turn escalation from the student to a stronger teacher (improvement plan, Phase 3).

Unlike ``router.RoutedAgent`` (escalate once, the teacher keeps control for
the rest of the episode), ``TurnRoutedAgent`` decides at every agent turn and
hands control back to the student afterwards. Student and teacher are both
tau2 ``LLMAgent``s built for the same task, so they share one agent state:
whichever model answers a turn appends to the same message list.

Policies (``--turn-router`` in collect_traces):

``never``           student only; with ``--disagree-k`` it still logs the
                    agreement score of every turn (threshold calibration)
``always``          teacher only (upper bound)
``random:P``        escalate each turn with probability P
``tool_error``      escalate the turn right after a tool error
``disagree:T``      sample the student K more times at temperature 1; escalate
                    when the share of samples taking the greedy action is <= T
``oracle``          ask the teacher every turn; use its reply only where its
                    action differs from the student's (tool vs text, other
                    tool, other arguments). Not a deployable router (it pays
                    for a teacher call every turn), but its escalation rate and
                    pass rate bound what a learned router can reach, and on
                    TRAIN its overrides are multi-point DAgger corrections.

Every turn is logged on ``agent.turn_log`` (student and teacher action
signatures, the score, whether the teacher answered).
"""
from __future__ import annotations

import hashlib
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from tau2_evolve.collect_error_points import action_signature, divergence_kind

POLICIES = ("never", "always", "random", "tool_error", "disagree", "oracle")


@dataclass
class TurnRouterConfig:
    policy: str
    param: float | None = None
    teacher_spec: Any = None
    sampler_spec: Any = None  # student at temperature 1, for ``disagree``
    disagree_k: int = 0

    @classmethod
    def parse(cls, text: str, **kwargs) -> "TurnRouterConfig":
        name, _, value = text.partition(":")
        if name not in POLICIES:
            raise ValueError(f"unknown turn-router policy {name!r}; expected one of {POLICIES}")
        if name in ("random", "disagree") and not value:
            raise ValueError(f"turn-router policy {name} needs a parameter, e.g. {name}:0.2")
        return cls(policy=name, param=float(value) if value else None, **kwargs)

    @property
    def needs_teacher(self) -> bool:
        return self.policy != "never"


def _tool_error(message) -> bool:
    msgs = getattr(message, "tool_messages", None) or [message]
    return any(getattr(m, "role", None) == "tool" and getattr(m, "error", False) for m in msgs)


def _dump(message) -> dict:
    return message.model_dump(mode="json")


class TurnRoutedAgent:
    def __init__(self, student, teacher, sampler, config: TurnRouterConfig, rng_key: str):
        self.student = student
        self.teacher = teacher
        self.sampler = sampler
        self.config = config
        self.rng = random.Random(int(hashlib.sha256(rng_key.encode()).hexdigest()[:12], 16))
        self.turn_log: list[dict] = []

    def __getattr__(self, name: str):
        return getattr(self.student, name)

    def set_seed(self, seed: int) -> None:
        for agent in (self.student, self.teacher, self.sampler):
            if agent is not None and hasattr(agent, "set_seed"):
                agent.set_seed(seed)

    def _agreement(self, message, state, student_sig) -> float | None:
        k = self.config.disagree_k
        if not k or self.sampler is None:
            return None
        same = 0
        for _ in range(k):
            sample, _ = self.sampler.generate_next_message(deepcopy(message), deepcopy(state))
            same += action_signature(_dump(sample)) == student_sig
        return same / k

    def generate_next_message(self, message, state):
        policy, param = self.config.policy, self.config.param
        if policy == "always":
            reply, state = self.teacher.generate_next_message(message, state)
            self.turn_log.append({"turn": len(state.messages) - 1, "teacher": True})
            return reply, state

        s_reply, s_state = self.student.generate_next_message(deepcopy(message), deepcopy(state))
        s_dump = _dump(s_reply)
        s_sig = action_signature(s_dump)
        entry: dict[str, Any] = {"student_action": repr(s_sig)}
        escalate = False
        t_reply = None
        if policy == "random":
            escalate = self.rng.random() < param
        elif policy == "tool_error":
            escalate = _tool_error(message)
        elif policy in ("disagree", "never"):
            score = self._agreement(message, state, s_sig)
            entry["agreement"] = score
            escalate = policy == "disagree" and score is not None and score <= param
        elif policy == "oracle":
            t_reply, t_state = self.teacher.generate_next_message(deepcopy(message), deepcopy(state))
            kind = divergence_kind(s_dump, _dump(t_reply))
            entry["divergence"] = kind
            escalate = kind is not None
            if escalate:
                entry["student_message"] = s_dump
                entry["teacher_message"] = _dump(t_reply)
                # agent-visible history this turn answers (DAgger SFT context)
                entry["context_messages"] = [_dump(m) for m in s_state.messages[:-1]]

        if escalate:
            if t_reply is None:
                reply, state = self.teacher.generate_next_message(message, state)
            else:
                reply, state = t_reply, t_state
        else:
            reply, state = s_reply, s_state
        entry.update({"turn": len(state.messages) - 1, "teacher": escalate})
        self.turn_log.append(entry)
        return reply, state

    def summary(self) -> dict:
        turns = len(self.turn_log)
        escalated = sum(e["teacher"] for e in self.turn_log)
        return {"policy": self.config.policy, "param": self.config.param, "turns": turns,
                "teacher_turns": escalated, "teacher_share": escalated / turns if turns else 0.0,
                "log": self.turn_log}
