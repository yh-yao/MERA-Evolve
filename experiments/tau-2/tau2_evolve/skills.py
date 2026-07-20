"""Domain-bucketed SkillBook for tau2-bench, mirroring verl_code_rl/skills.py's
design (static When-to-use/Procedure + LLM-distilled Common-pitfalls/Recurring-
patterns from real successful exemplars) but bucketed by tau2 domain
(airline/retail/telecom) instead of humaneval/mbpp, and rendered as an
addition to the agent's <policy> block instead of a prepended user prompt.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from tau2_evolve import benchmark

DOMAINS = benchmark.DOMAINS

_STATIC_SECTIONS: dict[str, dict[str, str]] = {
    "airline": {
        "when_to_use": (
            "Task is an airline customer-service conversation: booking, modifying, or "
            "cancelling a flight reservation, or handling refunds/compensation."
        ),
        "procedure": (
            "1. Authenticate the user (locate their profile) before taking any action.\n"
            "2. Before booking, modifying, or cancelling anything, state the exact action "
            "details back to the user and get an explicit yes before calling the tool.\n"
            "3. Only call one tool at a time; never call a tool and message the user in the "
            "same turn.\n"
            "4. Only act on what the policy and available tools support -- never invent "
            "information, procedures, or subjective recommendations.\n"
            "5. If a request is outside policy scope, transfer to a human agent via "
            "transfer_to_human_agents, then send the required transfer message."
        ),
    },
    "retail": {
        "when_to_use": (
            "Task is a retail customer-service conversation: cancelling/modifying a pending "
            "order, returning/exchanging a delivered order, or updating the user's address."
        ),
        "procedure": (
            "1. Authenticate the user FIRST -- via user id, or via email, or via name + zip "
            "code -- even if the user already gave a user id.\n"
            "2. Confirm the exact action (which order, which items, refund method) before "
            "calling a mutating tool.\n"
            "3. Only call one tool at a time; never call a tool and message the user in the "
            "same turn.\n"
            "4. Only act on what the policy and available tools support."
        ),
    },
    "telecom": {
        "when_to_use": (
            "Task is a telecom customer-service conversation: technical support, overdue "
            "bill payment, line suspension, or plan changes."
        ),
        "procedure": (
            "1. Identify the customer and affected line, then verify that the line is active "
            "before device troubleshooting.\n"
            "2. For mobile-data issues, ask the user to run the policy's device diagnostics "
            "and follow the matching branch: turn mobile data on when disabled; when abroad, "
            "enable roaming on the line if needed and ask the user to turn device roaming on; "
            "turn Data Saver off; disconnect a problematic VPN; change a 2G/3G preference to "
            "4G/5G; or offer data refueling when the plan limit is exhausted.\n"
            "3. Device diagnostics and fixes (for example check_network_status, toggle_data, "
            "toggle_roaming, toggle_data_saver_mode, disconnect_vpn, and "
            "set_network_mode_preference) are actions the USER performs after your clear "
            "instruction. Do not emit them as agent-side tool calls or invent similarly named "
            "agent tools.\n"
            "4. Confirm account mutations before calling one available agent tool at a time; "
            "never call a tool and message the user in the same turn.\n"
            "5. Only act on what the policy and available tools support -- no invented "
            "procedures or subjective recommendations."
        ),
    },
}

DISTILL_SYSTEM_PROMPT = """You are distilling a customer-service agent skill from real \
successful conversation transcripts for the {domain} domain.

Given several transcripts of an agent that SUCCEEDED at its task, write exactly two \
labeled bullet lists grounded in what you actually observe in these transcripts:

## Common pitfalls
- 3-5 bullets: specific mistakes a weaker agent would make on tasks like these \
(wrong tool call order, skipping confirmation, wrong authentication method, etc.)

## Recurring patterns
- 3-5 bullets: specific successful strategies/tool-call sequences that recur across \
these transcripts.

Return ONLY the two labeled sections in markdown, nothing else."""


@dataclass
class Skill:
    domain: str
    when_to_use: str = ""
    procedure: str = ""
    pitfalls: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    exemplars: list[dict] = field(default_factory=list)

    def render(self) -> str:
        sections = [
            f"# Skill: {self.domain}",
            f"## When to use\n{self.when_to_use}",
            f"## Procedure\n{self.procedure}",
        ]
        if self.pitfalls:
            sections.append("## Common pitfalls\n" + "\n".join(f"- {p}" for p in self.pitfalls))
        if self.patterns:
            sections.append("## Recurring patterns\n" + "\n".join(f"- {p}" for p in self.patterns))
        return "\n\n".join(sections)


class SkillBook:
    def __init__(self) -> None:
        self.skills: dict[str, Skill] = {
            d: Skill(domain=d, **_STATIC_SECTIONS[d]) for d in DOMAINS
        }

    def add_exemplar(self, domain: str, trace_row: dict) -> None:
        if trace_row.get("passed") and benchmark.trace_action_complete(trace_row):
            self.skills[domain].exemplars.append(trace_row)

    def distill_all(self, distiller=None) -> None:
        for domain, skill in self.skills.items():
            if not skill.exemplars or distiller is None:
                continue
            pitfalls, patterns = distiller(domain, skill.exemplars)
            skill.pitfalls = pitfalls
            skill.patterns = patterns

    def as_dict(self) -> dict[str, str]:
        """{domain: rendered_skill_text} -- what collect_traces.py's --skillbook expects."""
        return {d: s.render() for d, s in self.skills.items()}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.as_dict(), indent=2, ensure_ascii=False))

    @staticmethod
    def load(path: Path) -> dict[str, str]:
        return json.loads(Path(path).read_text())


def make_llm_distiller(model: str, base_url: str, api_key: str):
    """Mirrors verl_code_rl/skills.py's make_llm_distiller: returns a callable
    distiller(domain, exemplars) -> (pitfalls: list[str], patterns: list[str]).
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    # LiteLLM uses the ``openai/`` prefix to select its provider, while a
    # directly served OpenAI-compatible endpoint expects the served model
    # name. Keep external provider model names intact.
    local_endpoint = "127.0.0.1" in base_url or "localhost" in base_url
    request_model = model.removeprefix("openai/") if local_endpoint else model

    def _distill(domain: str, exemplars: list[dict]) -> tuple[list[str], list[str]]:
        transcripts = []
        for ex in exemplars[:8]:
            msgs = ex.get("messages", [])
            lines = []
            for m in msgs:
                role = m.get("role", "?")
                content = m.get("content") or ""
                tool_calls = m.get("tool_calls") or []
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("name") or (tc.get("function") or {}).get("name")
                        args = tc.get("arguments") or (tc.get("function") or {}).get("arguments")
                        lines.append(f"{role} [tool_call {fn}]: {args}")
                if content:
                    lines.append(f"{role}: {content}")
            transcripts.append("\n".join(lines))
        joined = "\n\n---\n\n".join(transcripts)

        resp = client.chat.completions.create(
            model=request_model,
            messages=[
                {"role": "system", "content": DISTILL_SYSTEM_PROMPT.format(domain=domain)},
                {"role": "user", "content": joined[:20000]},
            ],
            max_tokens=1500,
        )
        text = resp.choices[0].message.content or ""
        return _parse_pitfalls_and_patterns(text)

    return _distill


def _parse_pitfalls_and_patterns(text: str) -> tuple[list[str], list[str]]:
    pitfalls: list[str] = []
    patterns: list[str] = []
    section = None
    for line in text.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("## common pitfalls"):
            section = "pitfalls"
            continue
        if low.startswith("## recurring patterns"):
            section = "patterns"
            continue
        if stripped.startswith("- ") and section:
            item = stripped[2:].strip()[:400]
            (pitfalls if section == "pitfalls" else patterns).append(item)
    return pitfalls, patterns
