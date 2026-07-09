"""Lightweight procedural skills for the MERA evolve loop.

The first MERA version keeps one global skill bucket ("coding"). The skillbook
therefore does not route by itself; it only stores solved examples and distills
a stable prompt prefix that can be fed into the small model and verl prompts.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def extract_signature(prompt: str) -> str:  # noqa: ARG001
    """All code tasks share one global skill bucket."""
    return "coding"


_CODE_BLOCK = re.compile(r"```[\w+-]*\n(.*?)```", re.DOTALL)
_CALL = re.compile(r"\b([a-z_][a-z0-9_]{2,})\s*\(")
_SKIP_CALLS = {"if", "for", "while", "print", "return", "range", "len"}


def _select_examples(exemplars: list[dict[str, Any]], k: int | None = None) -> list[dict[str, str]]:
    if k is None:
        k = int(os.environ.get("SKILL_FEWSHOT_K", "2"))
    if k <= 0:
        return []
    pool = [ex for ex in exemplars if ex.get("prompt") and ex.get("completion")]
    chosen = sorted(pool, key=lambda ex: len(str(ex["completion"])))[:k]
    return [
        {
            "task_id": str(ex.get("task_id", "")),
            "prompt": str(ex["prompt"]).strip()[:500],
            "completion": str(ex["completion"]).strip()[:700],
        }
        for ex in chosen
    ]


def _render_examples(examples: list[dict[str, str]]) -> str:
    if not examples:
        return ""
    parts = ["## Worked examples"]
    for idx, ex in enumerate(examples, 1):
        parts.append(
            f"### Example {idx}\n"
            f"Problem:\n```python\n{ex.get('prompt', '')}\n```\n"
            f"Solution:\n```python\n{ex.get('completion', '')}\n```"
        )
    return "\n\n".join(parts)


def _heuristic_procedure(signature: str, exemplars: list[dict[str, Any]], max_chars: int = 1400) -> str:
    snippets: list[str] = []
    calls: list[str] = []
    for ex in exemplars:
        completion = str(ex.get("completion") or "")
        for block in _CODE_BLOCK.findall(completion):
            block = block.strip()
            if block and block not in snippets:
                snippets.append(block)
        for call in _CALL.findall(completion):
            if call not in _SKIP_CALLS and call not in calls:
                calls.append(call)

    lines = [
        f"# Procedure for `{signature}`",
        f"# distilled from {len(exemplars)} successful solution(s)",
        "",
        "Return only complete Python code. Prefer compact, direct solutions.",
    ]
    if calls:
        lines.extend(["", "Recurring helper/function patterns: " + ", ".join(calls[:12])])
    if snippets:
        lines.extend(["", "Reusable solution shape:", "```python", snippets[0][:650], "```"])
    else:
        sample = str(exemplars[0].get("completion") or "").strip() if exemplars else ""
        if sample:
            lines.extend(["", "Reference solution shape:", sample[:650]])
    return "\n".join(lines)[:max_chars]


class Skill:
    """Success statistics plus successful exemplars for one signature."""

    def __init__(self, signature: str):
        self.signature = signature
        self.stats: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.history: list[dict[str, Any]] = []
        self.exemplars: list[dict[str, str]] = []
        self.procedure = ""
        self.procedure_source = ""
        self.examples: list[dict[str, str]] = []

    @property
    def max_exemplars(self) -> int:
        return int(os.environ.get("SKILL_MAX_EXEMPLARS", "10000"))

    def update(self, model_id: str, success: bool, task_id: str = "", completion: str = "",
               prompt: str = "") -> None:
        self.stats[model_id][1] += 1
        if success:
            self.stats[model_id][0] += 1
        self.history.append({"task_id": task_id, "model": model_id, "success": bool(success)})
        if success and completion and prompt:
            self.exemplars.append({
                "task_id": task_id,
                "prompt": prompt,
                "completion": completion,
                "model": model_id,
            })
            if len(self.exemplars) > self.max_exemplars:
                self.exemplars = self.exemplars[-self.max_exemplars:]

    def distill(self) -> str:
        if not self.exemplars:
            return self.procedure
        self.procedure = _heuristic_procedure(self.signature, self.exemplars)
        self.procedure_source = "heuristic"
        self.examples = _select_examples(self.exemplars)
        return self.procedure

    def can_downgrade_to_small(self, small_model: str, min_rate: float = 0.8,
                               min_samples: int = 1) -> bool | None:
        successes, total = self.stats[small_model]
        if total < min_samples:
            return None
        return successes / total >= min_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "stats": {key: list(value) for key, value in self.stats.items()},
            "history": self.history,
            "exemplars": self.exemplars,
            "procedure": self.procedure,
            "procedure_source": self.procedure_source,
            "examples": self.examples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Skill":
        skill = cls(str(data["signature"]))
        skill.stats = defaultdict(lambda: [0, 0])
        for key, value in data.get("stats", {}).items():
            skill.stats[str(key)] = list(value)
        skill.history = list(data.get("history", []))
        skill.exemplars = list(data.get("exemplars", []))
        skill.procedure = str(data.get("procedure", ""))
        skill.procedure_source = str(data.get("procedure_source", ""))
        skill.examples = list(data.get("examples", []))
        return skill


class SkillBook:
    """Container for all skills."""

    def __init__(self) -> None:
        self.skills: dict[str, Skill] = {}

    def get_or_create(self, signature: str) -> Skill:
        if signature not in self.skills:
            self.skills[signature] = Skill(signature)
        return self.skills[signature]

    def update(self, prompt: str, model_id: str, success: bool, task_id: str = "",
               completion: str = "") -> None:
        skill = self.get_or_create(extract_signature(prompt))
        skill.update(model_id, success, task_id=task_id, completion=completion, prompt=prompt)

    def update_from_trace(self, trace: dict[str, Any], small_model: str = "small",
                          large_model: str = "large") -> None:
        prompt = str(trace.get("prompt") or "")
        task_id = str(trace.get("task_id") or "")
        self.update(prompt, small_model, bool(trace.get("small_success")), task_id,
                    str(trace.get("small_completion") or ""))
        if not trace.get("large_skipped", False):
            self.update(prompt, large_model, bool(trace.get("large_success")), task_id,
                        str(trace.get("large_completion") or ""))

    def get_procedure(self, prompt: str) -> str:
        skill = self.skills.get(extract_signature(prompt))
        if not skill:
            return ""
        examples = _render_examples(skill.examples)
        if skill.procedure and examples:
            return f"{skill.procedure}\n\n{examples}"
        return skill.procedure or examples

    def distill_all(self) -> int:
        count = 0
        for skill in self.skills.values():
            if skill.distill():
                count += 1
        return count

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"skills": [s.to_dict() for s in self.skills.values()]}, indent=2))

    def load(self, path: str | Path) -> None:
        src = Path(path)
        if not src.exists():
            return
        data = json.loads(src.read_text())
        self.skills = {
            skill.signature: skill
            for skill in (Skill.from_dict(item) for item in data.get("skills", []))
        }

    def summary(self) -> dict[str, Any]:
        return {
            "total_skills": len(self.skills),
            "total_observations": sum(sum(v[1] for v in s.stats.values()) for s in self.skills.values()),
            "signatures": sorted(self.skills),
        }
