"""Lightweight procedural skills for the MERA evolve loop.

Skills are bucketed one-per-dataset (HumanEval, MBPP) -- each dataset has a
distinct task shape (docstring-driven vs. assert-driven), so a single shared
procedure would blur two genuinely different sets of solving advice. Each
bucket's procedure is a written, dataset-specific description of how to solve
that shape of task, not a tally of frequently-seen helper-function names.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def extract_signature(prompt: str, dataset: str = "") -> str:  # noqa: ARG001
    """Bucket a task by its dataset. Unknown/missing datasets fall back to "coding"."""
    dataset = (dataset or "").strip().lower()
    return dataset if dataset in ("humaneval", "mbpp") else "coding"


_STATIC_SKILL_SECTIONS: dict[str, dict[str, Any]] = {
    "humaneval": {
        "when_to_use": (
            "Task is a HumanEval-style problem: a complete function signature and "
            "docstring, often with worked examples written as doctests."
        ),
        "procedure": [
            "Read the docstring's examples carefully -- they define the exact "
            "expected behavior, including edge cases.",
            "Keep the given function name and signature exactly as written.",
            "Handle boundary cases explicitly: empty input, negative numbers, "
            "single-element input, boundary values.",
            "Return one self-contained function; do not add extra top-level code, "
            "explanations, or tests.",
            "Prefer a direct, correct implementation over a clever one-liner.",
        ],
    },
    "mbpp": {
        "when_to_use": (
            "Task is an MBPP-style problem: the goal is a plain-English sentence, "
            "and the exact function name/signature is only implied by the assert "
            "statements that follow the prompt."
        ),
        "procedure": [
            "Infer the exact function name and argument order from the assert "
            "statements before writing any code.",
            "Solutions are typically short, direct algorithms (loops, basic "
            "string/list operations); avoid over-engineering or unnecessary "
            "abstractions.",
        ],
    },
}
_FALLBACK_SECTIONS: dict[str, Any] = {
    "when_to_use": "",
    "procedure": ["Return only complete Python code. Prefer compact, direct solutions."],
}


def _render_sections(
    signature: str,
    when_to_use: str,
    procedure: list[str],
    pitfalls: list[str],
    patterns: list[str],
    max_chars: int = 4000,
) -> str:
    """Render a skill like a real runbook: distinct, independently scannable
    sections instead of one paragraph the model has to parse unstructured."""
    parts = [f"# Skill: {signature}"]
    if when_to_use:
        parts.append(f"## When to use\n{when_to_use}")
    if procedure:
        steps = "\n".join(f"{i}. {step}" for i, step in enumerate(procedure, 1))
        parts.append(f"## Procedure\n{steps}")
    if pitfalls:
        bullets = "\n".join(f"- {p}" for p in pitfalls)
        parts.append(f"## Common pitfalls\n{bullets}")
    if patterns:
        bullets = "\n".join(f"- {p}" for p in patterns)
        parts.append(f"## Recurring patterns\n{bullets}")

    # Truncate at whole-section boundaries rather than mid-sentence.
    kept: list[str] = []
    budget = max_chars
    for part in parts:
        cost = len(part) + (2 if kept else 0)  # "\n\n" join separator
        if cost > budget:
            break
        kept.append(part)
        budget -= cost
    return "\n\n".join(kept)


def _select_examples(exemplars: list[dict[str, Any]], k: int | None = None) -> list[dict[str, str]]:
    if k is None:
        # A global skill must not inject arbitrary solved programs into every
        # task.  Keep examples persisted for inspection/retrieval, but opt out
        # of prompt injection unless an experiment explicitly enables it.
        k = int(os.environ.get("SKILL_FEWSHOT_K", "0"))
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


def _parse_pitfalls_and_patterns(text: str) -> tuple[list[str], list[str]]:
    pitfalls: list[str] = []
    patterns: list[str] = []
    bucket: list[str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("PITFALLS"):
            bucket = pitfalls
            continue
        if upper.startswith("PATTERNS"):
            bucket = patterns
            continue
        if bucket is not None and stripped.startswith(("-", "*")):
            item = stripped.lstrip("-*").strip()
            if item:
                bucket.append(item[:400])
    return pitfalls[:4], patterns[:4]


def make_llm_distiller(model: str, base_url: str, api_key: str, max_examples: int = 40):
    """Return a bounded, incremental distiller of `pitfalls`/`patterns` sections.

    It is deliberately optional and deliberately narrow: the `when_to_use`/
    `procedure` sections are fixed, hand-written per dataset
    (`_STATIC_SKILL_SECTIONS`) and never touched here -- only the
    exemplar-grounded `pitfalls`/`patterns` sections are LLM-distilled, so a
    bad distillation can't corrupt the always-present static scaffolding.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    def distill(
        signature: str,
        exemplars: list[dict[str, Any]],
        previous_pitfalls: list[str],
        previous_patterns: list[str],
    ) -> tuple[list[str], list[str]]:
        examples = exemplars[-max(1, max_examples):]
        rendered = "\n\n".join(
            "Problem:\n```python\n" + str(item.get("prompt", ""))[:400]
            + "\n```\nSolution:\n```python\n" + str(item.get("completion", ""))[:500] + "\n```"
            for item in examples
        )
        prompt = (
            f"From these successful solutions to a `{signature}` coding skill, extract "
            "concrete, transferable lessons in exactly this format:\n\n"
            "PITFALLS:\n- <a specific trap or gotcha for this task shape>\n"
            "PATTERNS:\n- <a recurring algorithmic pattern worth reusing>\n\n"
            "At most 4 pitfalls and 4 patterns. Be specific and transferable -- no "
            "task-specific names, no generic advice, no full solutions.\n\n"
        )
        if previous_pitfalls or previous_patterns:
            prompt += (
                "Previous pitfalls (keep the ones still useful, refine or replace others):\n"
                + "\n".join(f"- {p}" for p in previous_pitfalls) + "\n"
                "Previous patterns:\n" + "\n".join(f"- {p}" for p in previous_patterns) + "\n\n"
            )
        prompt += f"Successful traces:\n{rendered}"
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                # Reasoning-capable models (e.g. GPT-5.5) spend part of
                # max_tokens on hidden reasoning tokens before any visible
                # text -- 500 was enough for the visible answer alone but not
                # for reasoning + answer, so real calls came back empty
                # (finish_reason="length") with zero usable content.
                max_tokens=3000,
            )
            text = (response.choices[0].message.content or "").strip()
            pitfalls, patterns = _parse_pitfalls_and_patterns(text)
            if pitfalls or patterns:
                return pitfalls, patterns
        except Exception:  # noqa: BLE001 - keep previous sections rather than fabricate
            pass
        return previous_pitfalls, previous_patterns

    return distill


class Skill:
    """Success statistics plus successful exemplars for one signature."""

    def __init__(self, signature: str):
        self.signature = signature
        self.stats: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.history: list[dict[str, Any]] = []
        self.exemplars: list[dict[str, str]] = []
        self.pitfalls: list[str] = []
        self.patterns: list[str] = []
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

    def distill(self, distiller=None) -> str:
        if not self.exemplars:
            return self.procedure
        sections = _STATIC_SKILL_SECTIONS.get(self.signature, _FALLBACK_SECTIONS)
        if distiller is None:
            # No LLM configured: pitfalls/patterns stay whatever they already
            # were (empty on a fresh skill) rather than being fabricated from
            # exemplar statistics.
            self.procedure_source = "heuristic"
        else:
            try:
                self.pitfalls, self.patterns = distiller(
                    self.signature, self.exemplars, self.pitfalls, self.patterns,
                )
                self.procedure_source = "llm"
            except Exception:  # noqa: BLE001 - static sections still render below
                self.procedure_source = "heuristic"
        self.procedure = _render_sections(
            self.signature, sections["when_to_use"], sections["procedure"],
            self.pitfalls, self.patterns,
        )
        self.examples = _select_examples(self.exemplars)
        return self.procedure

    def can_downgrade_to_small(self, small_model: str, min_rate: float = 0.8,
                               min_samples: int = 1) -> bool | None:
        successes, total = self.stats[small_model]
        if total < min_samples:
            return None
        return successes / total >= min_rate

    def to_dict(self) -> dict[str, Any]:
        # `procedure` is deliberately excluded: it's a deterministic rendering
        # of (signature, pitfalls, patterns) via `_render_sections`, not a
        # separate source of truth -- see SkillBook.save()/skill.md.
        return {
            "signature": self.signature,
            "stats": {key: list(value) for key, value in self.stats.items()},
            "history": self.history,
            "exemplars": self.exemplars,
            "pitfalls": self.pitfalls,
            "patterns": self.patterns,
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
        skill.pitfalls = list(data.get("pitfalls", []))
        skill.patterns = list(data.get("patterns", []))
        skill.procedure_source = str(data.get("procedure_source", ""))
        skill.examples = list(data.get("examples", []))
        sections = _STATIC_SKILL_SECTIONS.get(skill.signature, _FALLBACK_SECTIONS)
        skill.procedure = _render_sections(
            skill.signature, sections["when_to_use"], sections["procedure"],
            skill.pitfalls, skill.patterns,
        )
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
               completion: str = "", dataset: str = "") -> None:
        skill = self.get_or_create(extract_signature(prompt, dataset))
        skill.update(model_id, success, task_id=task_id, completion=completion, prompt=prompt)

    def update_from_trace(self, trace: dict[str, Any], small_model: str = "small",
                          large_model: str = "large") -> None:
        prompt = str(trace.get("prompt") or "")
        task_id = str(trace.get("task_id") or "")
        dataset = str(trace.get("dataset") or "")
        self.update(prompt, small_model, bool(trace.get("small_success")), task_id,
                    str(trace.get("small_completion") or ""), dataset=dataset)
        if not trace.get("large_skipped", False):
            self.update(prompt, large_model, bool(trace.get("large_success")), task_id,
                        str(trace.get("large_completion") or ""), dataset=dataset)

    def get_procedure(self, prompt: str, dataset: str = "") -> str:
        skill = self.skills.get(extract_signature(prompt, dataset))
        if not skill:
            return ""
        examples = _render_examples(skill.examples) if os.environ.get("SKILL_INCLUDE_EXAMPLES", "0") == "1" else ""
        if skill.procedure and examples:
            return f"{skill.procedure}\n\n{examples}"
        return skill.procedure or examples

    def distill_all(self, distiller=None) -> int:
        count = 0
        for skill in self.skills.values():
            if skill.distill(distiller=distiller):
                count += 1
        return count

    def save(self, path: str | Path) -> None:
        """Write `path` as a directory holding two files: `skill.md` (the real
        skill -- exactly the text that gets injected into prompts) and
        `skill_statistics.json` (stats/history/exemplars/pitfalls/patterns --
        the source data skill.md is deterministically rendered from)."""
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        rendered = [skill.procedure for skill in self.skills.values() if skill.procedure]
        (out_dir / "skill.md").write_text(
            ("\n\n---\n\n".join(rendered) + "\n") if rendered else "",
        )
        (out_dir / "skill_statistics.json").write_text(
            json.dumps({"skills": [s.to_dict() for s in self.skills.values()]}, indent=2),
        )

    def load(self, path: str | Path) -> None:
        src = Path(path) / "skill_statistics.json"
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
