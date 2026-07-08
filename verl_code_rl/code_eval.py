"""Utilities for HumanEval/MBPP-style code extraction and execution.

The public surface is intentionally tiny because this module is imported by the
verl reward function. Keep dependencies stdlib-only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any


def extract_code(response: str, entry_point: str = "", original_prompt: str = "") -> str:
    """Extract executable Python from a model response."""
    text = response or ""
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.find("```", start)
        text = text[start: end if end != -1 else len(text)]
    elif "```" in text:
        start = text.index("```") + 3
        end = text.find("```", start)
        text = text[start: end if end != -1 else len(text)]
    code = text.strip()

    for marker in ("\n### Instruction:", "\n### Response:", "\nExplanation:", "\nThe code"):
        idx = code.find(marker)
        if idx > 0:
            code = code[:idx].strip()

    starts = []
    if entry_point:
        starts.append(code.find(f"def {entry_point}"))
    starts.extend([code.find("import "), code.find("from ")])
    starts = [idx for idx in starts if idx >= 0]
    if starts:
        code = code[min(starts):].strip()

    if entry_point and f"def {entry_point}" in code:
        return code

    if original_prompt:
        body = []
        for line in code.splitlines():
            if line.strip() and not line.startswith((" ", "\t")):
                body.append("    " + line)
            else:
                body.append(line)
        return original_prompt + "\n".join(body)

    return code


def load_ground_truth(ground_truth: Any, extra_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Decode the task payload stored in verl's reward_model.ground_truth field."""
    if isinstance(ground_truth, dict):
        task = dict(ground_truth)
    elif isinstance(ground_truth, str):
        task = json.loads(ground_truth)
    else:
        raise TypeError(f"unsupported ground_truth type: {type(ground_truth).__name__}")
    if extra_info:
        task.setdefault("task_id", extra_info.get("task_id", ""))
        task.setdefault("dataset", extra_info.get("dataset", ""))
    return task


def run_code_tests(task: dict[str, Any], generated_code: str, timeout: int = 10) -> tuple[bool, str]:
    """Run a generated solution against a HumanEval/MBPP test string."""
    entry_point = task["entry_point"]
    test_code = task["test"]
    prompt = task.get("prompt", "") or ""
    prompt_imports = "\n".join(
        line for line in prompt.splitlines()
        if line.lstrip().startswith(("import ", "from "))
    )
    header = (
        "from typing import *\n"
        "import math, re, collections, itertools, functools, heapq, bisect, string\n"
        f"{prompt_imports}\n"
    )
    program = f"""
{header}
{generated_code}

{test_code}

check({entry_point})
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-"],
            input=program,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"

    if proc.returncode == 0:
        return True, ""
    lines = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, lines[-1][:200] if lines else f"exit {proc.returncode}"


def score_solution(solution_str: str, ground_truth: Any, extra_info: dict[str, Any] | None = None) -> float:
    """Return 1.0 when the generated code passes all tests, otherwise 0.0."""
    task = load_ground_truth(ground_truth, extra_info=extra_info)
    code = extract_code(solution_str, task.get("entry_point", ""), task.get("prompt", ""))
    ok, _ = run_code_tests(task, code, timeout=int(task.get("timeout", 10)))
    return 1.0 if ok else 0.0
