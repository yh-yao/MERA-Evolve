from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_experiment.sh"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_launcher_lists_both_benchmarks() -> None:
    result = _run("--list")
    assert result.returncode == 0
    assert "humaneval_mbpp" in result.stdout
    assert "tau2" in result.stdout
    assert "sft-grpo" in result.stdout


def test_launcher_dry_run_resolves_aliases() -> None:
    tau2 = _run("tau-2", "sft-grpo", "--dry-run")
    code = _run("code", "skills", "--dry-run")
    assert tau2.returncode == 0
    assert "experiments/tau-2/04_sft_grpo.sh" in tau2.stdout
    assert code.returncode == 0
    assert "experiments/humaneval_mbpp/01_skills_only.sh" in code.stdout


def test_launcher_rejects_unknown_recipe() -> None:
    result = _run("tau2", "missing", "--dry-run")
    assert result.returncode == 2
    assert "Unknown experiment" in result.stderr
