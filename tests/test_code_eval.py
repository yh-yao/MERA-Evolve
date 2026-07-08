from verl_code_rl.code_eval import extract_code, run_code_tests, score_solution


TASK = {
    "task_id": "toy/0",
    "prompt": "def add_one(x: int) -> int:\n",
    "entry_point": "add_one",
    "test": "def check(candidate):\n    assert candidate(1) == 2\n    assert candidate(-1) == 0\n",
}


def test_extract_code_from_markdown():
    text = "```python\ndef add_one(x: int) -> int:\n    return x + 1\n```"
    assert extract_code(text, "add_one").startswith("def add_one")


def test_run_code_tests_passes():
    ok, error = run_code_tests(TASK, "def add_one(x: int) -> int:\n    return x + 1\n")
    assert ok, error


def test_score_solution_fails_and_passes():
    assert score_solution("def add_one(x):\n    return x\n", TASK) == 0.0
    assert score_solution("def add_one(x):\n    return x + 1\n", TASK) == 1.0
