from verl_code_rl.prepare_data import convert_rows


def test_convert_rows_makes_verl_shape():
    rows = [{
        "task_id": "HumanEval/0",
        "prompt": "def f():\n",
        "entry_point": "f",
        "test": "def check(candidate):\n    assert candidate() is None\n",
        "split": "train",
        "_src": "humaneval",
    }]
    out = convert_rows(rows, "train")
    assert out[0]["data_source"] == "code/humaneval"
    assert out[0]["ability"] == "code"
    assert out[0]["prompt"][0]["role"] == "system"
    assert "ground_truth" in out[0]["reward_model"]
