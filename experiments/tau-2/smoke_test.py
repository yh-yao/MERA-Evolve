"""One-task smoke test: local 1.5B as tau2 agent, local 3B as user simulator.

Run with the tau2_stage2 venv (has tau2 + core.schemas importable):
  /shared_home/yuhang.yao/router-skills-evolve/.venv_tau2/bin/python3 \
    experiments/tau-2/smoke_test.py
"""
import json
import sys

sys.path.insert(0, "/shared_home/yuhang.yao/router-skills-evolve/tau2_stage2/code")

from adapters.tau2_bench.adapter import Tau2BenchAdapter
from core.schemas.artifacts import LLMSpec, RunTaskConfig

VENDOR_ROOT = "/shared_home/yuhang.yao/router-skills-evolve/tau2_stage2/code/vendor/tau2-bench"
DOMAIN = "airline"
TASK_ID = "0"

agent = LLMSpec(model="openai/evol-llm-agent", args={"api_base": "http://127.0.0.1:8200/v1", "api_key": "EMPTY"})
user = LLMSpec(model="openai/evol-llm-user", args={"api_base": "http://127.0.0.1:8201/v1", "api_key": "EMPTY"})

adapter = Tau2BenchAdapter(vendor_root=VENDOR_ROOT, domain=DOMAIN)
tasks = adapter.load_tasks(DOMAIN)
task = next(t for t in tasks if str(t["id"]) == TASK_ID)
print(f"[smoke] loaded task {DOMAIN}/{TASK_ID}")

config = RunTaskConfig(agent=agent, user=user, seed=300, max_steps=30, max_errors=10)
result = adapter.run_task(task, config, domain=DOMAIN)

print(f"[smoke] passed={result.passed} reward={result.reward} termination={result.termination_reason}")
print(f"[smoke] n_agent_steps={len(result.steps)} agent_cost=${result.agent_cost_usd:.4f} user_cost=${result.user_cost_usd:.4f}")
print(f"[smoke] system_prompt len={len(result.system_prompt)} n_tools={len(result.tools)}")
if result.steps:
    print(f"[smoke] first step response: {json.dumps(result.steps[0].response)[:300]}")
