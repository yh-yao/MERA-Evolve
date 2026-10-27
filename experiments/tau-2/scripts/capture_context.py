"""Capture the tau2 agent system prompt and OpenAI tool schemas per domain (what the eval endpoint sees)."""
import hashlib, json, sys
from tau2.data_model.simulation import TextRunConfig
from tau2.registry import registry
from tau2.runner import build_text_orchestrator

out = {}
for dom in ("airline", "retail", "telecom"):
    loader = registry.get_tasks_loader(dom)
    try:
        tasks = loader(None)
    except TypeError:
        tasks = loader()
    cfg = TextRunConfig(domain=dom, agent="llm_agent", llm_agent="openai/x", llm_args_agent={},
                        user="user_simulator", llm_user="openai/x", llm_args_user={}, seed=300,
                        max_steps=60, max_errors=10, num_trials=1)
    agent = build_text_orchestrator(cfg, tasks[0], seed=300).agent
    sp = str(agent.system_prompt)
    tools = [t.openai_schema for t in agent.tools]
    out[dom] = {"system_prompt": sp, "tools": tools}
    print(dom, len(sp), len(tools), hashlib.sha256(json.dumps([sp, tools], sort_keys=True).encode()).hexdigest()[:16])
json.dump(out, open(sys.argv[1], "w"))
