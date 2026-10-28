"""Sanity checks for AReaL RL tasks in our tau2 (CPU only).

1. DB switch: a lookup from an AReaL task's first gold action returns different
   data under the task's snapshot than under tau2's default DB (or fails there).
2. Gold replay: replaying each task's gold assistant actions as a trajectory gets
   reward 1 from tau2's environment (DB-state) evaluator under the task's DB.

Usage (tau2 venv): python check_areal_tasks.py [--per-domain 40]
"""
import argparse, collections, json, random, sys

sys.path.insert(0, "experiments/tau-2")
from tau2_evolve import areal_tasks, benchmark  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--per-domain", type=int, default=40)
ap.add_argument("--write-valid", help="check every task and write the ids whose gold replay scores 1")
args = ap.parse_args()

from tau2.data_model.message import AssistantMessage, ToolCall  # noqa: E402
from tau2.data_model.tasks import Task  # noqa: E402
from tau2.evaluator.evaluator_env import EnvironmentEvaluator  # noqa: E402
from tau2.registry import registry  # noqa: E402

areal_tasks.install()
print("held-out telecom tasks excluded:", sorted(areal_tasks.heldout_telecom_ids()))
ids = areal_tasks.task_ids()
print("usable tasks:", collections.Counter(d for d, _ in ids))
rng = random.Random(0)
valid = []
out = collections.Counter()
for dom in ("airline", "retail", "telecom"):
    pool = [t for d, t in ids if d == dom]
    sample = pool if args.write_valid else rng.sample(pool, min(args.per_domain, len(pool)))
    for tid in sample:
        task = Task.model_validate(benchmark.load_task(dom, tid))
        acts = [a for a in (task.evaluation_criteria.actions or []) if a.requestor == "assistant"]
        # 1) DB switch, on the first lookup-style action
        if dom != "telecom" and acts:
            a = acts[0]
            def call(env):
                try:
                    return str(env.get_response(ToolCall(id="x", name=a.name, arguments=a.arguments, requestor="assistant")).content)[:200]
                except Exception as e:  # noqa: BLE001
                    return f"ERR {e}"
            default = call(registry.get_env_constructor(dom)())
            with areal_tasks.use_db(tid):
                snap = call(registry.get_env_constructor(dom)())
            out[f"{dom}_db_differs"] += default != snap
            out[f"{dom}_db_checked"] += 1
        # 2) gold replay -> DB-state reward
        traj = []
        with areal_tasks.use_db(tid):
            env = registry.get_env_constructor(dom)()
            if task.initial_state is not None:
                env.set_state(initialization_data=task.initial_state.initialization_data,
                              initialization_actions=task.initial_state.initialization_actions, message_history=[])
            from tau2.data_model.message import UserMessage
            for i, a in enumerate(task.evaluation_criteria.actions or []):  # assistant and user (device) actions
                tc = ToolCall(id=f"g{i}", name=a.name, arguments=a.arguments, requestor=a.requestor)
                cls = AssistantMessage if a.requestor == "assistant" else UserMessage
                traj.append(cls(role=a.requestor, content=None, tool_calls=[tc]))
                traj.append(env.get_response(tc))
            try:
                info = EnvironmentEvaluator.calculate_reward(
                    environment_constructor=registry.get_env_constructor(dom), task=task, full_trajectory=traj)
                ok = float(info.reward) == 1.0
            except Exception as e:  # noqa: BLE001
                ok = False
                out[f"{dom}_replay_error"] += 1
                if out[f"{dom}_replay_error"] <= 2:
                    print("replay error", tid, repr(e)[:300])
        out[f"{dom}_replay_ok"] += ok
        if ok:
            valid.append(tid)
        out[f"{dom}_replay_n"] += 1
print(json.dumps(dict(sorted(out.items())), indent=1))
if args.write_valid:
    json.dump(sorted(valid), open(args.write_valid, "w"))
    print("wrote", len(valid), "valid task ids to", args.write_valid)
