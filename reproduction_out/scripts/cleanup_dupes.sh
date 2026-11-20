#!/usr/bin/env bash
# Kill the stale/duplicate teacher runs so nothing contends for GPUs 2-3.
for pat in 'chain_seed1300' 'chain_frontier' 'tau2_evolve.collect_traces'; do
  for p in $(pgrep -f "$pat"); do
    kill "$p" 2>/dev/null
  done
done
sleep 3
for s in teacherval teacherval1300 teacherfrontier tfix teacherrerun; do
  tmux kill-session -t "$s" 2>/dev/null
done
sleep 3
echo "cleaned"
tmux ls
