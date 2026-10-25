#!/usr/bin/env bash
# Run tau2_evolve.collect_traces in chunks, restarting the student and user
# vLLM servers between chunks.
#
# Long-lived Qwen3.5 servers drift into a GDN "zero-throughput" state: 99%
# GPU utilisation, 0-1 running requests, clients blocked. A k=8 TRAIN
# collection crawled at ~30 rollouts/hour overnight because of it. Bounding
# each server lifetime to one chunk avoids that state; --request-timeout
# turns any request that still hangs into an errored row that the next
# chunk's --resume retries.
#
# Usage:
#   SERVERS_DIR=/abs/dir BASE_MODEL=... USER_MODEL_PATH=... VLLM_BIN=... \
#   AGENT_GPU=7 USER_GPU=4 [AGENT_PORT=8280 USER_PORT=8281] [AGENT_ADAPTER=dir] \
#   [CHUNK_JOBS=128] [REQUEST_TIMEOUT=900] \
#     chunked_collect.sh --out run.jsonl <collect_traces args without --resume>
#
# The agent/user base URLs are added from AGENT_PORT/USER_PORT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
TAU2_WORKSPACE="${TAU2_WORKSPACE:-$(cd "$ROOT/.." && pwd)/router-skills-evolve}"
TAU2_PY="${TAU2_PYTHON:-$TAU2_WORKSPACE/.venv_tau2/bin/python3}"
SERVERS_DIR="${SERVERS_DIR:?set SERVERS_DIR (absolute; holds server pid files and logs)}"
AGENT_PORT="${AGENT_PORT:-8280}" USER_PORT="${USER_PORT:-8281}"
CHUNK_JOBS="${CHUNK_JOBS:-128}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
MAX_CHUNKS="${MAX_CHUNKS:-1000}"
export TAU2_WORKSPACE AGENT_PORT USER_PORT
export PYTHONPATH="$EXPERIMENT_DIR:$EXPERIMENT_DIR/compat:${PYTHONPATH:-}"
mkdir -p "$SERVERS_DIR"
log() { echo "[$(date -u +%FT%TZ)] [chunked_collect] $*" >&2; }

pid_on_port() {
  ss -ltnpH "sport = :$1" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2
}

stop_servers() {
  # Find servers by port, not by pid file: a server may have been started by
  # another run directory (e.g. the shared user simulator). Only processes
  # serving our evol-llm-* names are ever stopped.
  local port pid
  for port in "$AGENT_PORT" "$USER_PORT"; do
    pid="$(pid_on_port "$port")"
    [[ -n "$pid" ]] || continue
    if ! tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -qE -- '--served-model-name evol-llm-(agent|user)'; then
      log "port $port is held by a process that is not ours (pid $pid); leaving it"
      continue
    fi
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 90); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  done
  rm -f "$SERVERS_DIR"/evol-llm-*_server.pid
  sleep 5  # let the GPU memory drain before the next server binds
}

start_servers() {
  # 08 with BUCKETS="" only (re)starts servers that are not already serving.
  RESULTS_DIR="$SERVERS_DIR" BUCKETS="" bash "$EXPERIMENT_DIR/08_phase0_eval_protocol.sh" >&2
}

chunk=0
while (( chunk < MAX_CHUNKS )); do
  chunk=$((chunk + 1))
  start_servers
  summary="$("$TAU2_PY" -m tau2_evolve.collect_traces "$@" --resume \
      --max-jobs "$CHUNK_JOBS" --request-timeout "$REQUEST_TIMEOUT" \
      --agent-base-url "http://127.0.0.1:$AGENT_PORT/v1" \
      --user-base-url "http://127.0.0.1:$USER_PORT/v1" 2>&1 \
      | tee -a "$SERVERS_DIR/chunked_collect.log" | grep -E '^\[collect\] (bucket=|done)' || true)"
  remaining="$(sed -n 's/.* remaining=\([0-9]*\) this_chunk=\([0-9]*\).*/\1 \2/p' <<<"$summary" | tail -1)"
  read -r pending this_chunk <<<"${remaining:-0 0}"
  log "chunk $chunk: pending_before=$pending ran=$this_chunk"
  # Restart after every chunk; the last one leaves fresh servers for callers.
  stop_servers
  if (( pending <= this_chunk )); then
    # Everything was attempted; a final pass only retries errored rows.
    start_servers
    final="$("$TAU2_PY" -m tau2_evolve.collect_traces "$@" --resume --max-jobs 0 \
        --request-timeout "$REQUEST_TIMEOUT" \
        --agent-base-url "http://127.0.0.1:$AGENT_PORT/v1" \
        --user-base-url "http://127.0.0.1:$USER_PORT/v1" 2>&1 \
        | tee -a "$SERVERS_DIR/chunked_collect.log" | grep -E '^\[collect\] bucket=' || true)"
    left="$(sed -n 's/.* remaining=\([0-9]*\).*/\1/p' <<<"$final" | tail -1)"
    if [[ "${left:-1}" == "0" ]]; then
      log "complete"
      exit 0
    fi
    log "final pass retried $left errored jobs; continuing"
  fi
done
log "FATAL: gave up after $MAX_CHUNKS chunks"
exit 1
