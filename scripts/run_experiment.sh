#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# One public launcher for every benchmark. Benchmark-specific code remains in
# experiments/<benchmark>; this file owns naming, discovery, and dispatch.
declare -A RECIPES=(
  ["humaneval_mbpp:skills"]="experiments/humaneval_mbpp/01_skills_only.sh"
  ["humaneval_mbpp:sft"]="experiments/humaneval_mbpp/02_sft_only.sh"
  ["humaneval_mbpp:grpo"]="experiments/humaneval_mbpp/03_grpo_only.sh"
  ["humaneval_mbpp:4cycle-sft"]="experiments/humaneval_mbpp/04_4cycle_sft.sh"
  ["humaneval_mbpp:4cycle-sft-grpo"]="experiments/humaneval_mbpp/05_4cycle_sft_grpo.sh"
  ["tau2:skills"]="experiments/tau-2/01_skills_only.sh"
  ["tau2:sft"]="experiments/tau-2/02_sft_only.sh"
  ["tau2:grpo"]="experiments/tau-2/03_grpo_only.sh"
  ["tau2:sft-grpo"]="experiments/tau-2/04_sft_grpo.sh"
  ["tau2:fallback-smoke"]="experiments/tau-2/05_fallback_4b_smoke.sh"
)

usage() {
  cat <<'EOF'
Usage:
  scripts/run_experiment.sh --list
  scripts/run_experiment.sh BENCHMARK RECIPE [--dry-run] [-- SCRIPT_ARGS...]

Benchmarks:
  humaneval_mbpp   HumanEval + MBPP single-turn code generation
  tau2             tau2-bench multi-turn tool-use customer service

Configuration is supplied through environment variables so exact runs are
easy to reproduce in shell scripts and schedulers. See CLAUDE.md for setup and
complete examples.
EOF
}

list_recipes() {
  printf '%-18s %-22s %s\n' "BENCHMARK" "RECIPE" "SCRIPT"
  for key in "${!RECIPES[@]}"; do
    printf '%-18s %-22s %s\n' "${key%%:*}" "${key#*:}" "${RECIPES[$key]}"
  done | sort
}

if [[ "${1:-}" == "--list" ]]; then
  list_recipes
  exit 0
fi
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

benchmark="$1"
recipe="$2"
shift 2
[[ "$benchmark" == "tau-2" ]] && benchmark="tau2"
[[ "$benchmark" == "code" ]] && benchmark="humaneval_mbpp"

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
  shift
fi
[[ "${1:-}" == "--" ]] && shift

key="$benchmark:$recipe"
script="${RECIPES[$key]:-}"
if [[ -z "$script" ]]; then
  echo "Unknown experiment: $key" >&2
  echo "Run scripts/run_experiment.sh --list to see valid combinations." >&2
  exit 2
fi
if [[ ! -f "$script" ]]; then
  echo "Registered experiment script is missing: $script" >&2
  exit 2
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

echo "[experiment] benchmark=$benchmark recipe=$recipe script=$script"
if (( dry_run )); then
  printf '[experiment] command: bash %q' "$script"
  if (( $# )); then
    printf ' %q' "$@"
  fi
  printf '\n'
  exit 0
fi

exec bash "$script" "$@"
