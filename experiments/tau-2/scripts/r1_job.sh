# usage: r1_job.sh <gpu> <port_base> <bucket> <shard i/n> <out_dir>
# R1: 4B base, 4 trials per task at agent T=1.0 / user T=0, 40 steps (same as the frontier check),
# student + user simulator (both Qwen3.5-4B) on one GPU.
G=$1; P=$2; B=$3; SH=$4; O=$5
cd /mnt/local_data/users/luzheng/mera_tau2/MERA-Evolve
M=/mnt/data/users/luzheng/agent_rl/models
export TAU2_WORKSPACE=$PWD/../router-skills-evolve TAU2_PARTITION_PATH=$PWD/experiments/tau-2/data/partition_v2.json
export VLLM_BIN=/mnt/data/users/luzheng/agent_rl/.venv_vllm/bin/vllm GPU_MEMORY_UTILIZATION=0.42 VLLM_MAX_NUM_SEQS=128
export BASE_MODEL=$M/Qwen3.5-4B USER_MODEL_PATH=$M/Qwen3.5-4B AGENT_GPU=$G USER_GPU=$G AGENT_PORT=$P USER_PORT=$((P+1))
n=${SH//\//of}; mkdir -p $O/servers_${B}_$n
SERVERS_DIR=$O/servers_${B}_$n CHUNK_JOBS=128 REQUEST_TIMEOUT=900 \
bash experiments/tau-2/scripts/chunked_collect.sh --bucket $B --shard $SH --trials 4 --seed 4000 \
  --agent-temperature 1.0 --user-temperature 0.0 --workers 64 --max-steps 40 \
  --agent-model openai/evol-llm-agent --user-model openai/evol-llm-user \
  --agent-max-tokens 1024 --no-agent-thinking --no-user-thinking --out $O/${B}_$n.jsonl > $O/${B}_$n.log 2>&1
echo "R1_EXIT=$?" >> $O/${B}_$n.log
for p in $P $((P+1)); do
  pid=$(ss -ltnpH "sport = :$p" | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2); [ -n "$pid" ] && kill $pid
done
