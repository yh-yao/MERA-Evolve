# usage: r3_group.sh <name> <user_agent_gpu> <train_gpu> <rollout_gpu> <agent_port> <user_port> <traces.jsonl>
# One R3 GRPO run (4B, official reward only, 150 steps x 8 prompts x 8 samples, checkpoint every 25).
N=$1; UG=$2; TG=$3; RG=$4; AP=$5; UP=$6; TR=$7
cd /mnt/local_data/users/luzheng/mera_tau2/MERA-Evolve
M=/mnt/data/users/luzheng/agent_rl/models
export TAU2_WORKSPACE=$PWD/../router-skills-evolve TAU2_PARTITION_PATH=$PWD/experiments/tau-2/data/partition_v2.json
export PYTHONPATH=$PWD/experiments/tau-2:$PWD/experiments/tau-2/compat
export TAU2_SITE_PACKAGES=$(ls -d ../router-skills-evolve/.venv_tau2/lib/python3*/site-packages | head -1)
R=$PWD/results/r3_${N}_$(date -u +%Y%m%d_%H%M%S); mkdir -p $R; echo $R >> /mnt/data/users/luzheng/agent_rl/tmp_transfer/r1_4b/R3_DIRS
.venv_qwen35/bin/python -m tau2_evolve.prepare_grpo_data --traces $TR --output $R/grpo_u$UP.parquet \
  --user-base-url http://127.0.0.1:$UP/v1 --no-user-thinking --max-steps 40 --no-balance-domains > $R/prep.log 2>&1
env RESULTS_DIR=$R PHASE1_GROUPS=O ACTOR_LR=1e-5 TOTAL_STEPS=250 TRAIN_BATCH_SIZE=4 N_GENERATIONS=8 SAVE_EVERY=50 CHECKPOINT_BUCKET_MB=1536 MAX_RESPONSE_LENGTH=6144 SKIP_EVAL=1 \
  GRPO_PARQUET=$R/grpo_u$UP.parquet EXTRA_TRAIN_ARGS="actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" \
  USER_GPU=$UG AGENT_GPU=$UG TRAIN_GPU=$TG ROLLOUT_GPU=$RG AGENT_PORT=$AP USER_PORT=$UP \
  GPU_MEMORY_UTILIZATION=0.42 VLLM_MAX_NUM_SEQS=128 \
  BASE_MODEL=$M/Qwen3.5-4B USER_MODEL_PATH=$M/Qwen3.5-4B TEACHER_MODEL_PATH=$M/Qwen3.8-27B \
  VLLM_BIN=/mnt/data/users/luzheng/agent_rl/.venv_vllm/bin/vllm \
  bash experiments/tau-2/09_phase1_opd_ablation.sh > $R/driver.log 2>&1
echo "R3_EXIT=$?" >> $R/driver.log
