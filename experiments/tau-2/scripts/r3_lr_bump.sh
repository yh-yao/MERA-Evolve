# R3 plan rule "reward flat and the model barely moves -> lr 1e-5 -> 3e-5":
# stop both runs, continue each from its step-100 LoRA with lr 3e-5 for 150 more steps (fresh optimizer).
cd /mnt/local_data/users/luzheng/mera_tau2/MERA-Evolve
M=/mnt/data/users/luzheng/agent_rl/models; L=/mnt/data/users/luzheng/agent_rl/tmp_transfer/r1_4b/watchdog/lr_bump.log
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $L; }
G1=results/r3_G1_20261028_080127; G2=results/r3_G2_20261028_080258
# 1. watchdog (exact argv match) and the two training runs
for p in $(ps -eo pid,args | awk '$2=="bash" && $3=="/tmp/mera_r3_watchdog.sh" {print $1}'); do kill "$p" && log "stopped watchdog $p"; done
desc() { local c; for c in $(pgrep -P $1); do desc $c; echo $c; done; }
for p in $(pgrep -f "09_phase1_opd_ablation.sh"); do
  if grep -qa "RESULTS_DIR=$PWD/results/r3_G" /proc/$p/environ 2>/dev/null; then kill $(desc $p) $p 2>/dev/null; log "stopped run tree $p"; fi
done
sleep 20
for p in $(pgrep -f "ray::|raylet|gcs_server|default_worker|main_ppo"); do grep -qa "RESULTS_DIR=$PWD/results/r3_G" /proc/$p/environ 2>/dev/null && kill $p; done
for p in $(ps -ef | grep "[s]erved-model-name evol-llm" | grep -v VLLM:: | grep -E "port 828[0-3]( |$)" | awk '{print $2}'); do kill $p; done
sleep 20
log "GPUs: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
# 2. step-100 adapters
for d in $G1 $G2; do
  out=$PWD/$d/r4/step_100; mkdir -p $out
  [ -s $out/adapter/adapter_model.safetensors ] || PYTHONPATH=$PWD .venv_qwen35/bin/python -m verl_code_rl.extract_sft_lora_adapter \
    --checkpoint-dir $PWD/$d/group_O/verl/global_step_100/actor --base-model $M/Qwen3.5-4B --output $out/adapter \
    --lora-rank 16 --lora-alpha 32 --target-modules "[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" > $out/extract.log 2>&1
  [ -s $out/adapter/adapter_model.safetensors ] && log "adapter ok: $d step 100" || { log "FATAL adapter extraction failed for $d"; exit 1; }
done
# 3. continue with lr 3e-5
run() { # name src_dir ug tg rg ap up
  R=$PWD/results/r3_${1}_lr3e-5_$(date -u +%Y%m%d_%H%M%S); mkdir -p $R; echo $R >> /mnt/data/users/luzheng/agent_rl/tmp_transfer/r1_4b/R3_DIRS
  env RESULTS_DIR=$R PHASE1_GROUPS=O ACTOR_LR=3e-5 TOTAL_STEPS=150 TRAIN_BATCH_SIZE=4 N_GENERATIONS=8 SAVE_EVERY=50 \
    CHECKPOINT_BUCKET_MB=1536 MAX_RESPONSE_LENGTH=6144 SKIP_EVAL=1 GRPO_PARQUET=$PWD/$2/grpo_train.parquet \
    LORA_ADAPTER_PATH=$PWD/$2/r4/step_100/adapter \
    EXTRA_TRAIN_ARGS="actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" \
    USER_GPU=$3 AGENT_GPU=$3 TRAIN_GPU=$4 ROLLOUT_GPU=$5 AGENT_PORT=$6 USER_PORT=$7 \
    GPU_MEMORY_UTILIZATION=0.42 VLLM_MAX_NUM_SEQS=128 \
    BASE_MODEL=$M/Qwen3.5-4B USER_MODEL_PATH=$M/Qwen3.5-4B TEACHER_MODEL_PATH=$M/Qwen3.8-27B \
    VLLM_BIN=/mnt/data/users/luzheng/agent_rl/.venv_vllm/bin/vllm TAU2_WORKSPACE=$PWD/../router-skills-evolve \
    setsid nohup bash experiments/tau-2/09_phase1_opd_ablation.sh > $R/driver.log 2>&1 < /dev/null &
  log "started $1 lr3e-5 -> $R"
}
run G1 $G1 5 6 7 8280 8281
sleep 90
run G2 $G2 2 3 4 8282 8283
log "lr bump done"
