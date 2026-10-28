# R3 watchdog on jdc_031 (runs until both groups finish). Every 5 min:
#  * a group whose run ended without a final adapter is resumed from its latest checkpoint (<= 3 times);
#  * a group whose train.log has not changed for 45 min while running is reported as HANG;
#  * every new checkpoint (global_step_N) is turned into a LoRA adapter and evaluated on the
#    58 frontier TRAIN tasks (k=8, T=1, 40 steps; same as the 4B base check) on GPU 0, one at a time.
# Status lines go to $W/watchdog.log; R4 results to $W/r4_summary.log.
cd /mnt/local_data/users/luzheng/mera_tau2/MERA-Evolve
W=/mnt/data/users/luzheng/agent_rl/tmp_transfer/r1_4b/watchdog; mkdir -p $W
M=/mnt/data/users/luzheng/agent_rl/models; T=/mnt/data/users/luzheng/agent_rl/tmp_transfer
REF=$T/ref4b; EVAL_PORT=8460
log() { echo "[$(date -u +%FT%TZ)] $*" >> $W/watchdog.log; }
# name dir user_agent_gpu train_gpu rollout_gpu agent_port user_port
RUNS="G1:$(ls -d results/r3_G1_20261028_0801* | tail -1):5:6:7:8280:8281 G2:$(ls -d results/r3_G2_20261028_0802* | tail -1):2:3:4:8282:8283"
alive() { for p in $(pgrep -f "09_phase1_opd_ablation.sh"); do grep -qa "RESULTS_DIR=$PWD/$1" /proc/$p/environ 2>/dev/null && return 0; done; return 1; }
resume() { # dir ug tg rg ap up
  env RESULTS_DIR=$PWD/$1 PHASE1_GROUPS=O ACTOR_LR=1e-5 TOTAL_STEPS=250 TRAIN_BATCH_SIZE=4 N_GENERATIONS=8 SAVE_EVERY=50 \
    CHECKPOINT_BUCKET_MB=1536 MAX_RESPONSE_LENGTH=6144 SKIP_EVAL=1 \
    EXTRA_TRAIN_ARGS="actor_rollout_ref.actor.fsdp_config.model_dtype=bf16" \
    USER_GPU=$2 AGENT_GPU=$2 TRAIN_GPU=$3 ROLLOUT_GPU=$4 AGENT_PORT=$5 USER_PORT=$6 \
    GPU_MEMORY_UTILIZATION=0.42 VLLM_MAX_NUM_SEQS=128 \
    BASE_MODEL=$M/Qwen3.5-4B USER_MODEL_PATH=$M/Qwen3.5-4B TEACHER_MODEL_PATH=$M/Qwen3.8-27B \
    VLLM_BIN=/mnt/data/users/luzheng/agent_rl/.venv_vllm/bin/vllm TAU2_WORKSPACE=$PWD/../router-skills-evolve \
    setsid nohup bash experiments/tau-2/09_phase1_opd_ablation.sh >> $PWD/$1/driver_resume.log 2>&1 < /dev/null &
}
free_gpu() { # a GPU with >= 27 GB free (student + user at 0.42 each); never shares with others' jobs below that
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " '$2 >= 27000 {print $1; exit}'
}
eval_ckpt() { # name dir step gpu (runs in the background eval slot)
  local n=$1 d=$2 s=$3 out=$PWD/$2/r4/step_$3 EVAL_GPU=$4
  mkdir -p $out
  if [ ! -s $out/adapter/adapter_model.safetensors ]; then
    PYTHONPATH=$PWD .venv_qwen35/bin/python -m verl_code_rl.extract_sft_lora_adapter \
      --checkpoint-dir $PWD/$d/group_O/verl/global_step_$s/actor --base-model $M/Qwen3.5-4B --output $out/adapter \
      --lora-rank 16 --lora-alpha 32 --target-modules "[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" > $out/extract.log 2>&1
  fi
  [ -s $out/adapter/adapter_model.safetensors ] || { log "R4 $n step $s: adapter extraction FAILED"; return; }
  TAU2_WORKSPACE=$PWD/../router-skills-evolve TAU2_PARTITION_PATH=$T/frontier_base/split/partition_frontier_only.json \
  SERVERS_DIR=$out BASE_MODEL=$M/Qwen3.5-4B USER_MODEL_PATH=$M/Qwen3.5-4B AGENT_ADAPTER=$out/adapter \
  VLLM_BIN=/mnt/data/users/luzheng/agent_rl/.venv_vllm/bin/vllm AGENT_GPU=$EVAL_GPU USER_GPU=$EVAL_GPU \
  AGENT_PORT=$EVAL_PORT USER_PORT=$((EVAL_PORT+1)) GPU_MEMORY_UTILIZATION=0.42 VLLM_MAX_NUM_SEQS=128 \
  CHUNK_JOBS=128 REQUEST_TIMEOUT=900 \
  bash experiments/tau-2/scripts/chunked_collect.sh --bucket TRAIN --trials 8 --seed 2000 \
    --agent-temperature 1.0 --user-temperature 0.0 --workers 64 --max-steps 40 \
    --agent-model openai/evol-llm-agent --user-model openai/evol-llm-user \
    --agent-max-tokens 1024 --no-agent-thinking --no-user-thinking --out $out/train_k8_frontier.jsonl > $out/run.log 2>&1
  for p in $EVAL_PORT $((EVAL_PORT+1)); do
    pid=$(ss -ltnpH "sport = :$p" | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2); [ -n "$pid" ] && kill $pid
  done
  mkdir -p /tmp/mera_fr_ref && cp -n $REF/base4b_frontier_k8.jsonl /tmp/mera_fr_ref/train_k8_frontier.jsonl
  { echo "== $n step $s ($(date -u +%FT%TZ))"; python3 $REF/mera_diag_analyze.py $T/frontier_base /tmp/mera_fr_ref $out | tail -4;
    python3 $REF/mera_beh_stats.py $out/train_k8_frontier.jsonl; } >> $W/r4_summary.log 2>&1
  rows=$(wc -l < $out/train_k8_frontier.jsonl 2>/dev/null || echo 0)
  if [ "$rows" -ge 400 ]; then touch $out/EVAL_DONE; log "R4 $n step $s evaluated ($rows rows; see r4_summary.log)"
  else log "R4 $n step $s eval incomplete ($rows rows); will retry"; rm -rf $out/served_adapters $out/*.pid; fi
}
log "watchdog start: $RUNS"
while true; do
  all_done=1
  for spec in $RUNS; do
    IFS=: read n d ug tg rg ap up <<< "$spec"
    v=$PWD/$d/group_O/verl; tl=$PWD/$d/group_O/train.log
    if [ -s $v/final_adapter_path.txt ]; then
      [ -f $W/$n.final ] || { touch $W/$n.final; log "$n finished: $(cat $v/final_adapter_path.txt)"; }
    else
      all_done=0
      step=$(grep -oE "training/global_step:[0-9]+" $tl 2>/dev/null | tail -1 | cut -d: -f2)
      if alive $d; then
        age=$(( $(date +%s) - $(stat -c %Y $tl 2>/dev/null || date +%s) ))
        [ $age -gt 2700 ] && log "HANG? $n train.log unchanged for $((age/60)) min at step ${step:-0}"
        echo "$n step=${step:-0} alive" > $W/$n.status
      else
        r=$(cat $W/$n.restarts 2>/dev/null || echo 0)
        if [ $r -lt 3 ]; then
          echo $((r+1)) > $W/$n.restarts; log "$n not running at step ${step:-0} without final adapter: resume #$((r+1))"
          resume $d $ug $tg $rg $ap $up
        else
          echo "$n FAILED (3 resumes used) at step ${step:-0}" > $W/$n.status; log "$n FAILED after 3 resumes"
        fi
      fi
    fi
    # one evaluation at a time, in the background so failure/hang checks keep running
    if ! { [ -f $W/eval.pid ] && kill -0 $(cat $W/eval.pid) 2>/dev/null; }; then
      for ck in $(ls -d $v/global_step_* 2>/dev/null | sort -V); do
        s=${ck##*_}; [ -f $PWD/$d/r4/step_$s/EVAL_DONE ] && continue
        [ -s $ck/actor/model_world_size_1_rank_0.pt ] || continue
        g=$(free_gpu); [ -n "$g" ] || { log "R4 $n step $s: no free GPU yet, waiting"; break; }
        log "R4 $n step $s: evaluating on GPU $g"
        eval_ckpt $n $d $s $g & echo $! > $W/eval.pid
        break
      done
    fi
  done
  if [ $all_done = 1 ] && ! { [ -f $W/eval.pid ] && kill -0 $(cat $W/eval.pid) 2>/dev/null; }; then
    pending=0; for spec in $RUNS; do IFS=: read n d _ <<< "$spec"
      for ck in $(ls -d $PWD/$d/group_O/verl/global_step_* 2>/dev/null); do [ -f $PWD/$d/r4/step_${ck##*_}/EVAL_DONE ] || pending=1; done; done
    [ $pending = 0 ] && { log "both groups finished and all checkpoints evaluated"; break; }
  fi
  sleep 300
done
