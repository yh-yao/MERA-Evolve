# MERA-Evolve TAU-2 复现报告（Qwen3.5-4B 学生 / 9B 教师候选）

_2026-11-16 至 2026-11-19。基于 `research_plan.md`（分支 `tau2-4b-rl-handoff`，HEAD `95a0cff`）在
单机 4×H20 95 GiB 上完成。所有产物在 `$MERA_STORE=/data/htlcrosssale-gps-share/powerop_transfer/mera_evolve_repro`。_

---

## 1. 执行摘要

| 计划问题 | 结果 |
|---|---|
| **Q1** 去掉长度截断、加大 batch 后，GRPO 能否显著提升 4B？ | **否。** 200 步、截断 1.33%（<5%），8 个 checkpoint 的 frontier delta 在 −0.6…+3.0 pt，CI 全部跨 0，无一过线 |
| **Q5** 9B 教师相对 4B 学生是否有可测的 headroom？ | **否（同口径）。** 重采同 token 预算的 base 后：28.7% → 31.7%，+3.0 pt，CI [−1.3, +7.3]；优势全部来自 telecom |
| **Q4** 最终 EVAL 上 base 与最佳模型 | base 与 SFT adapter **pass^1 均为 31.7%**，配对差值 +0.0 pt CI[−4.5,+4.3] |
| **Q0** MERA 主循环的 cost–quality 曲线 | **无法测**：M 的前置条件（教师过 Step 0）未满足，按计划决策流停止教师驱动的 M/E6 |
| Q2（G1 vs G2 数据）、Q3（信用分配） | 未跑（E3 暂停；E4 是下一步建议） |

SFT-only（学生自身成功轨迹 rehearsal）是唯一方向为正的训练臂：VAL +3.4 pt（CI[−0.9,+7.6]，不显著），
**但最终 EVAL 上 +0.0 pt**，说明该方向是噪声。工具调用/步数无退化（无 "does less"）。

**一句话**：在本次硬件与预算下，"4B 学生 + 9B 教师"的组合既没有可测的教师 headroom，也没有可测的
学生更新收益；计划的主实验 M 不具备启动条件。与计划的决策流一致，下一步应转向 **E4（信用分配）**
或 **E5（gold-guided self-rollouts）**。

---

## 2. 环境与硬件

| 项 | 值 |
|---|---|
| GPU | **4 × NVIDIA H20 95 GiB**（计划目标为 8 × A100 40 GB，见偏离 #1） |
| 驱动 | 550.90.07 / CUDA 13.0 |
| 训练环境 | `/home/powerop/work/zsc/newproj/.venv`（VERL 0.8.0、vLLM 0.25.1、torch 2.11.0+cu130） |
| serving 环境 | `/home/powerop/.conda/envs/fave-vllm-py311`（vLLM 0.33.0） |
| tau2 环境 | `/tmp/mera_envs/router-skills-evolve/.venv_tau2`（tau2-bench 固定 commit `17e07b1`） |
| 模型 | `models/Qwen3.5-4B`（学生 + 冻结 user simulator）、`models/Qwen3.5-9B`（教师） |
| 关键补丁 | `fla-core 0.6.0`、`causal-conv1d 1.7.0`、`cupy-cuda13x`、`nixl 1.5.2`；rollout 用 Triton 3.6.0、actor 用 3.3.0；`qwen35_lora_patch.py` + sitecustomize 修复 VERL↔vLLM LoRA 命名 |

---

## 3. 各阶段结果

### 3.1 E0 基线（复现性检查）

| 指标 | 实测 | 计划参考 | 状态 |
|---|---|---|---|
| VAL seed300 | 27.6% | 34.0% | **低于参考 ~6.4 pt（超出 ±3 pt 带）** |
| VAL seed1300 | 26.9% | — | — |
| VAL 合并（8 trials） | 27.2% | — | — |
| frontier 58×8 | **61.6%**（464 条，0 错误） | 58.2% | +3.4 pt，略高于参考 |

产物：`results/e0_base_val/`。VAL 与参考值的差距原因未完全定位（已排除 user simulator 与 partition 差异；
候选原因：agent 1024 token 上限、vLLM/tau2 版本漂移）。

### 3.2 E1 R1 采样与选择

| 指标 | 实测 | 计划预期 |
|---|---|---|
| 采样行数 | AREAL 7340 + TRAIN 556，零缺失 | — |
| 0<p<1 任务 | **600 AREAL + 38 TRAIN** | ~546 + ~36（±10% 内）✅ |

产物：`results/e1_r1/`、`results/e2_arm1_lr1e-5/data/grpo_traces.jsonl`（638 prompts）。

### 3.3 E6 Step 0：教师资格判定（三种口径，最终结论：未通过）

| 口径 | base → 9B | delta | 95% CI | 结论 |
|---|---|---|---|---|
| 原判定（分种子 4-trial，CI 判定 + E2 的 +5 pt 门槛） | 27.6/26.9 → 34.5/32.0 | +6.8 / +5.1 | [−0.002,+0.138] / [−0.004,+0.103] | 判"未过" |
| 修正（计划口径：合并 8-trial，报错行记为失败） | 27.2 → 31.7 | **+4.5** | **[+0.2, +8.6]** | 边际通过 |
| **同口径重采 base（`AGENT_MAX_TOKENS=131072`）** | 28.7 → 31.7 | **+3.0** | **[−1.3, +7.3]** | **未通过** |

分域（同口径）：airline −2.1、retail −0.6、**telecom +11.9** —— 教师优势集中在 telecom。

原判定方法上的三个问题：① 用了分种子 4-trial CI（单种子 MDD≈10 pt）而非计划要求的 8-trial 合并；
② 套用了 E2 checkpoint 的 +5 pt 门槛（Step 0 只要求 CI 排除 0）；③ `clean/` 丢弃报错行后 trial 数不等，
`eval_stats` 无法合并，且原"敏感性检查"改写的是 `passed` 而统计读的是 `reward`，实际无效。

基础设施错误：thinking-on 教师产生空 assistant 消息（`AssistantMessage must have either content or
tool_calls`），seed300 3.0%、seed1300 10.4%，温度 0 下确定性复发；`collect_traces --resume` 无法修复，
需在 agent 侧加空消息重试（未做）。

产物：`results/e6_teacher_val/STEP0_VERDICT.txt`、`STEP0_VERDICT_AMENDED.txt`、
`STEP0_VERDICT_AMENDED_matched.txt`、`pooled/`。

### 3.4 Section I 基础设施验收

| 验收项 | 实测 | 结论 |
|---|---|---|
| token 截断率 | 均值 **1.33%**、p95 5.47%、max 7.81% | PASS（<5%） |
| actor 峰值显存 | 26.8 GiB（GRPO），无 OOM | PASS |
| TIS 生效 | `rollout_corr/*` 齐全，`rollout_is_mean` 0.997 | PASS |
| 步时 / 总量 | 566.8 s 均值（中位 554.7，p95 795）；200 步 31.5 h；**126 GPU·h** | 记录 |
| 轮数截断率 | **无法测**（VERL 只记 min/mean/max；74/129 步有 rollout 打到 30 轮上限） | 未闭合 |

产物：`results/e2_arm1_lr1e-5/SECTION_I_PROFILE.md`。

### 3.5 E2 臂 1：GRPO（LR=1e-5，200 步）

| 项 | 值 |
|---|---|
| 数据 | G1：600 valid AReaL + 38 TRAIN（638 prompts），无 VAL/EVAL 泄漏 |
| 设置 | 16 prompts × 8 gens；response 16384 / prompt 8192；30 turns；LoRA r16 α32（attn+MLP）；bf16；micro-batch 1；显式 TIS |
| 训练 | 200/200 步，31.5 h；`critic/score/mean` 分块 0.513→0.500（平坦）；entropy 0.163→0.173（稳定） |

frontier gate（58×8，base 61.6%）：

| ckpt | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 |
|---|---|---|---|---|---|---|---|---|
| delta | −0.6 | +0.2 | +1.5 | +0.4 | **+3.0** | +0.2 | +2.6 | +2.2 |
| CI 跨 0 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**无一过线** → VAL 未触发。由于 frontier 与训练任务重叠（计划已知），"平坦"意味着连训练任务都没有拟合。

**Q1 否定。** 这证伪了计划中"旧的 4B GRPO 平坦是因为截断+小 batch"的假设（截断已从 24–40% 降到 1.33%）。

产物：`results/e2_arm1_lr1e-5/SUMMARY.md`、`eval/step_*/compare.json`。

### 3.6 SFT-only（rehearsal，无 teacher corrections、无 GRPO）

| 项 | 值 |
|---|---|
| 数据 | 学生 E1 成功轨迹 816 行 → 去掉 4 行 >24576 token → **812 行**（仅 TRAIN + valid AReaL，0 泄漏） |
| 训练 | lr 1e-7、1 epoch、max_len 24576、LoRA r16 α32（attn+MLP）、1 卡、406 步、3h03m；loss 0.459→0.345 |
| VAL（8-trial） | 27.2% → 30.6%，**+3.4 pt CI[−0.9,+7.6]**（不显著，MDD 6.1）；per-seed +3.7 / +3.0 |
| pass^4 | 4.5/3.0% → **7.5/7.5%** |
| 行为 | 工具调用/对话 11.05→11.13、步数 39.4→40.1，无 "does less" |
| adapter 生效断言 | `max |lora_B|=4.3e-5`，`max |dlogprob| vs base = 0.11` ✅ |

产物：`results/sft_only/SUMMARY.md`、`eval/sft_vs_base.json`。

### 3.7 最终 EVAL（Q4；536 条/策略，EVAL 未参与任何选择）

| 策略 | pass^1（95% CI） | pass^4（95% CI） |
|---|---|---|
| base | **31.7%** [23.5, 40.3] | **14.8%** [7.5, 23.4] |
| sft | **31.7%** [24.4, 39.4] | **11.3%** [5.0, 18.7] |
| 配对 | **+0.0 pt** CI[−4.5, +4.3]，W/L/T 19/15/33 | — |

分域（pass^1 / pass^4）：airline 70.8/47.1 → 76.4/53.2；retail 60.4/31.2 → 50.0/15.2；
telecom 10.0/0.1 → 13.4/0.1。

**结论：SFT 在 VAL 上的 +3.4 pt 未在 EVAL 复现。** 计划 Q4 的报告项已完成；
headline 的 cost–quality 曲线不可得（M 未启动）。

产物：`results/final_eval/`（`README.md` 含协议与 sha256、`final_eval_stats.json`、
`base_vs_sft_EVAL.json`）。

---

## 4. 计划问题的最终回答

| 问题 | 回答 |
|---|---|
| Q0（主问题）MERA 循环是否移动 cost–quality 曲线？ | **未验证**（前置条件不满足：教师无 headroom） |
| Q1 GRPO（去截断、大 batch）是否显著提升 4B？ | **否**（+3.0 pt 最高，CI 跨 0） |
| Q2 AReaL 任务相对官方 TRAIN 是否有增量？ | 未跑（E3 暂停） |
| Q3 若 Q1 为否，是否是信用分配问题？ | 未验证；**建议 E4 action-recall 优先** |
| Q4 最终 EVAL 的 base 与最佳模型 | 已报告：31.7% vs 31.7%（+0.0 pt） |
| Q5 9B 教师是否可作为教师？ | **否（同口径 +3.0 pt，CI 含 0；优势仅 telecom）** |

---

## 5. 相对 `research_plan.md` 的偏离（完整清单）

1. **硬件**：4×H20 95 GiB（1 serve + 2 actor + 1 rollout），而非 8×A100 40 GB。
2. **Section I profiling**：跳过专门的 10–20 步 profile，用完整 E2 运行的 timing 代替（用户批准）。
3. **新增 LoRA 兼容补丁**（`qwen35_lora_patch.py` + sitecustomize）：VERL 0.8 ↔ vLLM 0.25 命名分叉，不改无法跑。
4. **教师 VAL/frontier 无 agent token 上限；E0 base VAL 为 1024**。后经 `matched_base` 重采（131072）修正教师判定。
5. **教师数据含 3.0%/10.4% 基础设施错误行**（空 assistant 消息）；分析中按失败处理，未修复采集端。
6. **SFT-only 使用学生成功轨迹 rehearsal**，无 teacher corrections、无 GRPO（用户决定）。
7. **臂 2（LR=3e-5）、E3、E4 未跑**（GRPO 暂停）。
8. E2 的 base 基线在 RUN_DIR 内以 pipeline 口径重跑（避免与 08 的 VAL 口径不对称）。
9. 教师 Step 0 改判：使用计划口径（合并 8-trial、报错行记为失败）并新增同口径 base 重采。
10. SFT 输入 816 → **812 行**（丢弃 4 条 >24576 token 的轨迹）。
11. SFT 用 **1 卡**（`nproc=1`）；LoRA target 由仓库默认 attention-only 改为 **attn+MLP**。
12. 最终 EVAL 的"最佳 checkpoint"用 **SFT adapter** 代替（无 E2 checkpoint 过线，且不得用 EVAL 选择）。
13. 计划的 headline **cost–quality 曲线**不可得（M 未启动）。
14. 轮数截断率未测（VERL 日志限制）。
15. frontier 与训练任务重叠（计划已注明，仅作 gate）。
16. E0 VAL 低于计划参考值 ~6.4 pt，原因未定位。

---

## 6. 踩过的技术坑（供后续复现参考）

| 问题 | 现象 | 修复 |
|---|---|---|
| `08` 的 adapter 路径 | evalckpt 189 s 内 8/8 失败，`PASS_COUNT=0` 假结果 | `08` 用 `${TRAIN_PY:-$ROOT/.venv_qwen35/…}`；设置 `TRAIN_PY` 后才能做 Qwen3.5 key 转换 + adapter-active 断言 |
| SFT 自定义 dataset 导入 | `ModuleNotFoundError: tau2_evolve` | SFT 的 `PYTHONPATH` 必须含 `experiments/tau-2`（和 `compat`） |
| 超长 SFT 样本 | `ValueError: sequence_length=33020 > max_length=24576` | 过滤 4 条超长行（0.5%），保留 24k 预算，不截断 |
| SFT LoRA 覆盖不足 | 仓库默认 target 只有 q/k/v/o | 改为 attn+MLP（计划 §8 的坑） |
| 空 assistant 消息 | thinking-on 教师在 tau2 `message.py:288` 抛错 | 需 agent 侧对空 content+无 tool_calls 重试（未做） |
| evalckpt 静默跳 checkpoint | 同一 checkpoint 失败两次后永久跳过 | 已加 GPU 空闲检测，避免把 eval 指到被占用的卡 |
| GPU0 残留 | 训练 stage 的 8890/8891 server 训练结束不退出，占 84 GiB | 串行链在训练后按端口 kill |
| 数据泄漏检查口径 | 同一 `task_id` 在不同 domain 可重复出现 | 必须用 `(domain, task_id)` 对检查；SFT/GRPO 与 VAL/EVAL 交集均为 0 |

---

## 7. 结论与建议

1. **GRPO 线**：在当前数据与预算下（LR 1e-5、200 步、16×8、16k cap、T=1.0），GRPO 对 4B 无可测收益，
   且不是截断问题。**建议下一步做 E4 中成本最低的 action-recall process reward**（仓库已实现，
   group O + `TAU2_ACTION_RECALL_WEIGHT`，约 30 h）。若无收益，再考虑 per-turn advantage / PPO+critic。
2. **教师线**：9B 教师在同口径下无 headroom（唯一亮点 telecom +11.9 pt，n=20）；按计划停止 E6/M。
   若仍要教师，需要显著更强的教师或针对 telecom 的专门教师，并先修空消息采集问题。
3. **SFT 线**：纯 rehearsal 在 VAL/EVAL 均无可测效果；若要继续，应按计划加入 **teacher corrections**
   （需要合格教师）或改用 E5 gold-guided 自采样。
4. **M 主循环**：不启动；前置条件缺失。
5. **工程**：补齐轮数截断日志；把 `TRAIN_PY`/`PYTHONPATH`/超长样本过滤固化进仓库脚本。

---

## 8. 产物地图

| 用途 | 路径 |
|---|---|
| 进度纪要 | `/home/powerop/work/yjh/.claude/PROGRESS_AND_GUIDE.md` |
| E0 | `results/e0_base_val/` |
| E1 | `results/e1_r1/` |
| 教师 Step 0 | `results/e6_teacher_val/`（原/修正/同口径三份判决 + `pooled/`） |
| E2 臂 1 | `results/e2_arm1_lr1e-5/`（`SUMMARY.md`、`eval/step_*/`、`SECTION_I_PROFILE.md`） |
| SFT | `results/sft_only/`（`SUMMARY.md`、`sft_adapter/`、`eval/`） |
| 最终 EVAL | `results/final_eval/`（`README.md`、`final_eval_stats.json`） |
| 复现脚本 | `reproduction_out/scripts/`、`reproduction_out/config/` |
| 分析脚本 | `analyze_teacher_step0_pooled.py`、`analyze_final_eval.py` |

主要复现命令见进度纪要 §9；本文所有数字均可由对应目录的 JSON/日志独立复核。
