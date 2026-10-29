# TAU-2 研究计划：4B 学生的强化学习（交接给大算力团队）

本文档写给接手实验、拥有更强算力的同学。读完应能独立复现已有结论，并跑完下一阶段的关键实验。
仓库的通用说明（环境、启动脚本、不变的约定）见 `CLAUDE.md`；本文只讲 tau2 这条线。

分支：`tau2-4b-rl-handoff`（基于 `tau2-phase0-1-opd-27b`，包含全部代码与实验脚本）。

---

## 0. 一句话现状

在 tau2-bench（airline / retail / telecom 客服任务）上，**学生从 Qwen3.5-2B 换成 Qwen3.5-4B 带来了唯一显著的提升（VAL +10.4 点）**；
之后所有“用老师/数据/RL 再往上推”的尝试都没有显著效果。最新的 4B GRPO 在 32 GB 显卡上做了 250 步，frontier 训练题上没有稳定提升，
**主要嫌疑是显存逼出来的长度上限：约 30% 的训练对话被截断、奖励记 0**。下一步的核心是在大显存上去掉截断、加大 batch，重新检验 GRPO。

---

## 1. 背景与已有结论

### 1.1 评测协议（Phase 0，已固定，请勿改动）

- 划分：`experiments/tau-2/data/partition_v2.json`，TRAIN 139 / VAL 67 / EVAL 67 题。选模型、调参只看 TRAIN 衍生集和 VAL；EVAL 只在最后报告一次。
- 每题 4 次试验（用户模拟器温度 0.7、每次换种子），报告 pass^1 与 pass^4，配对 bootstrap 95% CI 与 McNemar（`tau2_evolve/eval_stats.py`）。
- 单次 VAL（种子 300）的最小可检测差异约 **10 点**；合并种子 300 + 1300（8 次试验）约 **7 点**。
- **frontier 训练题**：58 道题（2B base 在 8 次采样里时对时错的 TRAIN 题），每题 8 次、智能体温度 1.0、用户温度 0、最多 40 步。
  比 VAL 灵敏得多（约 5 点即可见），用于判断“训练到底学进去没有”。这 58 题固定在
  `experiments/tau-2/data/partition_frontier_only.json`（retail 30 / airline 22 / telecom 6，放在 TRAIN 桶里），
  评测时用 `TAU2_PARTITION_PATH` 指向它；它们来自 `tau2_evolve/frontier.py` 对 2B base 8 次采样的分桶结果。
- 训练期与本文所有数字的用户模拟器：本地 **Qwen3.5-4B**。最终 EVAL 计划换成托管强模型做用户模拟器（需要 API key，尚未做）。

### 1.2 参照数字

| 模型 | VAL pass^1 | frontier（58 题 × 8） | TRAIN 贪心（139 题 × 1） |
|---|---|---|---|
| Qwen3.5-2B | 23.5% | 45.0% | 31.2% |
| **Qwen3.5-4B** | **34.0%**（vs 2B +10.4 [+4.1, +16.8]） | **58.2%** | 44.9%（airline 81.8 / retail 66.7 / telecom 1.9） |
| Qwen3.8-27B（全程作答） | 31.7% | 59.3% | — |

要点：4B 已与 27B 老师打平，所以“用 27B 教 4B”没有可测空间；telecom 对所有本地模型都接近 0（2–20%）。

### 1.3 试过且无效的方向（不要重复）

| 方向 | 结果 | 结论 |
|---|---|---|
| 2B：GRPO、全 token 蒸馏（强/弱）、出错点纠错 SFT 等 13 组配置 | 无一显著超过 base；强蒸馏导致重复调用坍塌 | 2B 难以吸收老师能力 |
| 出错点纠错 SFT（教老师在学生第一个错处的动作） | 能改对出错点决策（4/45 → 34/45），但 VAL 不升反降 | 一条失败轨迹往往不止一个错；只纠一处不够 |
| 逐轮 Router（不同时机交给 27B） | 10–30% 老师轮次下 −3 到 +5 点，均不显著；“分歧即用老师”的 oracle 仅 +1.5 | 逐动作替换拿不回老师的优势 |
| 4B 在 AReaL-tau2-data 的 1,983 段 SFT 对话上做 LoRA SFT | VAL 34.0 → 25.4 / 26.5（两档学习率，均显著下降） | 学到了合成数据（Claude 生成、合成数据库）的风格，“做得少了” |
| 4B GRPO（详见 1.4） | frontier 58–62% 来回，无趋势 | 见下 |

### 1.4 最近一次：4B GRPO（32 GB RTX 5090，LoRA）

- 数据：AReaL RL 任务中能通过标准动作回放的 1,844 个（airline 1148 / retail 563 / telecom 133），用 4B base 每题采样 4 次，
  保留 0 < p < 1 的 546 题；另加 TRAIN 中 0 < p < 1 的 36 题。
- 两组：G1 = 546 + 36，G2 = 只用 36 道 TRAIN 题（对照）。奖励只用 tau2 官方判定（组名 `O`）。
- 设置：LoRA rank 16（注意力 + MLP），每步 4 题 × 8 次生成，前 100 步 lr 1e-5，之后从第 100 步 adapter 以 lr 3e-5 再训 150 步；
  prompt 上限 8192，**回复（整段多轮）上限 5632–6144 token**，最多 30 轮。
- frontier 结果（4B base 58.2%）：

| checkpoint | G1 | G2 |
|---|---|---|
| 50（lr 1e-5） | 60.8%（+2.6 [−2.6, +7.8]） | 61.4%（+3.2 [−2.4, +9.1]） |
| 100（lr 1e-5） | 58.0% | — |
| 150（lr 3e-5） | 61.6%（+3.4 [−2.4, +9.3]） | 59.9% |
| 200（lr 3e-5） | 58.2% | — |

- 训练奖励 250 步内一直在 0.2–0.35；PPO 裁剪比例约 0.1–0.2%；行为健康（重复调用 0，工具调用数与 base 相同）。
- **诊断**：`response_length/clip_ratio` 显示 **24–40% 的训练对话被长度上限截断**，记为 0 分。
  这些题在 R1 采样（无长度上限）里 4B 的成功率约为一半，而训练奖励只有 0.25–0.3，差额主要是截断与轮数上限造成的“假失败”。
  同时每步只有 4 题，梯度方差大。两者都是 32 GB 显存逼出来的。

---

## 2. 下一阶段要回答的问题

1. **Q1（主问题）**：去掉截断、加大 batch 后，GRPO 能否让 4B 在 frontier 上显著提升（≥ +5 点且 CI 下界 > 0），并迁移到 VAL？
2. **Q2**：AReaL 合成任务（G1）相对只用官方 TRAIN 题（G2）是否有额外收益？
3. **Q3**：若 Q1 为否，问题是否在信用分配（长对话、只有终局奖励）？用 PPO+critic 或分轮次优势估计检验。
4. **Q4**：最终在 EVAL 上用托管用户模拟器报告 4B base 与最佳模型。

---

## 3. 硬件与环境

- 推荐：单机 8 × 80 GB（A100 / H100 / A800），**NVIDIA 驱动 ≥ 570（CUDA 12.8+）**。驱动低于 535 时，支持 Qwen3.5 的 vLLM 起不来，GRPO 无法运行（SFT 可以）。
- 训练环境（VERL 0.8 + 内置 vLLM 0.18，`.venv_qwen35`）与服务环境（vLLM 0.29，用于评测）按 `CLAUDE.md` 与 `requirements-qwen35-cu129.txt` 搭建；tau2 环境用独立 venv（`TAU2_WORKSPACE`）。
- Qwen3.5 的 GDN 层**必须走快速路径**：需要 `flash-linear-attention` 与 `causal-conv1d`。缺 `causal-conv1d` 时 transformers 会静默退回纯 torch 实现（训练慢 2.5 倍以上），日志里搜 “fast path is not available”。
- 模型：Qwen3.5-4B（学生、用户模拟器），可选 Qwen3.8-27B（老师/上限参照）。

---

## 4. 数据准备（约半天，CPU 为主）

1. tau2-bench 固定在提交 `17e07b1da2bbc0cadfddeea36412686e0604127b`（见 `CLAUDE.md`）。
2. AReaL 数据：`inclusionAI/AReaL-tau2-data`（Apache-2.0，ModelScope / HuggingFace 均有）。设 `MERA_AREAL_DIR` 指向下载目录。
3. 筛掉无法回放的任务（约 45% 的 telecom 任务在我们固定的 tau2 版本下回放不到断言状态）：
   ```bash
   python experiments/tau-2/scripts/check_areal_tasks.py --write-valid $MERA_AREAL_DIR/valid_task_ids.json
   ```
   `tau2_evolve/areal_tasks.py` 会自动读取该文件，并剔除与 VAL/EVAL 故障组合相同的 4 个 telecom 任务。
   预期：airline 1148/1148、retail 563/563、telecom 约 133/267 通过。
4. AReaL 任务按任务加载各自的数据库快照（contextvar 作用域内替换 tau2 的环境构造器，并发安全），运行与评测都生效；
   `collect_traces --bucket AREAL` 即可采样。

---

## 5. 实验（按优先级）

所有训练都用 `experiments/tau-2/09_phase1_opd_ablation.sh`，组 `O`（GRPO，只用官方奖励），`SKIP_EVAL=1`，评测另行进行（第 6 节）。
`experiments/tau-2/scripts/r1_job.sh`、`r3_group.sh`、`r3_watchdog.sh` 是我们在原机器上的包装脚本，**其中的路径是机器相关的**，
可作为参数组合的参考，需按你的目录改写。

### E0 复现基线（约 2 GPU 小时）

- 4B base 的 VAL（种子 300）与 frontier，应得到约 34% 与 58%（±3 点内算一致）。
- 用 `08_phase0_eval_protocol.sh`（VAL）与 `scripts/chunked_collect.sh --bucket TRAIN --trials 8 --seed 2000 --agent-temperature 1.0 --user-temperature 0.0 --max-steps 40`
  加 `TAU2_PARTITION_PATH=experiments/tau-2/data/partition_frontier_only.json`（frontier）。
- 不一致时先查：用户模拟器是否为 Qwen3.5-4B、`enable_thinking` 是否关闭、tau2 提交是否一致。

### E1 R1 筛选（约 10–20 GPU 小时，可并行）

- 4B base 在 AReaL 有效任务（1,844）与 TRAIN（139）上每题采样 4 次（温度 1.0，40 步）：
  `collect_traces --bucket AREAL|TRAIN --trials 4 --seed 4000 --agent-temperature 1.0 --user-temperature 0.0 --max-steps 40`（可用 `--shard i/n` 并行）。
- 用 `scripts/select_frontier.py` 挑出 0 < p < 1 的任务，再用 `tau2_evolve.prepare_grpo_data --max-steps 40 --no-balance-domains` 生成 GRPO parquet。
- 预期：AReaL 约 546 题、TRAIN 约 36 题（我们的结果；种子不同会有 ±10% 浮动）。

### E2 GRPO 无截断（主实验，回答 Q1）

| 项目 | 32 GB 上的旧设置 | 新设置 |
|---|---|---|
| 回复上限（整段多轮） | 5632–6144 | **16384**（目标截断率 < 5%） |
| prompt 上限 | 8192 | 8192 |
| 最多轮数 `MAX_TURNS` | 30 | 30（若截断主要来自轮数，再提到 40） |
| 每步题数 × 生成数 | 4 × 8 | **16 × 8**（或 8 × 16） |
| 学习率 | 1e-5 → 3e-5 | 1e-5 与 3e-5 各一组 |
| LoRA | rank 16，注意力 + MLP | 同左；另加一组 rank 64 或全参数（显存允许时） |
| actor 精度 | `fsdp_config.model_dtype=bf16` | 80 GB 上可回到 fp32 |
| 重要性采样修正 | 未开 | **开**：`algorithm.rollout_correction.rollout_is=token`、`rollout_is_threshold=2.0`、`actor_rollout_ref.rollout.calculate_log_probs=True`（需先确认 VERL 版本支持） |
| 步数 / checkpoint | 250 / 每 50 | 200 / 每 25 |

- GPU 布局（8 卡）：学生评测服务 + 用户模拟器 1 卡，训练 4 卡，rollout 3 卡，例如 `TRAIN_GPU=0,1,2,3 ROLLOUT_GPU=4,5,6`（`03_grpo_only.sh` 已支持逗号分隔多卡，**多卡尚未实测**，先跑 2 步冒烟）。
- 关键环境变量（在 `09` 之外的部分）：`MAX_RESPONSE_LENGTH`、`TRAIN_BATCH_SIZE`、`N_GENERATIONS`、`ACTOR_LR`、`TOTAL_STEPS`、`SAVE_EVERY`、
  `LORA_TARGET_MODULES`、`GRPO_PARQUET`、`EXTRA_TRAIN_ARGS`（追加 VERL 覆盖项）。
- **每步必看的指标**（`group_O/train.log`）：
  - `response_length/clip_ratio`（截断率，必须 < 5%）；
  - `critic/score/mean` 的 25 步滑动平均（应上升）；
  - `actor/pg_clipfrac`、`actor/ppo_kl`（过小说明几乎没更新，过大可能坍塌）；
  - 每条 rollout 的重复工具调用（base 约 0.01；> 0.2 视为坍塌前兆）。
- 判定：
  - **通过**：某 checkpoint 的 frontier 比 4B base 高 ≥ 5 点且 CI 下界 > 0 → 进 VAL（种子 300 + 1300），再进第 7 节最终评测。
  - **截断已 < 5% 但奖励仍平、frontier 不涨** → 转 E4。
  - **出现坍塌**（重复调用上升、工具调用数下降）→ 回退上一个 checkpoint，学习率减半。

### E3 数据对照（回答 Q2，与 E2 同设置）

- G1（AReaL frontier + TRAIN frontier）对 G2（只用 TRAIN frontier）。G2 题量少（约 36 题），需注意过拟合，看 VAL 而非 frontier 下结论。

### E4 信用分配（回答 Q3，仅当 E2 无效时做）

- **分轮次优势估计**：在中间回合分叉重采样，用蒙特卡洛估计每轮价值（可复用 `tau2_evolve/collect_error_points.py` 与 `turn_router.py` 的分叉/逐轮机制）。
- **PPO + critic**：VERL 支持 GAE；critic 与学生同大小（4B），80 GB 上可行。
- **过程奖励**（前期使用，后期退火到 0）：`TAU2_ACTION_RECALL_WEIGHT` 与 `TAU2_ACTION_RECALL_ANNEAL_ROLLOUTS` 已实现（动作召回率）。

---

## 6. 评测流程

- 训练出的 LoRA 若只覆盖注意力 + MLP，可由 vLLM 直接加载：`08_phase0_eval_protocol.sh` 的 `AGENT_ADAPTER` 会自动把 VERL 导出的
  键名转换成 vLLM 需要的 `language_model` 形式，并**断言 adapter 确实生效**（与 base 的 logprob 不同），未生效直接报错。
- 若 LoRA 覆盖了 GDN 投影（`in_proj_*`、`out_proj`），vLLM 的 LoRA 路径不完整，请用 `scripts/merge_lora_qwen35.py` 合并成完整 checkpoint 再评测。
- VERL checkpoint → adapter：`python -m verl_code_rl.extract_sft_lora_adapter --checkpoint-dir <run>/group_O/verl/global_step_N/actor ...`。
- 长时间运行的 Qwen3.5 vLLM 服务会进入 GDN“零吞吐”状态：评测一律走 `scripts/chunked_collect.sh`（每 128 个任务重启服务）。
- 两个 vLLM 服务挤在一张卡上时需要 `VLLM_MAX_NUM_SEQS=128`（否则 Mamba 状态块不够，服务起不来）。

---

## 7. 最终报告（Q4）

- 在 EVAL 上用托管强模型做用户模拟器（`COMMONSTACK_API_KEY` 或其他 OpenAI 兼容端点），报告 4B base 与最佳 checkpoint 的 pass^1、pass^4 与 95% CI，种子 300 + 1300。
- EVAL 在此之前不得用于任何选择。

---

## 8. 踩过的坑（都已修复，换环境时注意）

| 问题 | 现象 | 处理 |
|---|---|---|
| vLLM 静默忽略 VERL 导出的 Qwen3.5 LoRA | 输出与 base 完全相同 | 评测前转换键名，并断言 logprob 有差异 |
| 训练用 JSON 工具格式、评测用 Qwen3.5 原生 XML | 训练学到的用不上 | 训练改为原生协议（`TOOL_FORMAT=qwen3_coder`），`check_native_alignment.py` 校验 |
| LoRA 只挂 q/k/v/o | Qwen3.5 混合结构下只训到 6/24 层 | 默认加上 MLP |
| 注入的老师消息沿用旧时间戳 | tau2 按时间排序后顺序错乱，评测报 “Tool message expected” | 注入时重置时间戳 |
| telecom 用户自己的工具调用混进 agent 历史 | tau2 拒绝该历史；SFT 数据出现空消息 | 按 tau2 的 `is_valid_agent_history_message` 过滤 |
| 缺 `causal-conv1d` | GDN 退回 torch 实现，SFT 慢 2.5 倍 | 从源码编译（CUDA 11.8 / 12.x 均可） |
| actor 默认 fp32 | 4B 在 32 GB 上更新时 OOM | `fsdp_config.model_dtype=bf16` |
| 权重同步缓冲块小于最大张量 | 同步时报错 | `CHECKPOINT_BUCKET_MB` 要大于词表 embedding（4B bf16 约 1.27 GB） |
| 长度上限压太低 | 24–40% 对话被截断、奖励记 0 | 本次交接的核心问题，见 E2 |
| 一步异步 rollout 无重要性采样修正 | rollout 与训练策略的偏差未校正 | E2 中开启 TIS |

---

## 9. 代码地图

| 路径 | 作用 |
|---|---|
| `experiments/tau-2/08_phase0_eval_protocol.sh` | 评测协议（起服务、adapter 断言、分块采集、统计） |
| `experiments/tau-2/09_phase1_opd_ablation.sh` | 训练入口；组 `O` = GRPO 官方奖励；`EXTRA_TRAIN_ARGS`、`SKIP_EVAL` |
| `experiments/tau-2/03_grpo_only.sh`、`scripts/train_verl_grpo.sh` | VERL GRPO 启动（支持多卡列表） |
| `experiments/tau-2/scripts/chunked_collect.sh` | 分块采集（规避 GDN 卡死） |
| `experiments/tau-2/tau2_evolve/collect_traces.py` | 采样；`--bucket AREAL`、`--shard`、`--trials`、`--turn-router` |
| `experiments/tau-2/tau2_evolve/areal_tasks.py` | AReaL 任务加载与按任务数据库 |
| `experiments/tau-2/tau2_evolve/eval_stats.py` | pass^k、配对 bootstrap、McNemar |
| `experiments/tau-2/tau2_evolve/frontier.py`、`scripts/select_frontier.py` | frontier 划分与 GRPO 选题 |
| `experiments/tau-2/tau2_evolve/prepare_grpo_data.py` | GRPO parquet（原生协议） |
| `experiments/tau-2/tau2_evolve/collect_error_points.py`、`turn_router.py` | 出错点、逐轮路由（E4 可复用） |
| `experiments/tau-2/scripts/check_areal_tasks.py` | AReaL 任务回放校验 |
| `experiments/tau-2/scripts/merge_lora_qwen35.py` | 含 GDN 投影的 LoRA 合并 |
| `experiments/tau-2/PLAN_4B_A800.md`、`PLAN_4B_RL.md` | 上一阶段的计划与逐步调整记录 |

---

## 10. 时间与算力估算（8 × 80 GB）

| 步骤 | 估算 |
|---|---|
| 环境 + 数据（E0 前置） | 0.5–1 天 |
| E0 基线复现 | 约 2 GPU 小时 |
| E1 R1 筛选 | 10–20 GPU 小时（并行后数小时） |
| E2 两组 GRPO × 200 步 | 每组约 1 天（4 训练卡 + 3 rollout 卡）；每个 checkpoint 的 frontier 评测约 1–2 GPU 小时 |
| E3 | 与 E2 同时 |
| E4（若需要） | 2–3 天 |
| 最终 VAL / EVAL | 半天（EVAL 需 API 预算） |

## 11. 约定

- `RESULTS_DIR` 一次一配置，不在已有目录里混跑不同设置；续跑在原目录内进行。
- 不在 VAL/EVAL 上训练，不用 EVAL 结果做任何选择。
- 所有结论在重新加载（或合并）adapter 之后再下；评测必须通过 adapter 生效断言。
- `.env`、API key、模型权重、`results/` 不进仓库。
