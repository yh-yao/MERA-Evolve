# TAU-2：4B 学生 + AReaL 老师轨迹实验计划（A800）

2026-10-27 起。前一阶段（Phase 0/1，学生 Qwen3.5-2B，老师 Qwen3.8-27B）的结论见改进计划 v3：
13 组把 27B 能力迁移给 2B 的配置都没有显著超过 base；同时测得 4B 本身就比 2B 高约 10 点，
且已和 27B 打平。所以本阶段换成 **4B 学生 + 更强老师的现成轨迹**。

## 固定不变的约定

- 划分：`experiments/tau-2/data/partition_v2.json`（TRAIN 139 / VAL 67 / EVAL 67），不改。
- 只在 TRAIN 及与官方题目不重合的合成数据上训练。AReaL 数据里 telecom 对话带官方 `task_id`，
  凡属 VAL/EVAL 的整段剔除（`scripts/areal_to_tokens.py`，已剔除 9 段）。
- 选模型和调参只看 frontier 训练题和 VAL；EVAL 只在最后报告一次。
- 每个结果目录只对应一种配置；评测前必须确认 adapter 确实生效（与 base 的 logprob 不同）。
- 数据、模型、依赖一律在数据平台用国内源下载（ModelScope / hf-mirror / 阿里云 PyPI /
  PyTorch 官方源），再 rsync 到 A800 共享存储 `/public/home/ai_user_2/mera4b`。

## 参照数字（5090 上已测，本地 4B 用户模拟器，种子 300，每题 4 次）

| 模型 | VAL pass^1 | frontier 训练题（58 题 × 8 次） |
|---|---|---|
| Qwen3.5-2B | 23.5% | 45.0% |
| Qwen3.5-4B | 34.0% | 58.2% |
| Qwen3.8-27B（全程作答） | 31.7% | 59.3% |

单次 VAL 的最小可检测差异约 10 点；合并两个种子（8 次）约 7 点。frontier 训练题更灵敏，
用于判断“训练有没有学进去”，VAL 用于判断泛化。

## 数据

- `inclusionAI/AReaL-tau2-data`（Apache-2.0，ModelScope）：SFT 33,531 条按轮样本，来自 2,499 段对话；
  RL 1,982 个合成任务及 9 个数据库快照。
- 转换后：1,983 段成功对话（retail 992 / airline 824 / telecom 167），约 2,170 万 token；
  system prompt 与工具定义由 tau2 在数据平台重新生成，与评测端逐字节一致（哈希已核对）；
  思考内容丢弃（学生评测不开思考），只训练 assistant 回合，tau2 固定开场白不训练。
- 注意：对话由 Claude 生成（工具调用 id 为 `toolu_bdrk_*`）。研究用途；若要对外发布模型需另行确认条款。

## 步骤

### S0 环境与冒烟测试（A800，约 1 小时）

做什么：用 Python 3.12 + torch 2.7.1+cu118 + transformers 5.10.4 + peft + fla 0.5.2 建训练环境
（10 月在同型号机器上跑通过 Qwen3.8-27B 的 LoRA）；4B LoRA 跑 4 步。

预期：`torch.cuda.is_available()` 为真；fla 的 GDN 内核能编译；4 步 loss 有限且下降。

没达到时：
- CUDA 不可用：驱动 515 只支持到 CUDA 11.7，确认用的是 cu118 轮子并依赖 CUDA 次版本兼容；
  仍不行就换 torch 2.6+cu118。
- fla/Triton 报错：按 10 月的修法设 `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`、
  每个进程独立的 Triton 缓存目录；仍失败就用 transformers 自带的 GDN torch 实现（慢但能跑）。
- 显存或 logits 过大：分块交叉熵（训练脚本已有），减小单条序列上限（现为 32k token）。

### S1 评测通路（与 S2 并行，半天）

问题：评测要用 vLLM 服务 Qwen3.5，而支持 Qwen3.5 的 vLLM 需要 CUDA 12 及以上，A800 驱动是 515。

做法（按优先级）：
1. 在 A800 上试 CUDA 向前兼容（cuda-compat 用户态库 + CUDA 12.8 版 torch/vLLM）。
2. 不行：把训练好的 adapter（几百 MB）传到 5090（jdc），用现成的评测流程评测（需确认允许传出）。
3. 都不行：A800 上用 transformers 起一个简单的 OpenAI 兼容服务评测（慢，只评 frontier 子集）。

预期：至少一条路通，评测结果与 5090 上的 4B base 数字可比（同一用户模拟器、同一种子）。

### S2 4B 在 AReaL 对话上做 LoRA SFT（A800，约半天）

做什么：两组并行，各 4 张卡；LoRA rank 64、alpha 128，覆盖所有线性层；每组 2 个 epoch，
每个 epoch 存一次 adapter。
- A：lr 1e-4
- B：lr 3e-5

评测：每个 adapter 测 frontier 训练题和 VAL（种子 300）；表现最好的再补种子 1300。

预期（通过线）：frontier 训练题比 4B base 高 ≥ 5 点且 95% CI 下界 > 0；VAL 不低于 base，
最好显著更高。

没达到时：
- 训练 loss 不降或 NaN：检查 label 掩码（随机抽样解码训练 token）；lr 降一档。
- frontier 下降（重演 2B 的坍塌）：先看行为统计——重复工具调用、跑满步数的比例、对话长度。
  - 重复调用增多：lr 降到 1e-5，或只训 1 个 epoch。
  - 对话明显变长（学了老师的长篇确认）：评测步数上限与训练数据对齐，或过滤过长对话。
  - 都正常但仍下降：AReaL 合成数据库与官方数据库分布不同，加入学生自己在 TRAIN 上
    成功的轨迹做回放（每题 1 条）。
- frontier 上升但 VAL 不动：先补第二个种子；再按领域拆。若只有 airline/retail 涨，
  telecom 不动，是 telecom 数据太少（167 段），对 telecom 过采样 3 倍再训一组。
- 两组都只有 1–3 点的小提升：数据量是瓶颈，进入 S3 用 RL 任务继续训练。

### S3 GRPO（S2 通过后，1–2 天）

做什么：在 S2 最好的 adapter 上，用 AReaL 的 1,982 个 RL 任务做 GRPO（每个任务加载自己的数据库），
奖励只用 tau2 官方判定。需要先给环境加“按任务加载数据库”（`db_path`）。

前提：S1 第 1 条路通（VERL 需要 vLLM 做 rollout）。否则在 5090 上用 LoRA、16k 上下文、
batch 4 运行（2B 在 batch 8 时 OOM，4B 需要更保守）。

预期：训练奖励上升；frontier 训练题和 VAL 继续高于 S2。

没达到时：
- 奖励不升、模型几乎不动：GRPO 组内方差为 0 的任务太多，先用 S2 模型在 RL 任务上采样 8 次，
  只保留时对时错的任务。
- 奖励上升但评测下降：训练用的合成数据库与官方分布不一致，混入官方 TRAIN 任务（我们的 139 题）。
- OOM：减小 batch、响应长度，开启梯度检查点。

### S4 最终评测

在 EVAL 上用托管模型做用户模拟器（需要 API key），报告 4B base、S2、S3 的 pass^1 / pass^4 和 95% CI。
在拿到 key 之前，所有对比都用本地 4B 用户模拟器，并注明。

## 时间线（预估）

| 时间 | 内容 |
|---|---|
| 10/27 晚 | S0 环境与冒烟；S1 评测通路；S2 开训 |
| 10/28 | S2 评测与调整；决定是否进入 S3 |
| 10/29–30 | S3 GRPO |
| 之后 | S4 最终评测（需 API key） |
