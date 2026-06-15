# Online Policy Distillation (OPD) 实验方案

> **历史计划说明（2026-06-09）**：本文档是 OPD 早期实验方案，保留用于回看设计演化；最终结果、正确实现和负结果归因请以 `docs/experiment_report.md` 与 `docs/opd_implementation_analysis.md` 为准。后续已证明：OPD as shaped reward 在本代码任务中会崩溃；正确的 supervised OPD loss 必须包含 live student log-prob；K=1 `token_reward_direct_plus_grpo` 已跑通，K=16 top-K 单卡 OOM。

## 目标

在当前 GRPO 训练基础上引入教师模型（7B）的知识蒸馏信号，解决 1.7B 模型在代码 RL 训练中的**奖励稀疏问题**：当模型答错时 reward=0，梯度为零，无法学习。教师提供 token-level 稠密信号，覆盖 reward=0 的情况。

---

## 背景与动机

| 问题 | 当前情况 | OPD 如何帮助 |
|------|---------|------------|
| 奖励稀疏 | ~65% 样本 reward=0 | 教师 log-probs 提供全部样本的梯度信号 |
| Entropy collapse | step 20-40 坍缩 | 软标签蒸馏有隐式正则化效果 |
| 模型能力上限 | 1.7B 独立探索弱 | 7B 的解法分布作为"知识引导" |

---

## 技术方案

### 损失函数

```
L_total = L_pg (GRPO policy gradient)
        + α * KL(student || ref_initial)     # 现有 KL loss（防走远）
        + β * KL_distill                     # 新增蒸馏 loss（向教师靠近）
```

### KL 蒸馏方向：Reverse KL（推荐）

```
L_distill = KL(student || teacher) 
          = E_student[log(student/teacher)]
          = Σ_t student(t) * [log_student(t) - log_teacher(t)]
```

选择 reverse KL 的原因：
- **Mode-seeking**：学生专注于教师的某个解法模式，而非模糊平均——对代码生成（需要提交确定答案）更合适
- **计算高效**：期望在 student 分布下取，可直接用 rollout 样本计算，无需额外采样
- 与现有 PPO KL penalty 风格一致，便于统一调参

---

## 实现路径

### Phase 1：离线预计算（已完成）

**脚本**：`codellmRL/data/precompute_teacher_logprobs.py`

**功能**：
- 用 `Qwen2.5-Coder-7B-Instruct` 对 MBPP 训练集每个 prompt greedy 生成一条响应
- 计算教师在该响应上每个 token 的 log-prob
- 保存到 `data/mbpp_v2/mbpp_train_with_teacher.parquet`（新增列：`teacher_response_ids`, `teacher_logprobs`, `teacher_pass`）

**启动命令（CPU，GPU 被占用时）**：
```bash
nohup python data/precompute_teacher_logprobs.py \
    --teacher_model models/Qwen2.5-Coder-7B-Instruct \
    --train_file data/mbpp_v2/mbpp_train.parquet \
    --output_file data/mbpp_v2/mbpp_train_with_teacher.parquet \
    --device cpu --dtype float16 --batch_size 1 \
    > <repo>/precompute_teacher.log 2>&1 &
```

**GPU 空闲时（更快）**：
```bash
python data/precompute_teacher_logprobs.py \
    --device cuda --dtype bfloat16 --batch_size 4
```

**预期教师 pass rate**：~60-70%（7B 在 MBPP 上表现远优于 1.7B 的 ~35%）

---

### Phase 2：蒸馏 Loss 集成

**方式 A（推荐，改动小）**：在 reward 函数中叠加蒸馏分数
- 在 `mbpp_reward.py` 的 `compute_score` 中，额外返回 distillation bonus
- 对于 teacher 有正确解的样本，给予基于 token-level KL 相似度的额外分数
- 无需修改 veRL 核心代码

**方式 B（更标准）**：在 actor update loss 中直接加 KL 蒸馏项
- 修改 `verl/workers/fsdp/actor.py` 的 actor update 函数
- 从 batch 中读取 `teacher_logprobs`，计算 KL 蒸馏 loss
- 与 GRPO policy gradient loss 加权求和

---

### Phase 3：训练脚本

**计划脚本**：`codellmRL/scripts/run_mbpp_v2_opd_1_7b.sh`

关键超参：
```bash
DISTILL_COEF=0.1          # 蒸馏 loss 权重（需要 grid search）
KL_LOSS_COEF=0.001        # 保留现有 KL loss
REWARD_MODE=partial       # 保留 partial reward
```

---

## 扩展方向

### Curriculum Learning
利用教师预计算结果对训练题目分层：
- **Easy**：1.7B 已经能解（pass rate > 0）
- **Medium**：7B 能解但 1.7B 不能解（teacher_pass=True, student_pass=False）→ 优先训练
- **Hard**：7B 也解不了（teacher_pass=False）→ 早期跳过，后期加入

### 前向 KL 变体
若 reverse KL 效果有限，可尝试 Forward KL：
```
L_distill = KL(teacher || student) 
```
需要对教师分布采样，计算成本更高，但覆盖性更好（mean-seeking）。

---

## 实验进度

| 步骤 | 状态 | 备注 |
|------|------|------|
| 确定方案（reverse KL + offline teacher）| ✅ 完成 | 2026-06-01 |
| 编写预计算脚本 | ✅ 完成 | `data/precompute_teacher_logprobs.py` |
| CPU 预计算 teacher log-probs | 🔄 进行中 | PID 110718，<repo>/precompute_teacher.log |
| 集成蒸馏 loss 到训练 | ⏳ 待完成 | 等 KL loss 实验结束 |
| 对比实验 | ⏳ 待完成 | baseline: partial_v2_final = 0.3550 |

---

## 预期结果

- **乐观**：pass@1 提升至 0.37-0.39（教师的 60%+ 正确率覆盖了大量 student reward=0 的空白）
- **保守**：提升 1-2%，与现有最优（0.3550）拉开可测量差距
- **风险**：teacher 与 student 能力差距过大时，强制对齐可能反而限制探索；distill_coef 需要调参
