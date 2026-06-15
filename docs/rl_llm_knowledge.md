# RL for LLM 知识体系：原理讲解与实践经验

**项目**：veRL + GRPO + Online OPD，MBPP 代码生成  
**整理时间**：2026-06-03  
**最终更新**：2026-06-09（补充 LCB 结论、OPD 梯度 bug、Arm D token_reward_direct）

本文档从基础概念到框架实践，系统整理了本项目涉及的所有 RL/LLM/veRL 知识点，包括算法原理、工程细节和实践中发现的关键规律。

---

## 最终学习地图

项目结束后，最值得沉淀的知识不是“某个算法名”，而是下面几组因果关系：

| 主题 | 最终理解 |
|---|---|
| SFT vs RL | SFT 学模仿，RL 学结果；SFT 数据质量会决定 RL 起点和探索空间。APPS human SFT 提升 greedy 但压低 pass@5，teacher SFT 才是最终最佳起点。 |
| GRPO | 适合代码生成这种只有最终执行奖励的场景；但同组 reward 全相同会导致 advantage 为 0，因此数据/评测对齐比调参更早优先。 |
| pass@k | pass@1 衡量平均一次能否答对，pass@5 更能反映采样分布是否保留多样性；SFT/OPD 很容易提升 greedy 同时牺牲 pass@5。 |
| OPD 梯度 | teacher log-prob 是目标/权重，真正产生 student 梯度的必须是 live student log-prob。直接 `-mean(teacher_log_probs)` 是 detached 常数，没有梯度。 |
| OPD 注入位置 | OPD as loss 是参数级正则；OPD as reward 会改变 GRPO rollout 选择标准。teacher 偏好长代码时，reward 方式会引发 response length 正反馈。 |
| token_reward_direct | K=1 sampled-token 版本可单卡运行；K=16 top-K 版本理论更低方差，但 PPO 阶段需要对 top-K `log π_new` 反传，单卡 OOM。 |
| 单卡训练 | 显存瓶颈通常来自 FSDP fp32 master weights、vLLM wake/sleep 峰值、全词表 backward，而不是参数量的静态估算。 |

---

## 目录

1. [强化学习基础：RLHF 的本质](#1-强化学习基础rlhf-的本质)
2. [GRPO 算法：免 Critic 的策略优化](#2-grpo-算法免-critic-的策略优化)
3. [KL 散度：前向 vs 反向的深层差异](#3-kl-散度前向-vs-反向的深层差异)
4. [Entropy（策略熵）与 Entropy Collapse](#4-entropy策略熵与-entropy-collapse)
5. [奖励稀疏性与 GRPO 梯度消失](#5-奖励稀疏性与-grpo-梯度消失)
6. [Knowledge Distillation：从离线到在线](#6-knowledge-distillation从离线到在线)
7. [Online Policy Distillation（OPD）原理详解](#7-online-policy-distillationopd原理详解)
8. [浮点格式：float32 / bfloat16 / float16](#8-浮点格式float32--bfloat16--float16)
9. [veRL 框架架构](#9-verl-框架架构)
10. [FSDP 分布式训练原理](#10-fsdp-分布式训练原理)
11. [vLLM 推理引擎原理](#11-vllm-推理引擎原理)
12. [显存管理：单卡极限配置](#12-显存管理单卡极限配置)
13. [pass@k 评估指标详解](#13-passk-评估指标详解)
14. [代码执行沙箱设计](#14-代码执行沙箱设计)
15. [训练指标解读手册](#15-训练指标解读手册)

---

## 1. 强化学习基础：RLHF 的本质

### 1.1 为什么用 RL 训练 LLM

监督微调（SFT）让模型学习"模仿"，RL 让模型学习"结果"。两者的本质区别：

```
SFT：给定正确答案，最大化 P(correct_token | context)
     → 模型学习分布匹配，但不知道"为什么"这个答案是好的

RL：让模型自己生成，用执行结果打分，优化获得高分的行为
   → 模型通过试错发现"什么样的输出能通过测试"
```

对于代码生成，SFT 需要大量人工标注的"好代码"，而 RL 只需要一个代码执行器（判断代码是否通过测试用例），可以自动生成无限量的训练信号。

### 1.2 LLM 的 RL 问题形式化

将文本生成建模为序列决策问题：

```
状态 (State)：当前已生成的 token 序列（prompt + 已生成部分）
动作 (Action)：选择下一个 token（词表中的一个词）
策略 (Policy)：π_θ(token | context)，即模型的 softmax 输出
奖励 (Reward)：只在完整序列生成结束后给出（执行代码，计算通过率）
```

这是一个**稀疏奖励的序列决策问题**：中间过程无奖励，只有最终结果有奖励。

### 1.3 两大主流方法：PPO vs GRPO

**PPO（Proximal Policy Optimization）**：
- 需要 Critic 网络估计每个状态的价值（V function）
- Advantage = reward - V(state)（实际收益 - 预期收益）
- 适合有中间奖励的场景，但 Critic 需要额外的显存和计算

**GRPO（Group Relative Policy Optimization）**：
- 不需要 Critic
- 对同一个 prompt 生成多条回答，用组内相对排名作为 advantage
- 适合只有最终奖励的场景（如代码生成、数学推理）

---

## 2. GRPO 算法：免 Critic 的策略优化

### 2.1 核心思想

对同一个 prompt，生成 N 条（本项目 N=5 或 8）独立回答，计算各自 reward。用组内 reward 的相对高低来判断哪条回答"更好"：

```python
rewards = [r_1, r_2, ..., r_N]  # 例：[1, 0, 1, 0, 1]
baseline = mean(rewards)         # 例：0.6
std = std(rewards)               # 用于归一化

# advantage：高于均值为正（这条回答比同组平均更好），低于均值为负
advantages = (rewards - baseline) / (std + ε)  # 例：[0.8, -1.2, 0.8, -1.2, 0.8]

# 如果 std = 0（全组 reward 相同），则 advantages = 0，此步骤无梯度
```

**直觉**：不需要知道"好代码是什么样的"的绝对标准，只需要比较同一道题的不同解法谁更好。

### 2.2 完整损失函数

**Step 1: 重要性采样比（Importance Sampling Ratio）**

```
ρ_{i,t} = π_θ(token_{i,t} | context) / π_old(token_{i,t} | context)
```

训练时策略 π_θ 已经被更新过，但数据是用旧策略 π_old 采样的。需要用 ρ 纠正分布偏差（将"旧策略采样的数据"重新加权，变成"当前策略下的期望"）。

**Step 2: PPO-clip 目标**

```
J_clip = E[ min(ρ · A,  clip(ρ, 1-ε, 1+ε) · A) ]
```

**clip 的直觉**：
- A > 0（好的动作）且 ρ > 1+ε（概率已经大幅提升）→ 停止继续强化，防止单次更新过头
- A < 0（坏的动作）且 ρ < 1-ε（概率已经大幅降低）→ 停止继续惩罚

**完整 GRPO loss**：

```
L(θ) = -(1/N) Σ_i (1/|o_i|) Σ_t min(ρ_{i,t}·A_{i,t}, clip(ρ_{i,t}, 1-ε, 1+ε)·A_{i,t})
```

加负号：梯度下降等价于最大化 reward。

### 2.3 为什么需要 clip

如果不 clip，直接用 ρ × A 作为梯度：
- 好的样本（A > 0）的 ρ 越大，梯度越大，策略更新越激进
- 一批数据对策略的影响可能过大，导致"策略飞出"（kurtosis 很大的更新）

clip 确保单次更新时策略变化不超过 ε 范围，**多次复用同一批数据时不会发散**。

### 2.4 Off-policy 与数据复用

veRL 的训练循环：
```
1. π_old 采样一批数据（rollout，64 条 × N=5 = 320 条）
2. 切成 mini-batch，更新多次（每次 32 条，更新 2 次）
3. 更新后 π_θ 变成新的 π_old，重新采样
```

第 2 步的多次更新就是 off-policy 部分：数据是旧策略采的，但策略已经在更新。clip 保证在 ε 范围内的 off-policy 近似是安全的。

**N=5 的重要性**：
- N=1 时：advantage = reward - 0（无基线），等价于 REINFORCE，方差极大
- N=5 时：组内基线 = 0.6（5 条中 3 条正确），能区分好坏回答
- 如果全组 reward 相同（std=0），advantage=0，整批数据梯度为零（无效步）

---

## 3. KL 散度：前向 vs 反向的深层差异

### 3.1 基本定义

KL 散度衡量分布差异，**不对称**：

```
KL(P ‖ Q) = E_P[log(P/Q)] = Σ P(x) · log(P(x)/Q(x))
```

注意：KL(P‖Q) ≠ KL(Q‖P)

### 3.2 前向 KL：Mean-Seeking（覆盖所有模式）

```
KL(teacher ‖ student) = E_teacher[log(teacher/student)]
```

**惩罚机制**：若 teacher 对某处有概率（P(x) > 0）但 student 没有（Q(x) = 0）→ log(P/0) = +∞，无穷大惩罚。

**结果**：student **不敢遗漏** teacher 的任何 mode，被迫"撒网覆盖"，分布更宽泛。

**计算方式**：需要在 teacher 分布下采样，或遍历 teacher 的支撑集。

**适用场景**：希望保留多样性，不遗漏教师的任何能力。

### 3.3 反向 KL：Mode-Seeking（聚焦于最可能的模式）

```
KL(student ‖ teacher) = E_student[log(student/teacher)]
```

**惩罚机制**：若 student 对某处有概率（Q(x) > 0）但 teacher 没有（P(x) = 0）→ log(Q/0) = +∞，无穷大惩罚。

**结果**：student **不敢"越界"**，只在 teacher 有概率的地方放概率，分布更尖锐、聚焦。

**计算优势**：期望在 student 分布下取，可直接用已有的 rollout 样本（student 自己生成的序列）计算，**无需额外采样**。

**适用场景**：代码生成（需要提交确定的、正确的解法）、PPO KL penalty（防止策略走远）。

### 3.4 在本项目的使用

| 用途 | KL 方向 | 具体形式 |
|------|---------|---------|
| PPO KL penalty | 反向 | KL(π_actor ‖ π_ref)：防止策略偏离初始模型 |
| OPD supervised loss | 反向 sampled-token KL | `mean(log π_student_live(ŷ_t) - log π_teacher(ŷ_t).detach())`：teacher 分数是目标，student live log-prob 提供梯度 |
| KL 的两种实现 | 见 3.5 | KL loss vs KL in reward |

### 3.5 KL 的两种工程实现

**方式 1：KL loss（`use_kl_loss=True`）**

```
total_loss = L_GRPO + kl_loss_coef × KL(π_actor ‖ π_ref)
```

- KL 直接进 loss，通过 backward 作用于梯度
- 需要 ref model 前向（计算 ref_log_probs），ref_log_probs 参与 backward
- 显存：额外保留 ref model forward 的 computational graph

**方式 2：KL in reward（`use_kl_in_reward=True`）**

```
adjusted_reward = reward - kl_coef × KL(π_actor ‖ π_ref)
advantages = compute_advantage(adjusted_rewards)  # stop gradient
```

- KL 折算成负的 reward，在 advantage 计算前减掉
- advantage 是 stop gradient 的常数，KL 只间接影响梯度方向
- 显存：ref model 只需前向，不需要保留 computational graph，更省显存

**重要区别**：两者都需要 ref model 前向，区别在于 ref_log_probs 是否参与反向传播。

---

## 4. Entropy（策略熵）与 Entropy Collapse

### 4.1 策略熵的定义

```
H(π) = -E_π[log π(a|s)] = -Σ π(a) · log π(a)
```

**含义**：衡量策略的**不确定性/多样性**。
- 熵高 → 策略接近均匀分布，每个 token 的概率相近，探索多样
- 熵低 → 策略高度集中于某些 token，几乎只生成固定输出

### 4.2 Entropy Collapse 的机制

GRPO 通过 advantage 加权不断强化 reward 高的 token 序列：
```
1. 某些 token 序列（通过了测试）→ advantage 为正 → 概率提升
2. 其他 token 序列 → advantage 为负或零 → 概率降低
3. 随着训练，好的序列概率越来越高，策略越来越"确定"
4. 熵单调下降，最终接近 0
```

**后果**：一旦 entropy 趋近于 0，策略对所有输入几乎输出相同序列，新的 rollout 不再有多样性，整组 N 条回答的 advantage 全为 0，梯度信号消失。

**本项目观察**：
- Binary reward：step 20 左右 collapse
- Partial reward（无 KL）：step 40 左右 collapse
- Partial + KL loss：推迟到 step 40+ 才出现振荡，但仍最终 collapse
- SFT warmup + GRPO：step 27→0.163，step 80→0.055，**持续下降但无 collapse**
- Online OPD：step 0→0.275，step 80→0.109，**缓慢下降但无 collapse**

### 4.3 应对方案及可行性分析

**方案 1：entropy_coeff（直接最大化熵）**

```
total_loss = L_GRPO - entropy_coeff × H(π)
```

- 原理：直接在 loss 中加熵正则，阻止熵下降
- **问题**：entropy 计算需要全词表（151936 维）softmax + backward，额外约 5GB 显存峰值
- **结论**：在 1.7B 单卡极限配置下 OOM，不可用

**方案 2：KL loss（间接维持熵）**

```
total_loss = L_GRPO + kl_loss_coef × KL(π_actor ‖ π_ref)
```

- 原理：惩罚偏离 ref model，间接约束 entropy 不能偏离 ref 太远
- **优点**：只需要已选 token 的 log_prob，不需要全词表计算，显存开销极低
- **结论**：推迟了 collapse，但不能根本解决（最终还是 collapse，只是更慢）

**方案 3：SFT Warmup（改变初始分布）**

- 原理：用高质量样本预先微调，给 GRPO 一个更好的起点，减少 GRPO 需要"探索"的空间
- **效果**：全程（80步）无 collapse，梯度信号持续有效
- **结论**：最有效的解法（本项目 Exp 8 最优）

**方案 4：Online Policy Distillation（持续注入信号）**

- 原理：teacher 每步提供行为克隆信号，避免策略过于激进地压低 non-winning tokens
- **效果**：entropy 缓慢下降但无 collapse（0.275→0.109，比 SFT+GRPO 的 0.163→0.055 更缓慢）
- **结论**：对小模型（0.5B）有效，每步额外开销约 10 秒（teacher 的两次 PCIe 传输）

---

## 5. 奖励稀疏性与 GRPO 梯度消失

### 5.1 奖励稀疏性的本质

在代码任务早期，模型的大多数生成都是错误的：
- 函数名不对（本项目 v1 数据集问题）
- 语法错误
- 逻辑错误（算法思路错误）

当一个 prompt 对应的 N=5 条 rollout 全部 reward=0 时：

```python
rewards = [0, 0, 0, 0, 0]
advantages = (rewards - mean(rewards)) / std(rewards)
           = (0 - 0) / 0  → NaN 或直接置为 0（veRL 的处理方式）
```

**结果**：这批数据对梯度贡献为零，就像"白跑"了 N=5 次采样。

### 5.2 Batch Size 对信号质量的影响

| batch_size | 全零批次概率（估计）| 有效梯度步比例 |
|-----------|----------------|------------|
| 16 | ~22% | ~78% |
| 32 | ~8% | ~92% |
| 64 | ~2% | ~98% |

对于 MBPP 这样的小数据集（374 条），大 batch_size 使每步采样覆盖更多不同的 prompt，全零批次出现概率大幅降低。

**反直觉结论**：想增加训练步数，不应该减小 batch_size（全零批次增多），而应该增加 epochs（重复遍历数据集）。

### 5.3 GRPO 的"顿悟"现象

在稀疏奖励场景下，GRPO 常出现非线性的性能跃升：

```
早期（reward ≈ 0）：
  → 所有 group 的 advantage ≈ 0
  → 梯度几乎为零，策略几乎不更新
  → 表现上像是"什么都没学"

顿悟时刻（某个 prompt 偶然有 1 条回答答对）：
  → 这组的 advantage = [1 - 0.2, -0.2, ...] = [0.8, ...]（非零！）
  → 这条"幸运答案"的 token 概率大幅提升
  → 更多相似问题开始答对，正向飞轮启动
```

**为什么 n=5 比 n=1 更容易触发顿悟**：n=5 时，5 条回答中偶然有 1 条答对，就产生非零 advantage。n=1 时，只有答对才有梯度（reward - 0 = 1），答错完全无梯度，统计上需要更多的"幸运"样本。

---

## 6. Knowledge Distillation：从离线到在线

### 6.1 知识蒸馏的基本思路

大模型（teacher）的知识通过两种途径传递给小模型（student）：

**输出蒸馏（Logit Distillation）**：
```
L_distill = KL(teacher_probs ‖ student_probs)
```
用 teacher 的 softmax 分布作为"软目标"，比 one-hot 标签提供更丰富的梯度信号。

**行为克隆（Behavior Cloning，BC）/ sampled-token KL**：
```
L_BC = mean(log π_student_live(a_t|s_t) - log π_teacher(a_t|s_t).detach())
```
它等价于在 student rollout token 上做反向 KL 的单点估计。注意：`teacher_log_probs` 本身是 detached target，**不能单独作为 loss**；真正产生 student 梯度的是 `log π_student_live`。

本项目使用 BC loss，因为：
1. 只需要 sampled token 的 student/teacher log_prob，避免 full-distribution KL 的显存开销
2. teacher 只做 forward，不参与 backward
3. 可与 GRPO loss 在 actor update 内相加，作为参数级正则

### 6.2 离线蒸馏 vs 在线蒸馏

**离线蒸馏（Exp 8 的做法）**：
```
Phase 1（预处理）:
  teacher → 生成高质量代码样本 → 筛选 teacher_pass=True 的样本（~192条）

Phase 2（SFT warmup）:
  student 在 teacher 生成的好样本上做 SFT
  目标：提升 student 的初始代码能力

Phase 3（GRPO）:
  从 SFT 后的 student 起训，做普通 GRPO（无 teacher 参与）
```

**优点**：Phase 3 没有额外开销（teacher 已离线）  
**缺点**：静态标签，student 进步后标签不再引导更高水平；teacher 生成的样本可能有风格偏差

**在线蒸馏（Exp 9 的做法，Online Policy Distillation）**：
```
每步训练都有：
  student rollout → teacher 实时评分 → BC loss + GRPO 联合优化
```

**优点**：on-policy，teacher 始终评估当前 student 的生成  
**缺点**：每步需要 teacher 两次 PCIe 传输（~10s/步额外开销）

---

## 7. Online Policy Distillation（OPD）原理详解

### 7.1 核心设计

OPD 的核心创新：teacher 不是评估"好代码"的绝对标准，而是评估 **student 自己生成的代码有多合理**。

```python
# student rollout（vLLM 采样）
student_output: prompt + "def add(a, b):\n    return a + b\n"

# teacher forward（在 student 的输出上做前向传播）
input = tokenize(prompt + student_output)
teacher_logits = teacher_model(input)  # (B, seq_len, vocab_size)
teacher_log_probs = log_softmax(teacher_logits)

# 取 response 部分、已出现 token 的 log prob
# teacher_log_probs[t] = teacher 认为"在此位置出现 token_t 有多合理"
teacher_response_log_probs = gather(teacher_log_probs, response_token_ids)
# shape: (B, resp_len)
```

**teacher_log_probs 的直觉含义**：teacher 给 student 每个生成 token 的"认可分"。值越接近 0（即 log prob 越大），teacher 越认为"这个 token 在这里出现是合理的"。

### 7.2 损失函数分析

```python
total_loss = L_GRPO + alpha × L_BC

L_GRPO = -mean(clip(ρ, 1-ε, 1+ε) × advantage)  # 强化通过测试的写法
L_BC = mean(log π_student_live(ŷ_t) - log π_teacher(ŷ_t).detach())
# 引导 student 在自己采样到的 token 上向 teacher 认可的分布靠拢
```

**两个 loss 的互补关系**：
- `L_GRPO`：结果导向（代码能跑过测试用例吗？）→ 确保最终正确性
- `L_BC`：过程导向（每个 token 的选择 teacher 认可吗？）→ 引导"写出好代码的思路"

**曾经踩过的关键 bug**：如果写成 `L_BC = -mean(teacher_log_probs)`，由于 `teacher_log_probs` 与 student 参数无关且通常已经 detached，`∂L/∂θ = 0`，不会训练 student。正确实现必须让 live student log-prob 出现在 loss 里。

**α=0.1 的选择**：
- α 太大：L_BC 主导，student 变成 teacher 的"复印机"，失去 RL 的探索能力
- α 太小：L_BC 信号太弱，不足以对抗 GRPO 的 entropy collapse
- 0.1 是经验值，在本项目中表现良好（entropy 从 0.275→0.109，未 collapse）

### 7.3 "在线"的含义

"在线"（On-policy）在 OPD 中的含义：teacher 每步评分的是**当前 student 版本**生成的内容，而非 teacher 自己"想写"的代码。

这一点很关键：
- 如果 teacher 评分的是 teacher 自己生成的代码（离线），student 只是在"模仿"静态样本
- 如果 teacher 评分的是 student 生成的代码（在线），teacher 的评分信号随 student 能力提升而"升档"，始终引导 student 向更高水平写法靠拢

类比：在线 OPD 像是一个导师每次修改学生的作业，指出"这里写得不够好"；离线蒸馏像是导师先写一份范文，学生背诵范文。

### 7.4 Scheme B：内存高效的 teacher offload 方案

**问题**：teacher（7B，bfloat16，~14GB）和 student（0.5B，~2GB + 梯度 ~8GB）无法同时在 GPU 上做 backward。

**Scheme B 方案**（每步完整流程）：

```
① Rollout（~30s）
   student 用 vLLM 生成代码
   teacher 在 CPU 等待（idle）

② Log prob（~3s）
   student 计算 old_log_probs
   teacher 仍在 CPU

③ Reward（~5s）
   执行代码，计算 pass/fail

④ Advantage 归一化（~1s）
   GRPO 组内 advantage 计算

⑤ Update Actor（~20s）
   a. teacher.to("cuda")    ← PCIe CPU→GPU（~14GB，~5s）
   b. teacher forward       ← 计算 teacher_log_probs（~3s）
   c. teacher.cpu() +       ← PCIe GPU→CPU（~5s）【必须在 backward 之前！】
      empty_cache()
   d. student forward +     ← loss = L_GRPO + 0.1×L_BC
      backward + step       ← 此时 GPU 上只有 student

总计：每步约 60s，其中 OPD 额外开销约 10s（PCIe 传输）
```

**为什么 teacher offload 必须在 backward 之前**（Bug 8 的根本原因）：
- backward 时需要保留 forward 阶段的 activation（gradient checkpointing 会重计算，但仍需一定显存）
- 如果 teacher（14GB）在 GPU 上时做 student backward，student 的 activation 峰值（~9GB）叠加超出 GPU 余量
- **解决方法**：teacher forward 完立即 offload，再做 student backward，两者不同时在 GPU

---

## 8. 浮点格式：float32 / bfloat16 / float16

### 8.1 存储结构

所有 IEEE 754 浮点数由三部分组成：

```
value = (-1)^符号 × 2^(指数 - 偏置) × (1 + 尾数)
```

三种格式的位布局：

```
float32  (32位):  [1位符号][8位指数][23位尾数]
bfloat16 (16位):  [1位符号][8位指数][ 7位尾数]   ← float32 截掉后16位尾数
float16  (16位):  [1位符号][5位指数][10位尾数]
```

**关键规律**：
- 最大值由**指数位**决定：2^(2^指数位 - 1) ≈ 2^(128) ≈ 3.4×10³⁸（fp32/bf16），2^(32) ≈ 65504（fp16）
- 精度由**尾数位**决定：23位尾数 ≈ 7位十进制精度，7位 ≈ 2-3位，10位 ≈ 3-4位

### 8.2 关键特性对比

| 特性 | float32 | bfloat16 | float16 |
|------|---------|----------|---------|
| 总位数 | 32 | 16 | 16 |
| 指数位 | 8位 | **8位（同 fp32）** | 5位 |
| 尾数位 | 23位 | 7位 | 10位 |
| 最大值 | ~3.4×10³⁸ | **~3.4×10³⁸** | ~65504 |
| 精度（十进制）| ~7位 | ~2-3位 | ~3-4位 |
| 显存/参数 | 4 bytes | 2 bytes | 2 bytes |
| 溢出风险 | 无 | **无** | **有**（大模型易触发）|
| 转换成本 | — | 直接截断 fp32 后16位 | 需要指数重映射 |

**bfloat16 的设计哲学**：牺牲精度（7位 vs 23位），换取与 fp32 相同的数值范围（8位指数）。这在深度学习中是完美权衡：
- **精度损失可接受**：SGD 本身就是随机的，梯度噪声远大于精度误差
- **数值范围关键**：大模型 logits 量级大，fp16 的 65504 上限不够用

### 8.3 float16 溢出问题（本项目实际踩坑）

```python
# 7B 模型以 float16 加载，GPU 推理时 logits 可能超出 65504
logits = teacher_model(input_ids)
# → 某些位置的 logit 变成 inf（overflow）

# 对 [inf, inf, ..., inf] 做 log_softmax：
# softmax([inf,...]) = exp(inf) / sum(exp(inf)) = inf/inf = NaN
# log(NaN) = NaN
log_softmax([inf, ...]) = NaN  ← 崩溃根源

# NaN 传播链：
# teacher_log_probs = NaN
# bc_loss = -mean(NaN) = NaN
# total_loss = L_GRPO + 0.1 × NaN = NaN
# loss.backward() 在 NaN gradient 上 → OOM crash
```

**关键教训**：推理大模型时，**永远用 bfloat16，不要用 float16**。两者显存开销相同（都是 2 bytes/参数），但 bfloat16 完全消除了 overflow 风险。

### 8.4 在本项目的使用策略

| 场景 | 格式 | 原因 |
|------|------|------|
| 1.7B/0.5B student 训练（FSDP）| bfloat16 | veRL 默认，节省显存，训练稳定 |
| 7B teacher 推理（OPD）| **bfloat16** | 防 logits overflow，显存与 fp16 相同 |
| 4D attention mask | float16 | 仅做 masked_fill，精度要求低 |
| bc_loss 中 log_softmax | float32（cast）| 中间精度防止数值不稳定 |
| Adam 优化器状态 | float32 | 默认，optimizer_offload 时 CPU 上保持 fp32 |

---

## 9. veRL 框架架构

### 9.1 整体设计

veRL（Volcano Engine RL）是字节跳动开源的 LLM 强化学习框架，主要特点：
- 支持 GRPO/PPO 等主流 RL 算法
- 使用 Ray 分布式计算，支持多机多卡
- vLLM 负责高效 rollout 采样
- FSDP 负责 actor 训练（全参数微调）

**单卡 colocate 模式**（本项目的架构）：
```
┌──────────────────────── 单张 L20 48GB ─────────────────────────┐
│  actor FSDP（训练时）              ┐                             │
│  ref FSDP（log_prob 时，否则 CPU）  ├── 共享单卡，轮流使用         │
│  vLLM（rollout 时）               ┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 训练循环的三个阶段

**阶段 1：Rollout**
- vLLM 加载 actor 的最新权重（通过 `update_weights`）
- 对所有 prompt 采样 N 条回答（vLLM 高效并行推理）
- 结束后 vLLM 进入 sleep 模式（或销毁，取决于 `free_cache_engine` 设置）

**阶段 2：Computing（log_prob + reward）**
- FSDP 重新获得 GPU 控制权
- 计算 actor 的 old_log_probs 和 ref 的 log_probs（用于 KL 计算）
- 用户定义的 reward function 计算每条回答的 reward

**阶段 3：Training（update_actor）**
- GRPO advantage 计算
- Actor FSDP forward + backward + optimizer.step()
- OPD 模式下：teacher 在此阶段上 GPU → 前向 → 下 GPU → student backward

### 9.3 数据流格式

veRL 使用 `DataProto` 和 `TensorDict` 在组件间传递数据：

```python
# DataProto 结构
data.batch["input_ids"]          # shape: (B, seq_len)，tensor
data.batch["attention_mask"]     # shape: (B, seq_len)，tensor
data.batch["advantages"]         # shape: (B, resp_len)，tensor
data.non_tensor_batch["indices"] # numpy array，packed format 的索引
data.non_tensor_batch["max_seq_len"]  # int，packed 后的最长序列长度
```

**packed format（`left_right_2_no_padding`）**：
为了避免 padding 浪费，veRL 将多条序列打包成 nested tensor（jagged layout）。`indices` 记录每个有效 token 在 flat 序列中的位置，用于后续的 `pad_input` 还原。

### 9.4 自定义奖励函数接口

```python
# 奖励函数必须满足的签名
def compute_score(data_source: str, 
                  solution_str: str,
                  ground_truth: any,
                  extra_info: dict) -> float:
    """
    data_source: 数据集名称（如 "mbpp"）
    solution_str: 模型生成的完整字符串（含 <think> 标签等）
    ground_truth: 数据集中的正确答案（如 assert 语句列表）
    extra_info: 额外信息（如题目类型）
    return: 0.0 ~ 1.0 的 reward 值
    """
    pass

# 在训练脚本中指定
reward.custom_reward_function.path=<repo>/rewards/mbpp_reward.py
reward.custom_reward_function.name=compute_score
```

**重要限制**：奖励函数在 Ray worker 启动时一次性导入（通过 `importlib` 缓存），运行中修改文件**不会**热生效，必须重启训练进程。

---

## 10. FSDP 分布式训练原理

### 10.1 FSDP 的核心思想

FSDP（Fully Sharded Data Parallel）是 PyTorch 的分布式训练策略，将模型参数按照 rank 分片存储：

```
单机（N=4 GPU）的 FSDP：
  GPU 0: 保存 1/4 的模型参数
  GPU 1: 保存 1/4 的模型参数
  GPU 2: 保存 1/4 的模型参数
  GPU 3: 保存 1/4 的模型参数

前向传播时：
  All-Gather → 重建完整模型 → 计算 → 释放其他 GPU 的参数

反向传播时：
  All-Reduce → 梯度聚合 → 各 GPU 更新自己的参数分片
```

**单卡场景**（本项目）：FSDP 在单卡上相当于全参数持有，但 offload 机制仍然可用（将部分张量 offload 到 CPU）。

### 10.2 FSDP checkpoint 格式

veRL 保存的 checkpoint：
```
global_step_70/
  actor/
    model_world_size_1_rank_0.pt   ← FSDP shard 格式，不是 HuggingFace 格式！
    huggingface/                   ← 已合并的 HuggingFace 格式（如果 save_hf=True）
      config.json
      model.safetensors
      tokenizer.json
      ...
```

**合并命令**（当没有 `save_hf` 时手动合并）：
```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $VERL_DIR/checkpoints/.../global_step_70/actor \
    --target_dir models/eval_merged/model_name
```

### 10.3 optimizer_offload 和 param_offload

**optimizer_offload**（`fsdp_config.optimizer_offload=True`）：
- 将 Adam 优化器的 momentum 和 variance（各一份 fp32 权重大小）offload 到 CPU
- 显存节省：model_params × 2 × 4 bytes（fp32）
- 代价：每步 CPU↔GPU 传输 Adam 状态，每步约慢 10-20%
- 对于 1.7B：节省约 3.4 × 2 = 6.8 GB（fp32 Adam state）

**param_offload**（`fsdp_config.param_offload=True`，对 ref model）：
- 将 ref model 的权重 offload 到 CPU，只在计算 log_prob 时临时加载
- ref model 不需要梯度，不需要常驻 GPU
- 显存节省：ref model 全部权重（1.7B 约 3.4GB bf16，7B 约 14GB bf16）

---

## 11. vLLM 推理引擎原理

### 11.1 KV Cache 的本质

KV Cache 是 Transformer 自回归推理的核心优化：

```
没有 KV Cache：
  生成第 t 个 token 时，需要重算前 t-1 个 token 的 Key 和 Value 矩阵
  复杂度：O(t²)

有 KV Cache：
  保存历史 token 的 Key 和 Value（缓存在 GPU HBM 中）
  生成第 t 个 token 时，只需计算第 t 个 token 的 K/V，与历史 cache 拼接
  复杂度：O(t)（时间），空间 O(t × layers × heads × dim)
```

**vLLM 的 PagedAttention**：将 KV Cache 按页（page）管理，类似操作系统的虚拟内存，避免 KV Cache 的内存碎片化。

### 11.2 gpu_memory_utilization 的计算逻辑

```python
# vLLM 初始化时的内存分配逻辑（概念代码）
total_memory = gpu.total_memory           # e.g. 48 GB
model_weights_memory = load_model()       # e.g. 3.4 GB（1.7B bf16）
cuda_reserved = ...                       # CUDA framework overhead

free_after_model = total_memory - model_weights_memory - cuda_reserved
kv_cache_budget = free_after_model × gpu_memory_utilization  # 例：35 GB × 0.30 = 10.5 GB

# 将 kv_cache_budget 换算为 token 数
kv_per_token = layers × 2 × head_dim × num_heads × 2 bytes  # Qwen3-1.7B: 0.22 MB/token
max_tokens_in_cache = kv_cache_budget / kv_per_token         # 例：47727 tokens
```

**约束**：kv_cache_budget 必须能容纳至少一条 max_model_len 长度的序列，否则 vLLM 拒绝启动。

### 11.3 sleep/wake 机制

vLLM 在 veRL 的 colocate 模式下需要让出 GPU 给 FSDP 训练。通过 sleep/wake 机制切换：

| Level | sleep 行为 | wake_up 行为 | 适用 vLLM 版本 |
|-------|-----------|------------|------------|
| Level 1 | 解除虚拟地址映射，物理内存**不释放** | 重新映射，很快 | < 0.8.5 |
| Level 2 | 解除虚拟地址 + **释放物理内存** | 重新申请 + 映射，较慢 | >= 0.8.5 |

**Level 2 的 OOM 陷阱**：sleep 时物理内存被释放，wake_up 时需要重新申请**连续**物理内存块（cumem 机制）。如果此时 FSDP 已经占用了大量碎片化内存，找不到足够的连续块，wake_up 失败。

**根本解法**：`free_cache_engine=True` → 每步 rollout 结束后**完全销毁** vLLM 引擎（所有内存归还 OS），下步重建。代价：每步多几秒（重建引擎时间），但完全消除 cumem OOM。

---

## 12. 显存管理：单卡极限配置

### 12.1 单卡 L20-48GB 的显存分配

**Rollout 阶段**（vLLM 独占）：
```
vLLM 模型权重（bf16）:    1.7B × 2 = 3.4 GB
vLLM KV Cache:            ~8.7 GB（util=0.30）
CUDA framework:           ~1.5 GB
合计 rollout 峰值:        ~13.6 GB  （48GB 中 ~34GB 空闲）
```

**训练阶段**（FSDP 独占）：
```
actor 权重（bf16）:       3.4 GB
梯度（bf16）:             3.4 GB
Adam state（fp32，CPU）:  【offload，不占 GPU】
ref 权重（bf16，CPU）:    【offload，不占 GPU】
activation 峰值:          ~18 GB（PPO_MAX_TOKEN_LEN_PER_GPU=4096）
CUDA framework:           ~1.5 GB
合计训练峰值:             ~26.3 GB  ✓（<48GB）
```

**free_cache_engine=True** 确保两个阶段完全隔离，各自的峰值独立计算即可。

### 12.2 OPD 模式的额外开销

当 teacher（7B）参与时：

```
Update Actor 阶段（teacher 上 GPU 时）：
  actor 权重（bf16）:    0.5GB
  梯度 + activation:    ~8 GB（0.5B 小模型）
  teacher 权重（bf16）:  14 GB  ← 临时
  teacher activation:   ~4 GB  ← teacher forward 时
  合计峰值（teacher forward 时）: ~26.5 GB

teacher offload 后 student backward：
  actor 权重 + 梯度 + activation:  ~10 GB  ✓
```

**关键时序**：teacher forward 完后立即 offload（`teacher.cpu() + empty_cache()`），再做 student backward。两个高峰不重叠。

### 12.3 显存调参指南

| 参数 | 作用 | 单卡 1.7B 配置 | 单卡 0.5B+7B OPD |
|------|------|-------------|----------------|
| `model_dtype` | master weights 精度 | bfloat16 | bfloat16 |
| `optimizer_offload` | Adam 状态 CPU | True | True |
| `param_offload`（ref）| ref 权重 CPU | True | True |
| `free_cache_engine` | 销毁 vLLM 引擎 | True | True |
| `gpu_memory_utilization` | KV Cache 比例 | 0.30 | 0.25（给 teacher 留空间）|
| `max_model_len` | 最大序列长度 | 1024 | 1024 |
| `PPO_MAX_TOKEN_LEN_PER_GPU` | micro-batch token 数 | 4096 | 4096 |
| `enable_activation_offload` | 激活值 CPU offload | True（3B）| 视情况 |

---

## 13. pass@k 评估指标详解

### 13.1 为什么不用 accuracy

直接用准确率（accuracy）的问题：
- 每题只采样一次，方差极大（尤其是 pass@1）
- 无法评估模型在多次尝试时的"能力上界"

### 13.2 pass@k 的正式定义

来自 HumanEval 论文（Chen et al., 2021）的无偏估计：

```
pass@k = 1 - C(n-c, k) / C(n, k)

其中：
  n = 每题采样总数
  c = 正确答案数（通过全部测试用例的个数）
  k = 允许尝试次数
  C(a, b) = 组合数 = a! / (b! × (a-b)!)
```

**直觉推导**：
```
失败概率（k 次全失败）= C(n-c, k) / C(n, k)
     = [从 n 个样本中选 k 个，全部选中坏样本的概率]

pass@k = 1 - 失败概率
```

**本研究配置**：n=20，同时计算 pass@1 和 pass@5：
```python
from math import comb

def pass_at_k(n, c, k):
    if n - c < k:  # 坏样本不够选 k 个，则必然有好样本
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

# 例：20 次采样，12 次正确
pass_at_1 = pass_at_k(20, 12, 1)  # ≈ 0.6
pass_at_5 = pass_at_k(20, 12, 5)  # ≈ 0.93
```

### 13.3 为什么 n=20 而非直接 n=1（估计 pass@1）

直接 n=1 的问题：单次采样的方差太大。题目难度不同时，一道题的单次正确率可能是 0.7，另一道可能是 0.1。用 n=1 来估计每道题的 pass@1，聚合后的均值方差很大。

n=20 的无偏估计：通过多次采样，可以更精准地估计每道题的真实通过率，最终 pass@k 的估计更稳定。

### 13.4 val-core vs test pass@1 的偏差来源

**val-core**（训练期间）：
- greedy 采样（temperature=0，n=1）
- 在验证集（训练集前 250 条）上评估
- 快速，但有系统偏差

**偏差来源**：
1. **Greedy vs Sampling**：greedy 总是选最高概率的 token，当模型高度确定时，greedy 表现比 sampling 更好
2. **KL 约束效应**：KL loss 将 actor 约束在 ref 附近，ref 在很多题上的 greedy 输出是正确的，所以 val-core 偏高
3. **训练集 vs 测试集**：val-core 在训练集上评估，可能有一定程度的记忆效应

**规律**：
- 无 KL loss 时：val-core ≈ test pass@1，偏差 < 0.001（Exp 9d 验证）
- 有 KL loss 时：val-core ≈ test pass@1 + 0.03（系统性高估）

---

## 14. 代码执行沙箱设计

### 14.1 为什么需要沙箱

在 RL 训练中，奖励函数需要**执行模型生成的代码**。直接用 `exec()` 有三大风险：

```python
# 风险 1：无限循环（卡死训练进程）
model_output: "while True: pass"

# 风险 2：内存炸弹（OOM 杀死整个训练进程）
model_output: "x = [0] * (10 ** 12)"

# 风险 3：危险命令（破坏文件系统）
model_output: "import os; os.system('rm -rf /')"
```

**核心原则**：用**子进程**隔离执行，子进程的崩溃/超时/OOM 不影响主进程（训练进程）。

### 14.2 完整沙箱实现

```python
import multiprocessing
import resource

def _worker(code: str, test_case: str, result_queue: multiprocessing.Queue):
    """在子进程中执行代码，结果通过 Queue 传回"""
    try:
        # 保护 1：限制虚拟地址空间 512MB（防内存炸弹）
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
        
        # 保护 2：在空命名空间执行（防污染主进程变量）
        exec(compile(code + "\n" + test_case, "<string>", "exec"), {})
        result_queue.put(True)  # 通过所有测试
    except Exception:
        result_queue.put(False)  # 任何异常都算失败

def run_in_sandbox(code: str, test_case: str, timeout: int = 5) -> bool:
    """
    子进程隔离执行代码
    - 超时（5秒）自动 kill 子进程
    - 子进程 OOM 被系统 kill，主进程安全
    - 返回 True/False 表示测试是否通过
    """
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(code, test_case, q))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        p.join()
        return False
    return q.get() if not q.empty() else False
```

### 14.3 执行内容

```python
# 拼接代码和测试用例
full_script = code + "\n" + test_case

# 例：
# def add(a, b):          ← 模型生成的代码
#     return a + b
#
# assert add(1, 2) == 3   ← MBPP 的测试用例

# exec 执行时：
# - assert 通过 → 静默（True）
# - assert 失败 → AssertionError（被 except Exception 捕获，返回 False）
```

### 14.4 沙箱的局限性

| 风险 | 是否防住 | 说明 |
|------|---------|------|
| 无限循环 | ✅ 防住 | timeout=5s 后 kill |
| 内存炸弹 | ✅ 防住 | RLIMIT_AS=512MB |
| 文件系统读写 | ❌ 未防 | 子进程继承文件权限 |
| 网络访问 | ❌ 未防 | 无网络命名空间隔离 |
| fork 炸弹 | ❌ 未防 | 无进程数限制 |

对 MBPP 竞赛题场景已足够。生产环境应使用 Docker 或 gVisor 级别隔离。

---

## 15. 训练指标解读手册

### 15.1 核心指标含义

| 指标路径 | 含义 | 理想趋势 | 危险信号 |
|----------|------|---------|---------|
| `val-core/…/acc/mean@1` | 验证集准确率（最终目标）| 稳定上升 | 长期平坦或下降 |
| `critic/score/mean` | 训练批次平均 reward | 随训练上升 | 长期 ~0（奖励稀疏问题）|
| `actor/entropy` | 策略熵（多样性）| 缓慢下降，不骤降 | 骤降至 ~0 → collapse |
| `actor/pg_loss` | 策略梯度损失 | 正负波动，非零 | 长期 ~0 → 无梯度信号 |
| `actor/pg_clipfrac` | PPO clip 触发比例 | <0.1 | 持续 >0.2 → 学习率过大 |
| `actor/ppo_kl` | 与 ref 的 KL 散度 | 很小（<0.01）| 增大 → 偏离初始模型太远 |
| `response_length/mean` | 平均回答长度 | 随训练稳定（GSM8K 会下降）| 持续增加 → 模型在"水字数" |
| `bc_loss` | 行为克隆损失（OPD 专属）| 随训练下降 | NaN → teacher overflow 问题 |

### 15.2 GRPO 特有的梯度消失现象

**正常运行时**：
```
pg_loss:    -0.0023  # 非零，有梯度
pg_clipfrac: 0.0082  # 约 0.8% 的 token 被 clip
ppo_kl:      0.0004  # 极小的 KL 偏差
```

**全零批次时**（奖励稀疏期）：
```
pg_loss:     0.0000  # 零，无梯度
pg_clipfrac: 0.0000  # 无 clip
advantages:  全为 0  # 所有样本 reward 相同
```

**Entropy collapse 后**：
```
pg_loss:    ~0.0000  # 策略几乎不变，ρ≈1，所有优化方向都被 clip
actor/entropy: 0.01  # 接近 0，几乎只生成一种输出
```

### 15.3 OPD 额外指标

| 指标 | 含义 | 正常范围 |
|------|------|---------|
| `bc_loss` | teacher 认可分（越低越好）| 随训练从 ~0.58 下降至 ~0.32 |
| `bc_loss_coef` | BC loss 权重 | 0.1（实验值） |

BC loss 全程有效下降，表明 teacher 蒸馏信号被稳定吸收，是 OPD 训练健康的标志。

### 15.4 训练阶段的典型模式

**正常训练（SFT warmup 后的 GRPO）**：
```
Step 0:   val-core=0.36，entropy=0.65，score=0.28
Step 10:  val-core=0.37，entropy=0.55，score=0.31
Step 20:  val-core=0.38，entropy=0.42，score=0.33
Step 40:  val-core=0.40，entropy=0.27，score=0.36  ← entropy 下降但梯度仍有效
Step 70:  val-core=0.41（峰值），entropy=0.16，score=0.40
Step 80:  val-core=0.408，entropy=0.055，score=0.41  ← 轻微回落
```

**Entropy Collapse 模式**（纯 binary/partial RL）：
```
Step 0:   val-core=0.25，entropy=0.65
Step 10:  val-core=0.28，entropy=0.40
Step 20:  val-core=0.30（峰值！），entropy=0.12  ← collapse 开始
Step 30:  val-core=0.29，entropy=0.02  ← collapse 完成，梯度消失
Step 80:  val-core=0.28，entropy=0.001  ← 平台期，几乎不学习
```

关键特征：entropy collapse 发生后，val-core 的"峰值"是虚高的。此后的 val-core 反映的不是真正的学习，而是 greedy 采样对 collapsed policy 的偶然优势。
