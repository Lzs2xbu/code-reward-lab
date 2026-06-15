# RL for LLM 核心概念笔记

本文档记录在 codellmRL 项目中学到的 RL / 信息论概念，供后续复习和对比参考。

> **最终更新 2026-06-09**：项目后期发现早期 OPD 公式笔记里有一个关键误区：`teacher_log_probs` 是 teacher 对 student 生成 token 的评分，但它本身不提供 student 梯度。正确的 supervised OPD loss 必须包含 `log π_student_live`。

---

## 1. KL 散度：前向 vs 反向

### 基本定义

KL 散度衡量两个概率分布的"差异"，**不对称**（KL(P‖Q) ≠ KL(Q‖P)）：

```
KL(P ‖ Q) = E_P[log(P/Q)] = Σ P(x) * log(P(x)/Q(x))
```

### 前向 KL（Forward KL）

**定义**：KL(teacher ‖ student) = E_teacher[log(teacher/student)]

**特性：Mean-seeking（均值寻找）**

- 惩罚来源：若 teacher 对某处有概率但 student 没有 → log(teacher/0) = +∞，无穷大惩罚
- 结果：student **不敢遗漏** teacher 分布的任何 mode，被迫"撒网覆盖"
- 分布形态：student 的分布倾向于**宽泛、分散**，覆盖 teacher 的所有模式

**直觉**：老师说"这道题有三种解法"，学生必须把三种都学会，不能遗漏。

**适用场景**：
- 希望学生保留多样性（多种解法风格）
- 知识蒸馏中希望覆盖教师所有能力

**计算代价**：需要在 teacher 分布下采样，或已知 teacher 概率密度。

---

### 反向 KL（Reverse KL）

**定义**：KL(student ‖ teacher) = E_student[log(student/teacher)]

**特性：Mode-seeking（模式寻找）**

- 惩罚来源：若 student 对某处有概率但 teacher 没有 → log(student/0) = +∞，无穷大惩罚
- 结果：student **不敢"越界"**，只能在 teacher 有概率的地方放概率
- 分布形态：student 的分布倾向于**尖锐、聚焦**，选择 teacher 的某一个主要 mode 深度对齐

**直觉**：学生不敢写老师不认可的答案，因此专注于老师最可能给分的那种写法。

**适用场景**：
- 代码生成（需要提交确定的、正确的解法，不需要"模糊平均"）
- PPO/GRPO 的 KL penalty（防止策略走到 reference 不覆盖的地方）
- OPD 蒸馏（希望学生 commit 到教师的某个 high-quality mode）

**计算优势**：期望在 student 分布下取，可直接用 rollout 样本（已有的生成序列）计算，**无需额外采样**。

---

### 对比总结

| | 前向 KL | 反向 KL |
|--|---------|---------|
| 公式 | KL(P‖Q) = E_P[log P/Q] | KL(Q‖P) = E_Q[log Q/P] |
| 别名 | Inclusive KL, M-projection | Exclusive KL, I-projection |
| 行为 | Mean-seeking（覆盖所有 mode）| Mode-seeking（选择一个 mode）|
| student 分布 | 宽泛、均匀 | 尖锐、聚焦 |
| 计算方式 | 在 teacher 分布下取期望 | 在 student 分布下取期望 |
| 适用 | 保留多样性 | 提交确定答案 |
| 代码生成 RL | 一般不首选 | **首选** |

---

### 在本项目的使用

| 用途 | KL 方向 | 具体含义 |
|------|---------|---------|
| PPO KL penalty（现有） | Reverse | KL(student ‖ ref_initial)：防止策略走远 |
| OPD supervised loss | Reverse sampled-token KL | `mean(log π_student_live - log π_teacher.detach())`：引导学生向教师认可的 token 分布靠拢 |
| 前向 KL 变体（备用）| Forward | KL(teacher_7B ‖ student)：覆盖教师所有能力 |

---

## 2. Entropy（策略熵）

**定义**：H(π) = -E[log π(a|s)] = -Σ π(a) log π(a)

**含义**：衡量策略的**不确定性/多样性**。熵高 = 策略均匀分散；熵低 = 策略高度集中于某些 token。

### Entropy Collapse

**现象**：训练过程中策略熵单调下降，趋近于 0。

**原因**：GRPO 通过 advantage 加权更新，会不断增强 reward 高的 token 的概率，压低其他 token。随着策略越来越确定，探索空间越来越小，最终梯度几乎为零。

**在本项目的观察**：
- Binary reward：step 20 左右 entropy collapse
- Partial reward（无 KL）：step 40 左右 entropy collapse  
- Partial reward + KL loss：entropy 在 step 20 仍 ~0.046，且出现振荡（被 KL 约束住）

**应对方案**：
| 方案 | 原理 | 在本项目的可行性 |
|------|------|---------------|
| `entropy_coeff` | 直接最大化策略熵，加入到 loss | ❌ 需要全词表 softmax backward，~5GB 额外显存，OOM |
| `use_kl_loss` | 惩罚偏离 ref 策略，间接维持 entropy | ✅ 已实现，效果明显 |
| OPD supervised loss | teacher 软标签约束 sampled token；必须由 live student log-prob 提供梯度 | 已验证，A/B/C/D/E 均完成 |

---

## 3. GRPO（Group Relative Policy Optimization）

**与 PPO 的区别**：不使用 critic 网络估计 value，而是通过**组内相对排名**计算 advantage。

**流程**：
1. 同一 prompt 生成 N 个响应（rollout_n=5）
2. 计算每个响应的 reward
3. 组内 reward 归一化：advantage = (r - mean(r)) / std(r)
4. 按 PPO clip ratio 更新策略

**优点**：无需 critic，内存占用更少，适合单卡训练。

**稀疏奖励的问题**：若一个 prompt 的 N 个响应 reward 全为 0（或全为 1），则 advantage ≈ 0，梯度消失。这是我们项目早期（v1 数据集）训练无效的根本原因。

---

## 4. Partial vs Binary Reward

| | Binary | Partial |
|--|--------|---------|
| 设计 | 全通过=1，否则=0 | 通过率映射（0/0.2/0.6/1.0）|
| 信号稠密性 | 稀疏（全对或全错）| 较稠密（部分通过也有分）|
| 训练稳定性 | 信号清晰，收敛快（collapse 也快）| 信号丰富，但早期更嘈杂 |
| 在本项目 | step80 pass@1 = 0.3530 | step80 pass@1 = 0.3550（略优）|

---

## 5. KL Loss vs Entropy Coeff

两者都是为了防止 entropy collapse，但机制不同：

| | `entropy_coeff` | `use_kl_loss` |
|--|----------------|--------------|
| 目标 | 直接最大化 H(π) | 惩罚 KL(π ‖ π_ref)，间接维持 entropy |
| 梯度来源 | 需要全词表 log_softmax | 只需已选 token 的 log_prob（PPO 已有）|
| 额外显存 | ~5GB（全词表梯度张量）| ~0GB（复用 PPO 计算） |
| 在 1.7B 单卡上 | ❌ OOM | ✅ 可用（step 20 entropy 仍 0.046）|
| 效果对比 | 直接，但本项目无法使用 | 间接，但已观察到 entropy 振荡而非坍缩 |

---

## 6. Online Policy Distillation（OPD）训练流程

### 核心思路

让大模型（teacher）对小模型（student）每步生成的内容打分，引导 student 向 teacher 的输出分布靠拢，同时用 RL 强化真正能通过测试的写法。

### teacher_log_probs 的含义

teacher 做前向时，**输入是 student 自己生成的序列**，输出是 teacher 认为这段序列有多合理：

```
student rollout:  prompt + "def add(a, b):\n    return a + b\n"

teacher forward:
  input  = prompt_tokens + response_tokens
  logits = teacher(input)               # (B, seq_len, vocab_size)
  log_probs = log_softmax(logits)       # 对每个位置所有 token 的概率
  
  # 取 response 中「实际出现 token」的概率
  teacher_log_probs[i] = log_probs[i, prompt_len:][gather by actual token]
  # shape: (B, resp_len)，每个值表示 teacher 认为这个 token 在此出现的概率
```

**teacher_log_probs ≈ teacher 对 student 这段生成内容的「逐 token 认可分」**。值越接近 0，teacher 越认可。

### BC / sampled-token KL Loss 的作用

```python
L_BC = mean(log π_student_live(ŷ_t) - log π_teacher(ŷ_t).detach())
# teacher 认可的 token → student live log-prob 被推高
# teacher 不认可的 token → student live log-prob 被相对压低
L = L_GRPO + 0.1 × L_BC
```

- `L_GRPO`：强化「通过测试」的生成方式（从结果学）
- `L_BC`：让 student 的 token 分布向 teacher 靠拢（从过程学）

**重要坑**：`L_BC = -mean(teacher_log_probs)` 是错误实现，因为 teacher 分数与 student 参数无关，梯度为 0。teacher log-prob 只能作为 target/权重，student live log-prob 必须出现在 loss 中。

### 为什么叫 on-policy

teacher 评分的是**当前 student 版本生成的内容**，不是 teacher 自己写的代码。随着 student 能力提升，teacher 对其内容的评分反馈也在引导更高水平的写法。

### 每步完整流程（Scheme B 离线加载）

```
① rollout      student 生成 512 条代码（vLLM，~30s）
② log_prob     student 计算自己的 log_prob（teacher 在 CPU）
③ reward       执行代码，获得通过率奖励
④ advantage    GRPO 组内归一化
⑤ update_actor
   a. teacher.to("cuda")          ← PCIe: CPU→GPU (~14 GB，~5s)
   b. teacher 前向：计算 teacher_log_probs
   c. teacher.cpu() + empty_cache ← PCIe: GPU→CPU (~5s)  【Bug 8 修复点】
   d. student forward + backward  ← loss = GRPO + 0.1×BC
   e. optimizer.step()
```

每步 2 次 PCIe 传输（步骤 a/c），但 teacher 不参与 backward，不占 GPU backward 显存。

### 与 Exp 8（离线 OPD）的区别

| | Exp 8（离线）| Exp 9（在线）|
|--|------------|------------|
| teacher 参与时机 | 训练前预计算一次 | 每步 update_actor 时 |
| BC 数据 | teacher 自己生成的高质量样本 | student 当前 rollout |
| 优点 | 无额外 GPU 开销 | on-policy，随 student 成长 |
| 缺点 | 静态标签，student 进步后失效 | 每步 2 次 PCIe 传输 |

---

## 7. 浮点格式：float32 / bfloat16 / float16

### 浮点数的存储结构

所有 IEEE 754 浮点格式由三部分组成：

```
value = (-1)^符号 × 2^(指数-偏置) × (1 + 尾数)
```

三种格式的位布局：

```
float32  (32位):  [1位符号][8位指数][23位尾数]
bfloat16 (16位):  [1位符号][8位指数][ 7位尾数]   ← float32 的前 16 位
float16  (16位):  [1位符号][5位指数][10位尾数]
```

**bfloat16 = float32 直接截掉后 16 位尾数**，转换极其高效（无需运算，直接截断）。

---

### 关键特性对比

| 特性 | float32 | bfloat16 | float16 |
|------|---------|----------|---------|
| 总位数 | 32 | 16 | 16 |
| 指数位 | 8位 | **8位（同 fp32）** | 5位 |
| 尾数位 | 23位 | 7位 | 10位 |
| 最大值 | ~3.4×10³⁸ | **~3.4×10³⁸** | ~65504 |
| 十进制精度 | ~7位 | ~2-3位 | ~3-4位 |
| 显存/参数 | 4 bytes | **2 bytes** | 2 bytes |
| 溢出风险 | 无 | **无** | **有**（大模型 logits 易触发）|

**关键规律**：
- **最大值由指数位决定**（与尾数位无关）
- **精度由尾数位决定**
- bfloat16 用 7 位尾数换来了与 fp32 相同的数值范围，同时节省一半显存

---

### float16 溢出问题（本项目实际踩坑）

**Exp 9a / 9b 崩溃根因**：

```python
# 7B 模型以 float16 加载，GPU 推理时 logits 可能超出 65504
logits = teacher_model(input_ids)  # logits → inf（溢出）

# 对 [inf, inf, ..., inf] 做 log_softmax：
log_softmax([inf, ...]) = inf - log(sum(exp(inf)))
                        = inf - inf
                        = NaN  ← 崩溃根源

# NaN 传播：bc_loss=NaN → total_loss=NaN → loss.backward() → OOM crash
```

**修复**：将 teacher 加载改为 bfloat16，同内存开销，彻底消除溢出：

```python
# 修复前
self._teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_path, torch_dtype=torch.float16, ...)  # max=65504，溢出

# 修复后
self._teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_path, torch_dtype=torch.bfloat16, ...)  # max=3.4e38，安全
```

---

### 为什么神经网络训练中精度损失可以接受？

1. **SGD 本身就是随机的**：梯度噪声远大于 bfloat16 的精度误差
2. **LayerNorm / BatchNorm**：会归一化激活值，抑制精度误差累积
3. **大量参数的统计平均**：单个参数的精度误差在批次维度上相互抵消
4. **实践验证**：Google TPU 最早推广 bfloat16，现已成为大模型训练标准格式

---

### 在本项目的使用场景

| 场景 | 格式选择 | 原因 |
|------|---------|------|
| 0.5B 学生模型训练（FSDP）| bfloat16 | veRL 默认，节省显存，训练稳定 |
| 7B 教师模型加载（OPD 在线推理）| **bfloat16** | 防止 logits 溢出，显存与 float16 相同 |
| 4D attention mask 构建 | float16 | 仅用于 masked_fill，精度要求低，节省显存 |
| bc_loss 中的 log_softmax | float32 | `.float()` cast，在计算时用全精度防止中间结果 NaN |
