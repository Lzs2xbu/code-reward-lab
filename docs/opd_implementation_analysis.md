# OPD 多种实现方式分析与最终实验复盘

> **目的**：基于论文 "Rethinking On-Policy Distillation" (arXiv:2604.13016) 的开源代码，
> 对比多种 OPD 实现方式在 0.5B student / 1.5B teacher (MBPP GRPO best) 上的效果差异
> **最终状态**：Arm A/B/C/D/E 均已完成或确认不可行；最后更新 2026-06-09

---

## 实现方式总览

### 方式 A：Sampled-Token OPD as Supervised Loss（已实现，Arm B）

**论文**：Eq (3) 直接优化  
**代码对应**：`losses.py ppo_loss()` 中 bc_loss 段

```python
# 梯度来自 live student logits
bc_loss = mean(log π_student_live(ŷ_t) - log π_teacher(ŷ_t))
∂L/∂θ = ∂log π_student / ∂θ  ← 直接 minimize KL
```

- 在每个 actor update mini-batch 内计算（live logits，有梯度）
- Teacher log-probs：`_compute_teacher_log_probs_bc`（每次 mini-batch 调用）
- **当前状态**：已实现，Arm B 使用此方式运行完毕

---

### 方式 B：k1 OPD as Policy Gradient Reward（已实现，Arm C）

**参考代码对应**：`fsdp_workers.py` line 2720 (`rm_scores = -(student_logp - teacher_logp)`)  
→ `token_level_rewards = rm_scores` → GRPO advantage → policy gradient

**我们的近似实现（losses.py ppo_loss 内）**：

```python
# Arm C: k1 OPD as PG reward
# 参考代码：rm_scores = teacher_lp - student_lp_rollout  作为 GRPO token reward
#   = 教师更喜欢这个 token（相对学生）时 reward 高
#   → policy gradient: -rm_score × ∂log π_student/∂θ
#
# 我们的近似：用 π_teacher(ŷ_t) 直接作为 reward 权重（不减 student_lp）
# 等价于：β × log π_teacher 作为 shaped reward，最大化 log π_student 加权期望

L_C = -mean( π_teacher(ŷ_t).detach() × log π_student_live(ŷ_t) )
∂L_C/∂θ = -(π_teacher(ŷ_t)) × ∂log π_student / ∂θ
```

**与方式 A (supervised loss) 的核心差异**：

| | Arm B（supervised）| Arm C（k1 reward，已完成）|
|--|------------------------|--------------------------|
| **Loss** | `mean(log π_s - log π_T)` | `-mean(π_T × log π_s)` |
| **Gradient** | `∂log π_s / ∂θ`（每 token 等权）| `-(π_T(ŷ_t)) × ∂log π_s / ∂θ`（按教师概率加权）|
| **含义** | 直接最小化 KL 散度 | 优先对齐教师高概率 token，低概率 token 更新步长小 |
| **激活参数** | `bc_loss_coef=0.05` | `bc_reward_coef=0.05, bc_loss_coef=0` |

**实现位置**：`losses.py ppo_loss()` 中已有 `teacher_log_probs` 数据，一处修改

**与参考代码的差异说明**：
- 参考代码：`rm_scores = teacher_lp - student_rollout_lp`（KL 值作为 reward，通过 GRPO advantage 注入，在 advantage 计算前）  
- 我们：`π_teacher × log π_student_live`（类似 BC loss 的 PG 形式，在 actor update 内）  
- 两者 gradient 方向相同（都驱动 student 靠近 teacher），但 reward 幅度的归一化不同（GRPO 会对 reward 归一化，我们没有）

---

### 方式 C：forward_kl_topk OPD as Reward（top-K 单卡不可行；K=1 近似已完成，Arm D）

**论文**：Eq (5) 的 RL reward 变体  
**代码对应**：参考代码 `dp_actor.py compute_distillation_reward()` + `core_algos.py token_reward_direct`

```python
# 关键：student top-K 必须来自学生分布（参考代码从 rollout 存储，我们用 live logits）
S_t = TopK(π_student, K=16)              # ← 学生概率最高的 K 个 token（不是教师！）
S_logp = log π_student(S_t)             # (B, T, K=16) detached（参考代码从 rollout 取）
T_on_S = log π_teacher(S_t)             # (B, T, K) 教师在学生 top-K 位置的 log-prob
kl_val  = S_logp - T_on_S              # KL per top-K token，detached
weights = softmax(S_logp, dim=-1)       # 学生概率归一化权重
rm_scores = -kl_val * weights           # (B, T) 每 token 的 KL reward
# → REINFORCE: L = -mean(rm_scores × Σ_k log π_student_live(k))
```

**与 bc_topk (Arm B 变体) 的核心差异（token 集合相同，梯度路径不同）**：

| | bc_topk（Arm B 变体）| Arm D（forward_kl_topk reward）|
|--|---------------------|--------------------------------|
| Student top-K 来源 | live logits（有梯度） | live logits（有梯度）**相同** |
| KL 计算 | KL WITH student gradient | KL DETACHED（stop_grad） |
| Loss | `KL(p̄_student ‖ q̄_teacher)` 直接 backprop | `-KL_det × Σ_k log π_topK_live` REINFORCE |
| Gradient | `∂KL/∂θ`（直接优化 KL） | `-(rm_score) × ∂Σ_k log π_topK/∂θ` |

**最终工程结论**：
- K=16 top-K 版本在单卡 L20-48GB 上不可行，OOM 根因不在 teacher KL 计算，而在 PPO 反传阶段 `log π_new(top-K)` 的 `gather.backward`。
- K=1 sampled-token 近似已实现为 Arm D_direct / Arm D_plus_grpo，复用 PPO 已有 sampled-token `log_prob`，不引入新的全词表 gather 梯度路径。
- 服务器实现位置：`core_algos.py` 注册 `token_reward_direct` / `token_reward_direct_plus_grpo`；`ray_trainer.py` 根据 `bc_shaped_reward_mode=direct|plus_grpo` 注入 token-level reward。

---

## 实验设计

### 对比矩阵

| 实验 | OPD 方式 | 激活参数 | gradient 路径 | 理论差异 | 状态 |
|------|---------|---------|--------------|---------|------|
| Arm A | GRPO only（对照） | 无 | GRPO PG | — | ✅ 完成 |
| Arm B | Sampled-token supervised | `bc_loss_coef=0.05` | 直接 ∂KL/∂θ | 均匀更新 | ✅ 完成 |
| Arm C | k1 reward (policy gradient) | `bc_reward_coef=0.05` | teacher-weighted PG | 教师概率加权，低置信 token 权重小 | ✅ 完成 |
| Arm D_topK | top-K PG reward（student top-K，KL detached）| `bc_topk_reward=16, coef=0.05` | KL_detached × ∂log π_topK/∂θ | 低方差 K-token 估计 | ❌ 单卡 OOM |
| Arm D_direct | sampled-token `token_reward_direct` | `adv_estimator=token_reward_direct` | rm_score 直接作为 advantage | K=1 近似，无 GRPO task advantage | ✅ 完成 |
| Arm D_plus_grpo | sampled-token direct + GRPO | `adv_estimator=token_reward_direct_plus_grpo` | rm_score + GRPO outcome advantage | K=1 近似，task reward 约束方向 | ✅ 完成 |

A/B/C/E 使用 0.5B student、1.5B MBPP GRPO best teacher、LR=5e-6、16 epochs；Arm D 使用 1.5B-aligned SFT warmup，并验证 K=1 `token_reward_direct` 的 direct / plus_grpo 两种 estimator。比较 Arm D 与 A/B/C 时必须保留“起点不同”的限定。

### 最终结论

- Arm B/C 相比纯 GRPO 的 pass@1 提升均在 1 SE 内，不构成强显著结论。
- Arm B 等权 KL loss 会让 pass@5 下降，说明分布收窄；Arm C 教师概率加权能较好保留 pass@5。
- Arm E shaped reward 进入 GRPO advantage 后结构性崩溃，说明 OPD reward 与 task reward 的方向一致性必须先验证。
- Arm D sampled-token direct 可稳定训练；D_plus_grpo 最优，说明 token-level teacher 信号需要 task reward 共同约束。

---

## 代码修改与实现记录

### 文件 1：`$VERL_DIR/verl/workers/config/actor.py`

```python
# 在 FSDPActorConfig 中添加：
bc_reward_coef: float = 0.0   # Arm C: k1 reward coef
bc_topk_reward: int = 0       # Arm D: top-K reward k
bc_topk_reward_coef: float = 0.0  # Arm D: top-K reward coef
```

### 文件 2：`$VERL_DIR/verl/workers/engine_workers.py`

```python
# update_actor() 中，在 actor.train_mini_batch 之前：

# Arm C: k1 reward path
bc_reward_coef = getattr(self.config.actor, 'bc_reward_coef', 0.0)
if bc_reward_coef > 0 and _teacher_path:
    data = self._compute_teacher_log_probs_bc(data)
    self._teacher_model = self._teacher_model.cpu()
    torch.cuda.empty_cache()
    # 加到 advantages（response 部分）
    teacher_lp = data["teacher_log_probs"]  # (B, resp_len)
    # old_lp_resp = 从 data["old_log_probs"] 切出 response 部分
    # kl_reward = teacher_lp - old_lp_resp
    # data["advantages"][:, -resp_len:] += bc_reward_coef * kl_reward
    # 简化版：直接用 teacher_lp 作为 reward（不减 student）
    assign_non_tensor_data(data, "bc_k1_reward", teacher_lp)
    assign_non_tensor_data(data, "bc_reward_coef", bc_reward_coef)

# Arm D: top-K reward path
bc_topk_reward = getattr(self.config.actor, 'bc_topk_reward', 0)
bc_topk_reward_coef = getattr(self.config.actor, 'bc_topk_reward_coef', 0.0)
if bc_topk_reward > 0 and _teacher_path:
    data = self._compute_teacher_topk_data(data, k_large=max(bc_topk_reward * 8, 128))
    # 计算 rm_scores（3D）
    # 然后叠加到 advantages 或替换
    assign_non_tensor_data(data, "distillation_use_topk_reward", True)
    assign_non_tensor_data(data, "bc_topk_reward", bc_topk_reward)
    assign_non_tensor_data(data, "bc_topk_reward_coef", bc_topk_reward_coef)
```

### 文件 3：`$VERL_DIR/verl/workers/utils/losses.py`

```python
# ppo_loss() 中，在现有 bc_loss 段之后，添加 k1 reward 路径：

# Arm C: k1 reward (policy gradient, weighted by teacher log-prob)
bc_reward_coef = get_non_tensor_data(data, "bc_reward_coef", 0.0)
if bc_reward_coef > 0 and "bc_k1_reward" in non_tensor_data:
    teacher_lp = get_non_tensor_data(data, "bc_k1_reward")
    # Policy gradient 形式：L = -mean(teacher_lp * log_student)
    # 等价于：以 teacher_lp 为权重最大化 student 的 log-prob
    k1_reward_loss = -agg_loss(
        loss_mat=(teacher_lp.detach() * log_prob),  # log_prob = live student log-probs
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
    )
    policy_loss = policy_loss + bc_reward_coef * k1_reward_loss
    metrics["actor/bc_k1_reward_loss"] = k1_reward_loss.detach()

# Arm D: top-K reward 
# 需要在 prepare_model_outputs 阶段计算 rm_scores（通过 logits_processor_func）
# 然后在这里：
# advantages_3d = advantages.unsqueeze(-1) + bc_topk_reward_coef * rm_scores
# policy_loss_3d = -mean(advantages_3d * ratio_3d)
```

---

## 参考代码 OPD 完整链路（文档备份）

### k1 模式（sampled-token，默认 opd.sh）

```
fsdp_workers.py line 2720-2721:
  rm_scores = -(student_logp - teacher_logp)    ← 2D (B, T), detached
      ↓ 存入 batch["rm_scores"]
ray_trainer.py line 1365:
  token_level_rewards = token_level_scores       ← 任务 reward 或 rm_scores（pure OPD）
      ↓
compute_advantage(grpo):
  advantages = GRPO_normalize(token_level_rewards)
      ↓
dp_actor.py update_policy():
  pg_loss = -advantages * ratio                  ← standard 2D PPO
```

### forward_kl_topk 模式（top-K，top_k=16）

```
ray_trainer.py line 1111:
  compute_log_prob(batch) → student_top_k_ids (B,T,K), student_top_k_log_probs (B,T,K)
      ↓
fsdp_workers.py _compute_teacher_top_k_log_probs() line 1891:
  chunk_log_probs = gather(logits, student_ids) - logsumexp  ← 全词表归一化
  → teacher_on_student_log_probs (B, T, K)                  ← 不重归一化
      ↓
dp_actor.py compute_distillation_reward() line 561-564:
  kl_val = S_logp - T_on_S
  weights = softmax(S_logp[valid]) / logsumexp  ← softmax over K
  rm_scores = -kl_val * weights                 ← (B, T, K) 3D KL reward
      ↓
ray_trainer.py line 1365 (token_reward_direct):
  token_level_rewards = rm_scores               ← 3D
      ↓
core_algos.py compute_token_reward_direct_advantage() line 875-878:
  if rewards.dim() == 3: mask = mask.unsqueeze(-1)
  advantages = rewards * mask                   ← (B, T, K) 3D
      ↓
dp_actor.py line 1118-1144 (3D PPO):
  ratio = exp(log_prob_K - old_log_prob_K)      ← (B, T, K)
  pg_losses = -advantages * ratio
  pg_losses = pg_losses.sum(dim=-1)             ← sum over K → (B, T)
  policy_loss = masked_mean(pg_losses, mask)
```

---

## 参考代码 top-K 方案详解（forward_kl_topk）

### 核心设计思路

top-K OPD 每个时间步使用 K=16 个 student top-K token 的 KL 散度作为 reward，而不是单个采样 token。这降低了 KL 估计的方差，提供更精确的梯度信号。

### 关键技术细节

#### 为什么不会 OOM（多卡环境）

**关键**：rm_scores 的计算链路**全程 detached**（无梯度），OOM 来自梯度路径：

```python
# fsdp_workers.py _compute_teacher_top_k_log_probs() — detached，无 OOM
t_logsumexp = logsumexp(teacher_logits, dim=-1, keepdim=True)  # 全词表归一化
teacher_on_student = gather(teacher_logits, student_ids) - t_logsumexp  # gather 无梯度

# dp_actor.py compute_distillation_reward() — detached，无 OOM
S_logp = student_top_k_log_probs    # rollout 时存储，detached
T_on_S = teacher_on_student_logps   # detached
kl_val  = S_logp - T_on_S          # (B, T, K) detached
weights = softmax(S_logp, dim=-1)   # detached
rm_scores = -kl_val * weights       # (B, T, K) detached，无梯度
```

**OOM 出现在 PPO policy gradient**，梯度需要流过 `log π_new(k)` for K top-K tokens：

```python
# token_reward_direct PPO loss (3D case) — 梯度路径
advantages = rm_scores  # (B, T, K) detached
log_prob_K = F.log_softmax(logits, dim=-1).gather(-1, top_k_indices)  # WITH gradient
                                                                        # ← gather.backward 需要 (nnz, vocab) 张量 = 4.76GB OOM!
ratio = exp(log_prob_K - old_log_prob_K)
pg_loss = -sum_k(advantages × ratio)
```

**参考代码为什么不 OOM**：在 4×GPU 环境，每卡 nnz = 全 nnz / 4。`nnz × vocab × 4 bytes = (7829/4) × 151936 × 4 ≈ 1.19GB`，在 40GB 卡上可以接受。

**我们单卡 nnz 是 4 倍，4.76GB 超出剩余显存，无法运行。**

#### 完整链路（参考代码 forward_kl_topk 模式）

```
Step 1: rollout 时存储 student top-K（也可在 compute_log_prob 阶段）
  student_top_k_ids        (B, T, K=16)  — rollout detached
  student_top_k_log_probs  (B, T, K=16)  — rollout detached

Step 2: teacher 在 student top-K 位置计算 log-probs（fsdp_workers.py）
  teacher_on_student_log_probs (B, T, K) — 全词表 logsumexp 归一化，detached

Step 3: 计算 rm_scores（dp_actor.py compute_distillation_reward，detached）
  kl_val[t,k]  = S_logp[t,k] - T_on_S[t,k]         # KL per top-K token
  weights[t,k] = softmax(S_logp[t,:], dim=-1)[k]    # 学生概率归一化权重
  rm_scores[t] = sum_k(-kl_val[t,k] × weights[t,k]) # (B, T) 聚合后 2D

  注意：参考代码 opd.sh 实际使用 3D rm_scores (B, T, K) 或聚合后 2D (B, T)

Step 4: token_reward_direct 计算 advantage（core_algos.py）
  advantages[t] = rm_scores[t]  or  advantages[t,k] = rm_scores[t,k]  # 无 GRPO 归一化

Step 5: PPO policy gradient（dp_actor.py, 3D case）
  ratio = exp(log π_new(top-K) - log π_old(top-K))  ← 梯度在此，需要 gather.backward → OOM
  L = -sum_k(advantages × ratio)
  pg_loss = masked_mean(L, mask)
```

#### 与 k1 模式（我们的 Arm D_direct）的差异

| 维度 | k1（sampled-token）| top-K（K=16）|
|------|-------------------|--------------|
| Token 覆盖 | 1 个（rollout 采样）| K=16 个（student top-K）|
| KL 估计方差 | 高（单样本 MC 估计）| 低（K 样本，更精确）|
| gradient 路径 | 通过 1 个 log π_new(ŷ_t) | 通过 K 个 log π_new(k) |
| 单卡 OOM | 无（reuses existing） | 有（gather.backward × K 次）|
| 参考代码可行性 | 4×GPU ✓ / 单卡 ✓ | 4×GPU ✓ / **单卡 ✗** |

---

## 新增实验（token_reward_direct 系列，2026-06-09）

基于上述分析，设计两个新实验，真正实现 `token_reward_direct` 机制：

### Arm D_direct（纯 token_reward_direct，无 GRPO）

```
advantages[t] = rm_score[t]   # 直接用 KL reward，无 group normalization
L = PPO(advantages)           # 梯度：-rm_score[t] × ratio[t] × ∂log π(ŷ_t)/∂θ
```

- 不依赖 GRPO——rm_scores 直接作为 advantage，无 group normalization
- 纯 OPD 驱动：模型只靠 teacher 对齐信号学习，无 task reward 在 advantage 中
- Task reward 仍计算（用于 val 监控），但不进入 advantage

### Arm D_plus_grpo（token_reward_direct_plus_grpo）

```
advantages[t] = rm_score[t] + grpo_outcome_weight × GRPO_task_adv[i]
L = PPO(advantages)
```

- 融合 per-token OPD 信号（不经 GRPO 归一化）+ per-sequence GRPO task 信号（经 GRPO 归一化）
- `grpo_outcome_weight=1.0`（默认）

### 两种方式的梯度对比

```
Arm D_direct:    -rm_score[t] × ratio[t] × ∂log π(ŷ_t)/∂θ
Arm D_plus_grpo: -(rm_score[t] + w × GRPO_adv[i]) × ratio[t] × ∂log π(ŷ_t)/∂θ
Arm C:           -π_T(ŷ_t) × ∂log π(ŷ_t)/∂θ          （无 ratio，不经 PPO clipping）
```

**Arm D_direct/plus_grpo vs Arm C 的本质区别**：
- Arm C：OPD 是 supervised loss，**不经 PPO ratio clipping**
- Arm D：OPD 进入 advantage，**经 PPO ratio clipping**（有信赖域保护）
- Arm D 的 rm_score 可正可负（双向），Arm C 的 π_T 权重恒正（单向）

---

## 最终实验结果（正式 eval，n=20，T=0.8）

| 实验 | 起点 | MBPP pass@1 best | MBPP pass@5 best | MBPP pass@1 step80 | MBPP pass@5 step80 | 备注 |
|------|------|-----------------:|-----------------:|-------------------:|-------------------:|------|
| 0.5B SFT warmup（7B 数据） | 7B SFT | 0.2727 | 0.3991 | — | — | 早期 OPD 起点 |
| Arm A: GRPO | 0.5B SFT warmup | 0.3188 | **0.3974** | 0.2852 | 0.3349 | 纯 GRPO 对照 |
| Arm B: supervised KL | 0.5B SFT warmup | 0.3218 | 0.3827 | 0.3077 | 0.3604 | pass@5 收窄 |
| Arm C: teacher-weighted PG | 0.5B SFT warmup | 0.3287 | 0.3951 | 0.2966 | 0.3380 | best=step20 |
| Arm E: shaped reward | 0.5B SFT warmup | 崩溃 | — | — | — | β=1/0.001/方差归一化均崩溃 |
| Arm E v2: shaped reward | 1.5B-aligned SFT | 崩溃 | — | — | — | step10=0.031，step20=0 |
| Arm D_direct: token_reward_direct | 1.5B-aligned SFT | 0.3501 | 0.4286 | **0.3516** | **0.4360** | adv=rm_score，无 GRPO |
| **Arm D_plus_grpo** | 1.5B-aligned SFT | **0.3607** | **0.4471** | 0.3535 | 0.4272 | adv=rm_score + GRPO task advantage |
| Arm D_topK: top-K reward (K=16) | 1.5B-aligned SFT | OOM | — | — | — | 单卡 vocab 梯度限制 |

### 如何解读 Arm D

Arm D_direct / D_plus_grpo 与 Arm A/B/C 使用的 SFT 起点不同，因此不能直接写成严格同起点消融；它们回答的是另一个问题：**在单卡不能跑 K=16 top-K 的条件下，K=1 sampled-token token_reward_direct 是否可行，以及是否需要叠加 task reward**。

结论：

- K=1 `token_reward_direct` 可行，且不会出现 Arm E 的 response length 正反馈崩溃。
- `plus_grpo` best `pass@1=0.3607`、`pass@5=0.4471`，优于 direct best `0.3501/0.4286`，说明 task reward 对 OPD reward 的方向约束仍然重要。
- top-K 版本仍建议留给多卡环境验证；单卡下继续压 batch 或 K 只会牺牲实验可比性。
