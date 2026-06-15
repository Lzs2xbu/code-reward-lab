# 实验报告：基于 veRL-GRPO 的代码生成强化学习

**框架**：veRL + GRPO + vLLM | **硬件**：单张 L20-48GB GPU  
**主线数据集**：MBPP v2 / APPS / LiveCodeBench  
**时间**：2026-05-25 ~ 2026-06-09 | **状态**：全部训练与正式评测完成  

---

## 0. 最终结论速览

本项目从单卡 veRL 跑通开始，逐步完成 MBPP 快速消融、APPS 训练迁移、LiveCodeBench 泛化评测，以及 OPD 正确性验证。最终结论可以压缩成四条：

1. **SFT 数据质量决定 RL 上限**：在 APPS+LCB 主实验中，7B teacher 生成的 SFT 数据明显优于 human SFT。Teacher SFT + GRPO 达到 LCB `pass@1=13.1%`、`pass@5=25.7%`，是最终最优。
2. **MBPP 上 LR 和 SFT warmup 比 OPD 名义算法更关键**：1.5B 在 LR=5e-6、SFT warmup 后纯 GRPO 已达到 `45.33%/55.76%`；历史 OPD 配置与其差距仅 `+0.66pp/+0.20pp`，不构成显著增益。
3. **OPD 的实现位置决定成败**：loss/actor update 后的 OPD 是参数级正则，能稳定训练；shaped reward 进入 GRPO advantage 后会改变 rollout 竞争标准，Arm E 系列全部崩溃。
4. **token_reward_direct 可工作，但 top-K 版本受单卡显存结构限制**：K=1 sampled-token 近似最终跑通，Arm D_plus_grpo 达到 MBPP `pass@1=36.1%`、`pass@5=44.7%`；K=16 top-K 需要对多个 token 的 `log π_new` 反传，单卡会在 `gather.backward` 上 OOM。

---

## 1. 正式结果表

### 1.1 LiveCodeBench 主结果（1.5B，APPS 训练，n=5，T=0.8）

| 模型/阶段 | LCB pass@1 | LCB pass@5 | Easy | Medium | Hard | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 1.5B Instruct baseline | 7.0% | 14.3% | 24.2% | 0.0% | 2.8% | 主基线 |
| 7B Instruct baseline | 21.1% | 41.0% | 63.3% | 8.3% | 8.8% | 教师上界 |
| Human SFT warmup | 6.1% | 13.3% | 22.5% | 0.0% | 1.8% | pass@5 低于 baseline，分布收窄 |
| Human SFT + GRPO best | 9.9% | 19.0% | 30.8% | 0.8% | 4.9% | GRPO 修复 human SFT 分布收窄 |
| Human SFT + OPD best | 8.4% | 19.0% | 25.8% | 4.2% | 2.8% | 与 GRPO 无显著差异 |
| Teacher SFT warmup | 11.0% | 20.0% | 33.3% | 2.5% | 5.3% | 单独超过 Human SFT+GRPO pass@1 |
| **Teacher SFT + GRPO best** | **13.1%** | **25.7%** | **35.8%** | **5.0%** | **7.0%** | **最终最优** |

核心增益：

- Teacher SFT + GRPO vs 1.5B baseline：`pass@1 +6.1pp`，`pass@5 +11.4pp`。
- Teacher SFT + GRPO vs Human SFT + GRPO：`pass@1 +3.2pp`，`pass@5 +6.7pp`。
- Teacher SFT + GRPO 把 1.5B 与 7B 的 pass@1 差距从 `14.1pp` 缩小到 `8.0pp`。

### 1.2 MBPP 1.5B 消融（n=20，T=0.8）

| 实验 | 起点 | 方法 | LR | MBPP pass@1 | MBPP pass@5 | 归因 |
|---|---|---|---:|---:|---:|---|
| 1.5B Instruct baseline | — | 无训练 | — | 33.44% | 48.57% | 基线 |
| SFT warmup | MBPP SFT | SFT only | — | 40.64% | 51.32% | SFT 本身有明显收益 |
| Exp 12 | SFT warmup | GRPO | 1e-6 | 41.04% | 51.88% | LR 太小，几乎无 RL 增益 |
| Exp 14 | 无 warmup | GRPO | 5e-6 | 40.64% | 50.84% | 无 SFT 也能学到 pass@1，但 pass@5 略弱 |
| **Exp 15** | SFT warmup | GRPO | **5e-6** | **45.33%** | **55.76%** | **LR 对齐后的纯 GRPO 主结果** |
| Exp 11b/v2 | SFT warmup | OPD/BC 配置 | 5e-6 | 45.99% | 55.96% | 与纯 GRPO 差距不显著 |
| 7B Instruct teacher | — | 无训练 | — | 49.74% | 59.44% | MBPP 上界参考 |

注意：历史 1.5B OPD 路线中曾发现 `teacher_log_probs` 作为 detached 常数使用导致 OPD 梯度为零的问题，因此这些结果不能作为“OPD 有显著增益”的强证据。更可靠的表述是：**在 LR 与 SFT 起点对齐后，OPD/BC 配置与纯 GRPO 差异落在统计噪声范围内**。

### 1.3 MBPP 0.5B OPD 正确性验证（n=20，T=0.8）

| Arm | 起点 | 方法 | best pass@1 | best pass@5 | step80 pass@1 | step80 pass@5 | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| A | 0.5B SFT warmup | GRPO only | 31.88% | 39.74% | 28.52% | 33.49% | 对照 |
| B | 0.5B SFT warmup | supervised KL loss | 32.18% | 38.27% | 30.77% | 36.04% | pass@5 收窄 |
| C | 0.5B SFT warmup | teacher-weighted PG loss | 32.87% | 39.51% | 29.66% | 33.80% | pass@5 接近对照 |
| E 系列 | 0.5B SFT warmup | shaped reward before GRPO | 崩溃 | — | — | — | response 变长、val 崩溃 |
| D_direct | 1.5B-aligned SFT warmup | token_reward_direct | 35.01% | 42.86% | 35.16% | 43.60% | K=1 direct 可稳定训练 |
| **D_plus_grpo** | 1.5B-aligned SFT warmup | token_reward_direct + GRPO | **36.07%** | **44.71%** | 35.35% | 42.72% | **Arm D 最优** |
| D_topK | 1.5B-aligned SFT warmup | top-K K=16 reward | OOM | — | — | — | 单卡 top-K 梯度路径不可行 |

Arm D 使用 1.5B-aligned SFT warmup，与 A/B/C 起点不同；因此 D 与 A/B/C 不应被写成完全干净的同起点消融。它更适合支撑两个结论：

- `token_reward_direct` 的 K=1 sampled-token 近似在工程上可行，且不会像 Arm E 那样崩溃。
- `plus_grpo` 比纯 direct 更好，说明 **token-level OPD 信号仍需要 task reward 约束方向**。

### 1.4 MBPP 1.7B 早期探索（n=20，T=0.8）

| 实验 | 方法 | pass@1 | pass@5 | 作用 |
|---|---|---:|---:|---|
| Qwen3-1.7B v2 baseline | 无训练 | 33.75% | 39.34% | 数据修复后的基线 |
| Binary GRPO | binary reward | 35.30% | 40.44% | 验证 RL 可提升 |
| Partial GRPO final | partial reward | 35.50% | 42.27% | partial 略优 |
| KL loss final | GRPO + KL | 35.24% | 40.83% | 推迟 collapse，但最终不优 |
| Offline SFT + GRPO | 7B 生成样本 SFT 后 GRPO | 37.86% | 44.78% | 1.7B 最优，说明 SFT warmup 有效 |

---

## 2. 实验路线复盘

### Phase 1：管道与数据对齐

先在 GSM8K 上跑通 veRL + GRPO，再迁移到 MBPP。早期最大问题不是算法，而是数据/评测不对齐：MBPP v1 prompt 没告诉模型函数名，assert 语句却要求精确函数名，导致 reward 大面积为 0。v2 数据从 assert 提取函数名写入 prompt 后，baseline reward 与评测恢复正常。

### Phase 2：MBPP 快速消融

MBPP 用于低成本验证：

- binary vs partial reward：差距有限，partial 在 pass@5 上略好。
- KL/entropy：entropy_coeff 需要全词表 backward，在单卡上 OOM；KL loss 可用但不是最终最优。
- SFT warmup：显著减少 entropy collapse，是让 RL 继续学习的关键起点。
- LR：1e-6 在 1.5B 上更新过小，5e-6 才能释放 GRPO 效果。

### Phase 3：迁移到 APPS + LiveCodeBench

MBPP 上的最优经验迁移到 APPS 后，出现一个新现象：**human SFT warmup 的 greedy 表现可提升，但 temperature sampling 的 pass@5 反而低于 instruct baseline**。这说明 APPS human solution SFT 可能让模型分布变窄。改用 7B teacher 生成的多样化 SFT 数据后，Teacher SFT 单独已明显强于 Human SFT，叠加 GRPO 后成为 LCB 最优。

### Phase 4：OPD 实现正确性验证

阅读 *Rethinking On-Policy Distillation* 与 thunlp/OPD 参考代码后，OPD 被拆成多个实现位点：

- **Arm B/C：OPD as loss**，在 GRPO advantage 之后作为 actor update 的参数级正则。
- **Arm E：OPD as shaped reward**，在 GRPO advantage 之前混入 token-level reward。
- **Arm D：token_reward_direct**，不做 GRPO group normalize，直接把 sampled-token KL reward 当作 token-level advantage；plus_grpo 再叠加 task advantage。

最终发现：Arm E 的 shaped reward 在本代码生成任务里结构性崩溃；Arm D 的 K=1 direct 可工作，plus_grpo 最优；top-K 版本因为单卡梯度路径 OOM 不可行。

---

## 3. 关键技术结论

### 3.1 SFT 不是必要条件，但高质量 SFT 是上限放大器

MBPP 上，无 warmup 直接 GRPO 在 LR=5e-6 时 `pass@1=40.64%`，与 SFT only 持平，说明 RL 本身能学到有效策略。但 SFT warmup 对 pass@5 和稳定性有额外贡献；在 APPS+LCB 上，SFT 数据质量进一步决定最终上限。

更准确的表述是：

- **SFT warmup 不是 RL 成功的必要条件**。
- **高质量且多样化的 SFT warmup 是提升 RL 上限的重要条件**。

### 3.2 OPD as loss vs OPD as reward 是机制差异，不是实现细节

OPD as loss：

```
L = L_GRPO(task advantage only) + beta * L_OPD
```

GRPO 仍只根据 task reward 判断哪个 rollout 更好；OPD 是额外的参数级约束。

OPD as reward：

```
advantage = GRPO_normalize(task_reward + beta * opd_token_reward)
```

OPD 改变了 rollout 的竞争标准。若 teacher 偏好更长或更复杂的代码，GRPO 会选择更长 rollout，策略进一步变长，形成正反馈。Arm E 的崩溃说明：**把 teacher 偏好直接当 reward 前，必须验证 teacher preference 与 task reward 是否方向一致**。

### 3.3 token_reward_direct 的经验

Arm D_direct 和 D_plus_grpo 的结果说明 sampled-token `token_reward_direct` 可以作为 top-K OPD 的单样本近似：

- direct 模式只靠 teacher KL reward，能稳定到 `35.2%/43.6%`。
- plus_grpo 叠加 task advantage，best 达到 `36.1%/44.7%`，说明 task reward 对方向约束仍有价值。
- top-K K=16 理论上方差更低，但单卡需要对 K 个 token 的 `log π_new` 反传，`gather.backward` 的 `(nnz, vocab)` 梯度张量约 4.76GB，超过剩余显存。

---

## 4. 工程经验

| 问题 | 现象 | 最终处理 |
|---|---|---|
| MBPP v1 函数名缺失 | reward 大面积为 0 | 生成 v2 prompt，显式写入函数名 |
| veRL chat format | prompt 被 tokenizer 当成异常格式 | parquet 中 prompt 使用 `list[dict]` chat format |
| Qwen3 `<think>`/代码块 | `exec()` 直接失败 | reward 中先抽取纯 Python 代码 |
| 单卡 3B backward OOM | FSDP fp32 master weights 占用过高 | `model_dtype=bfloat16` + activation offload + reduce dtype |
| vLLM wake_up cumem OOM | validation 后 update_weights 崩溃 | `free_cache_engine=True` 隔离 rollout 与训练显存峰值 |
| entropy_coeff OOM | 全词表 backward 额外显存过高 | 放弃 entropy loss，优先 SFT/OPD/KL 约束 |
| teacher fp16 NaN | logits overflow，bc_loss NaN | teacher 用 bfloat16 |
| OPD teacher 留在 GPU | student backward OOM | teacher 前向后立即 offload CPU |
| 历史 OPD 梯度为零 | detached `teacher_log_probs` 误作 loss | 改为 live student log_prob 参与梯度，重新设计 A/B/C/D/E |
| Arm E shaped reward 崩溃 | response 变长、val acc 归零 | 确认是机制问题，停止该路线 |
| Arm D top-K OOM | `gather.backward` 申请大梯度张量 | 单卡保留 K=1 sampled-token 近似 |

---

## 5. 项目表述建议

最短版本：

> 我从零搭了一套代码生成 RL pipeline，用 MBPP 做低成本消融，再迁移到 APPS 训练、LiveCodeBench 评测。最终发现 Teacher SFT 数据质量比单纯算法替换更关键：7B teacher 生成的 SFT 数据叠加 GRPO 后，1.5B 在 LCB 上达到 pass@1 13.1%、pass@5 25.7%。同时我系统复现和修正了 OPD 的几种实现，发现 OPD as loss 比较稳定，shaped reward 进入 GRPO advantage 会因为 teacher 偏好长代码导致策略漂移；token_reward_direct 的 K=1 近似可跑通，但 top-K 版本在单卡上受显存结构限制。

可复用的技术亮点：

- **实验设计**：MBPP 快速消融 → APPS 训练 → LCB 防污染评测。
- **算法理解**：GRPO、SFT warmup、OPD as loss/reward、token_reward_direct。
- **工程能力**：veRL/vLLM/FSDP 单卡显存管理、代码执行沙箱、checkpoint merge/eval 自动化。
- **负结果复盘**：OPD shaped reward 崩溃不是单纯 bug，而是 reward 方向与 task objective 冲突。

