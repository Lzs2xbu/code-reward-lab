# Code LLM RL 项目方案

> 目标：基于 veRL 框架，用强化学习训练代码生成模型，作为代码生成 RL 实践项目

---

## 1. 项目定位

### 核心问题
现有代码大模型在训练时通常只用监督学习（SFT），无法从"运行结果"中学习。强化学习可以让模型直接以"代码能否通过测试"为目标进行优化。

### 项目目标
在单张 L20-48GB GPU 上，基于 veRL + GRPO，对 Qwen2.5-Coder-7B 做 RL 后训练，引入**分层测试奖励**或**多轮调试 RL** 的创新点，在 MBPP/APPS 上取得超越 SFT baseline 的结果。

### 项目价值点
- 端到端实现了 Code RL 训练 pipeline
- 设计并验证了创新奖励函数
- 使用工业级框架 veRL（ByteDance 出品，21k+ stars）
- 对比实验证明 RL 提升效果

---

## 2. 技术选型

### 框架：veRL
- **GitHub**: https://github.com/verl-project/verl
- **文档**: https://verl.readthedocs.io
- **出处**: ByteDance Seed 团队，论文 "HybridFlow: A Flexible and Efficient RLHF Framework"
- **优势**:
  - 支持 GRPO、PPO、DAPO、PRIME 等主流算法
  - 3D-HybridEngine 架构，训练效率优于 TRL/OpenRLHF
  - 集成 vLLM/SGLang 做高效 rollout
  - 单 GPU 可用（FSDP backend）
  - 有完整的 code/math 任务示例

### 算法：GRPO
GRPO（Group Relative Policy Optimization）是 DeepSeek-R1 使用的核心算法，相比 PPO 不需要独立的 Critic 网络，显存占用更低，适合单卡训练。

```
对同一问题采样 G 个回答 → 执行测试 → 计算各回答 reward
→ 组内归一化得到 advantage → 用 PPO-clip 更新策略
```

### 基础模型：Qwen2.5-Coder-7B-Instruct
- HuggingFace: `Qwen/Qwen2.5-Coder-7B-Instruct`
- 7B 参数，代码能力强，中英文双语
- 48GB 显存跑 GRPO full fine-tune 约占 35-42GB

### 数据集
| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| MBPP | 374 题 | 每题 3 个测试用例，难度适中 | 主训练集 |
| APPS | 5000 题 | 分 intro/interview/competition 三档 | 进阶训练 |
| HumanEval | 164 题 | 业界标准评测 | 泛化性测试 |
| LiveCodeBench | 持续更新 | 防数据污染 | 最终评测参考 |

---

## 3. 创新方案（二选一）

### 方案 A：分层测试奖励（Partial Credit Reward）【推荐入门】

**问题**：传统 code RL 使用二元奖励（全过/全不过），稀疏信号导致学习效率低。

**方案**：用通过测试用例的比例作为连续奖励：

```python
def compute_reward(code: str, test_cases: list) -> float:
    passed = 0
    for test in test_cases:
        try:
            exec(code + "\n" + test, timeout=5)
            passed += 1
        except:
            pass
    
    base_reward = passed / len(test_cases)  # 0.0 ~ 1.0
    
    # 可选：全部通过给额外奖励
    if passed == len(test_cases):
        base_reward += 0.2
    
    return base_reward
```

**创新性**：大多数已有工作用 pass/fail 二元奖励，dense reward 在 code 任务上的系统验证较少。

---

### 方案 B：多轮调试 RL（Multi-turn Debugging RL）【进阶，工程价值更高】

**问题**：真实编程是迭代过程（写→报错→改），但现有 code RL 只训练单轮生成。

**方案**：构建多轮 MDP：

```
Turn 1: 问题描述 → 生成代码 v1
Turn 2: 代码 v1 + 错误信息 → 生成代码 v2
Turn 3: 代码 v2 + 错误信息 → 生成代码 v3（最多 N 轮）
Final reward: 最终版本通过测试用例的比例
```

**奖励设计**：
- 最终奖励：最后一轮代码通过率
- 可选中间奖励：每轮相比上轮通过率提升 → 正奖励；退步 → 负奖励

**veRL 支持**：veRL 有 `multiturn` 示例（`aime2024_multiturn_w_tool.py`），可以直接参考。

---

## 4. 实验设计

### Baseline 对比
| 实验组 | 描述 |
|--------|------|
| Baseline | Qwen2.5-Coder-7B-Instruct 原始模型，zero-shot |
| SFT | 在 MBPP 训练集上做监督微调 |
| **RL（本项目）** | SFT 基础上加 GRPO + 创新奖励 |

### 评测指标
- **pass@1**：生成 1 次，通过率
- **pass@10**：生成 10 次，至少 1 次通过率（体现模型上限）
- 评测集：MBPP test set (374)、HumanEval (164)

### 训练超参数参考
```yaml
# GRPO 关键参数
n_samples_per_prompt: 8      # 每题采样 8 个回答（G=8）
max_new_tokens: 512
kl_coef: 0.01
clip_ratio: 0.2
learning_rate: 1e-6
train_batch_size: 128
```

---

## 5. 项目结构规划

```
codellmRL/
├── project_plan.md          # 本文件
├── data/
│   ├── prepare_mbpp.py      # 下载和预处理 MBPP
│   └── prepare_apps.py      # 下载和预处理 APPS
├── rewards/
│   ├── executor.py          # 安全代码执行沙箱
│   └── reward_fn.py         # 奖励函数实现
├── configs/
│   ├── grpo_qwen7b.yaml     # veRL GRPO 训练配置
│   └── sft_baseline.yaml    # SFT baseline 配置
├── scripts/
│   ├── train_sft.sh
│   ├── train_grpo.sh
│   └── evaluate.sh
├── eval/
│   └── eval_pass_k.py       # pass@k 评测脚本
└── notebooks/
    └── analysis.ipynb       # 结果分析和可视化
```

---

## 6. 环境搭建步骤

### Step 1：安装 veRL

```bash
# 推荐用 conda 环境
conda create -n verl python=3.12
conda activate verl

# 安装 vLLM 和依赖（veRL 官方脚本）
git clone https://github.com/verl-project/verl.git
cd verl
bash scripts/install_vllm_sglang_mcore.sh

# 安装 veRL
pip install --no-deps -e .
```

### Step 2：验证环境

```bash
# 跑官方 GSM8K GRPO 示例（最小验证）
cd examples/grpo_trainer
bash run_qwen2.5_7b_fsdp.sh
```

### Step 3：适配 Code 任务

参考 veRL 的 `examples/` 目录，将 math reward 替换为 code execution reward。

---

## 7. 关键风险和注意事项

### 安全沙箱
执行模型生成的代码必须做沙箱隔离，防止恶意代码：
- 使用 `subprocess` + timeout 限制
- 限制可 import 的模块
- 推荐使用 `RestrictedPython` 或 Docker 容器隔离

### 显存管理（L20-48GB）
- GRPO full fine-tune 7B：约 35-42GB，需要关闭 gradient checkpointing 以外的冗余
- 若 OOM，切换 QLoRA（~18GB）或减小 `n_samples_per_prompt`
- 用 `verl` 的 FSDP backend，不需要多卡

### 数据污染
- HumanEval/MBPP 可能在预训练数据中出现
- 最终评测建议加入 LiveCodeBench（持续更新，防污染）

---

## 8. 时间规划（参考）

| 周次 | 任务 |
|------|------|
| 第 1 周 | 搭建环境，跑通 veRL 官方示例（GSM8K GRPO） |
| 第 2 周 | 准备 MBPP 数据，实现代码执行奖励函数 |
| 第 3 周 | 训练 SFT baseline，跑通 GRPO code 训练 |
| 第 4 周 | 实现创新奖励（方案A 或 B），对比实验 |
| 第 5-6 周 | 调参、补充实验、整理结果、写技术报告 |

---

## 9. 参考资料

- veRL 论文: [HybridFlow (arXiv:2409.19256)](https://arxiv.org/abs/2409.19256)
- GRPO 论文: [DeepSeekMath (arXiv:2402.03300)](https://arxiv.org/abs/2402.03300)
- DeepSeek-R1: [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- MBPP 数据集: [HuggingFace](https://huggingface.co/datasets/google-research-datasets/mbpp)
- APPS 数据集: [GitHub](https://github.com/hendrycks/apps)
- BigCodeBench: [GitHub](https://github.com/bigcode-project/bigcodebench)
- veRL 文档: https://verl.readthedocs.io
