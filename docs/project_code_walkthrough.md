# 项目代码走读：数据链路、reward、loss 与超参

本文档按答辩/复盘粒度梳理当前仓库。重点不是逐行翻译代码，而是回答：

- 数据如何构造，训练时如何被 veRL 消费。
- MBPP、APPS、LCB 三条数据线对实验结论的影响。
- SFT、GRPO、OPD 各自的训练入口、loss 路径和关键超参。
- reward/eval harness 如何把模型输出转成训练信号和 pass@k 指标。
- 当前仓库能确认什么，哪些由 `docs/verl_patch_inventory.md` 记录，哪些仍需训练日志补证。

读码过程中的随问随答记录在 `docs/code_reading_qa.md`。

## 0. 仓库边界

当前仓库保存的是：

- 数据准备脚本：`data/*.py`
- reward 函数：`rewards/*.py`
- vLLM 评测脚本：`eval/*.py`
- veRL 训练 launcher：`scripts/run_*.sh`
- SFT warmup 脚本：`scripts/run_sft_*.py`
- veRL patch 脚本和实验复盘：`scripts/apply_verl_opd_patch.sh`、`scripts/patch_arm_cd.py`、`docs/*.md`

当前仓库没有完整复制 veRL 的最终 patched 源码。因此：

- 训练 loop、GRPO advantage、PPO clip 等主体逻辑来自 veRL，通过 launcher 里的 Hydra override 控制。
- 本仓库的 `apply_verl_opd_patch.sh` 和 `patch_arm_cd.py` 记录了对 veRL 的修改意图和部分 patch 片段。
- 本次服务器对齐新增了 `docs/verl_patch_inventory.md`，记录远端 patched veRL 的配置项、teacher 数据流、loss/reward 注入点和已发现的不一致。

这条边界会影响几个问题的回答：例如“是否剔除 0 advantage group 并重采样”，本仓库没有自定义实现，只能看到 `algorithm.adv_estimator=grpo` 和 `trainer.balance_batch=True`。

## 1. 总体实验主线

主线由三段构成：

1. MBPP 快速消融：验证数据格式、函数名提示、binary/partial reward、SFT warmup、LR、KL/entropy、OPD/BC。
2. APPS 训练 + LCB 评测：把代码任务从函数级 MBPP 迁移到 stdin/stdout 竞赛题，重点比较不同 SFT 数据质量。
3. MBPP OPD 正确性验证：拆开 OPD 注入位置，比较 supervised loss、teacher-weighted PG、shaped reward、token_reward_direct。

最终实验文档给出的关键结论是：

- APPS+LCB 中，Teacher SFT + GRPO 最强，LCB `pass@1=13.1%`、`pass@5=25.7%`。
- MBPP 1.5B 中，LR=5e-6 + SFT warmup 的纯 GRPO 已到 `45.33%/55.76%`，OPD/BC 只到 `45.99%/55.96%`，差距不显著。
- OPD as loss 比较稳定；OPD as shaped reward 进入 GRPO advantage 后会改变 rollout 竞争标准，Arm E 崩溃。
- K=1 `token_reward_direct` 可行；K=16 top-K 在单卡上因 `gather.backward` OOM 不可行。

## 2. 数据链路

### 2.1 MBPP canonical：函数名已注入 parquet

入口：`data/prepare_mbpp.py`

数据源：

- `datasets.load_dataset("google-research-datasets/mbpp", "full")`
- 生成 train/test parquet，默认输出到 `data/mbpp_v2/`。
- 从 `test_list` 的 assert 语句提取被测函数名，并注入 prompt。

输出字段：

| 字段 | 内容 | 训练用途 |
|---|---|---|
| `prompt` | chat format，`[{"role": "user", "content": ...}]` | veRL 用 tokenizer chat template 编码 |
| `data_source` | `"mbpp"` | reward manager 路由 |
| `reward_model.ground_truth` | JSON 字符串，内容为 assert 列表 | `rewards/mbpp_reward.py` 执行 |
| `extra_info.task_id` | MBPP task id | 记录/评测追踪 |

本地历史样例 parquet：

- `data/mbpp_train.parquet`：374 条
- `data/mbpp_test.parquet`：500 条

注意：重点实验实际使用的是 `mbpp_v2/mbpp_train.parquet` 和 `mbpp_v2/mbpp_test.parquet`。

### 2.2 MBPP v1/v2 历史关系

旧入口：`data/prepare_mbpp_v2.py`

核心问题：

- v1 prompt 只描述功能，没有告诉模型函数名。
- ground truth assert 要求精确函数名。
- 模型即使思路正确，函数名猜错也会 reward=0，导致 GRPO group 内大量全 0。

修复方式：

1. 从 MBPP `test_list` 的第一条 assert 用正则提取函数名。
2. 在 prompt 中加入：

```text
Your function must be named `<func_name>`.
```

现在 `prepare_mbpp.py` 已直接生成修复后的 v2 数据；`prepare_mbpp_v2.py` 仅作为旧命令兼容 wrapper。早期 v1 格式可通过 `prepare_mbpp.py --legacy-no-function-name` 复现。

这是本项目最重要的数据影响之一：早期训练无效不是先由算法问题导致，而是数据/评测不对齐导致 reward 极度稀疏。

### 2.3 MBPP teacher 数据

入口：`data/precompute_teacher_logprobs.py`

流程：

1. 读取 MBPP v2 train parquet。
2. 用 teacher 模型 greedy 生成一条 response。
3. 计算 teacher 在这条 response 上每个 token 的 log-prob。
4. 先用 `rewards/mbpp_reward.py::extract_code` 去掉 `<think>` 和 Markdown code fence，再用 MBPP assert 测试 teacher response 是否全通过。
5. 写回新增列：

| 列 | 含义 |
|---|---|
| `teacher_response_ids` | teacher 生成 response 的 token id 列表 |
| `teacher_logprobs` | teacher 对这些 token 的 log-prob |
| `teacher_pass` | teacher response 是否通过全部测试 |

这条线主要支持 MBPP teacher-SFT warmup 和 OPD/CL 实验。

### 2.4 MBPP OPD + CL 数据

入口：`data/prepare_mbpp_opd_cl.py`

输入：

- `mbpp_train_with_teacher.parquet`

新增：

- `cl_weight`
- `has_teacher_solution`

默认策略：

- `teacher_pass=True`：保留，`cl_weight=2.0`
- `teacher_pass=False`：保留但降权，`cl_weight=0.5`
- `teacher_pass=None`：`cl_weight=1.0`
- 如果传 `--filter_hard`，会过滤 `teacher_pass=False`

注意：这个脚本生成的是 CL/OPD 数据版本，但当前重点实验的大部分 launcher 仍直接使用 `mbpp_train.parquet`，OPD 信号更多通过在线 teacher 或 patch 注入，而不是直接让 reward 函数读取 teacher 列。

### 2.5 APPS RL/SFT 数据

入口：`data/prepare_apps.py`

默认只保留：

- `difficulty=["interview"]`

原因：

- introductory 太简单，和 LCB 难度不匹配。
- competition 太难，7B 也容易失败，GRPO reward 过稀疏。
- interview 更接近 LCB 的 medium/hard 竞赛题。

输出两个文件：

| 文件 | 用途 | 关键字段 |
|---|---|---|
| `apps_rl_interview.parquet` | GRPO/OPD 训练 | `prompt/data_source/reward_model/extra_info` |
| `apps_sft_interview.parquet` | human SFT warmup | `prompt/response/question_id/difficulty` |

RL 数据：

- `data_source="apps"`
- `ground_truth` 是最多前 5 个 stdin/stdout tests 的 JSON。
- prompt 是 system+user chat message，要求完整 Python 3 程序、stdin/stdout、代码块输出。

SFT 数据：

- 从 APPS 自带 `solutions` 取 Python 解法。
- 过滤 Python 2 `print "..."` 风格、过短解法。
- 每题最多取 3 个解法。
- response 包装成 Markdown Python 代码块。

脚本注释中 APPS human SFT warmup 的数据量是 `5611` 条；APPS GRPO 脚本注释中 RL 训练题量是 `1692` 题，4 epoch 约 `6800` 样本。

### 2.6 LCB 评测/验证数据

入口：`data/prepare_lcb.py`

数据源：

- `livecodebench/code_generation_lite`
- 默认 split `v5`

处理：

1. 清理 HTML。
2. 标注 `execution_type`：
   - Codeforces/AtCoder：`stdin_stdout`
   - LeetCode：`leetcode_fn`
3. 只把 stdin/stdout 子集作为主评测/validation。
4. 额外保存 veRL 格式的 `lcb_v5_verl.parquet`，供训练时 `data.val_files` 使用。

LCB 不参与训练，是防污染的 validation/evaluation 集。

## 3. Reward 与 eval harness

### 3.1 MBPP reward

入口：`rewards/mbpp_reward.py`

流程：

1. 从模型输出中去掉 `<think>...</think>`。
2. 如果有 Markdown code block，提取其中 Python 代码。
3. 对每条 assert 启动 `multiprocessing.Process` 子进程。
4. 子进程设置 512MB 虚拟内存限制。
5. `exec(code + "\n" + test_case)`，assert 失败/异常/超时都记为失败。

reward 模式由环境变量控制：

| `REWARD_MODE` | 规则 |
|---|---|
| `binary` | 全部测试通过为 1，否则 0 |
| `partial` | MBPP 3 条测试映射为 `0/3=0.0, 1/3=0.2, 2/3=0.6, 3/3=1.0` |

### 3.2 OPD 版 MBPP reward

入口：`rewards/mbpp_reward_opd.py`

它没有计算 distillation loss，只保留 execution reward。

换句话说：OPD 的 loss/reward 注入是在 veRL actor update 或 advantage 计算周围做的，不在这个 reward 函数里完成。

### 3.3 APPS/LCB reward

入口：`rewards/lcb_reward.py`

区别于 MBPP：

- MBPP 是函数级 assert。
- APPS/LCB 是完整程序，stdin/stdout。

流程：

1. 提取代码块。
2. 把代码写入临时 `.py` 文件。
3. `subprocess.run(["python3", tmp_file], input=test_input, timeout=10)`。
4. 比对 stdout。

输出比对策略：

1. `strip()` 后精确比对。
2. 逐行去行尾空格比对。
3. 浮点数近似比对，容差 `1e-6`。

partial reward：

- `0/n=0.0`
- `n/n=1.0`
- 中间线性映射到 `[0.2, 0.6]`

### 3.4 pass@k eval

入口：

- `eval/eval_mbpp.py`
- `eval/eval_lcb.py`

共同点：

- 使用 vLLM 批量生成。
- 使用 tokenizer `apply_chat_template`。
- 每题采样 `n_samples` 条。
- 每条样本独立执行测试。
- `c` 是通过全部测试的样本数。
- pass@k 使用无偏估计：

```python
pass@k = 1 - C(n-c, k) / C(n, k)
```

默认 eval 设置：

| 数据集 | `n_samples` | `temperature` | `max_tokens` |
|---|---:|---:|---:|
| MBPP | 20 | 0.8 | 1024 |
| LCB | 5 | 0.8 | 2048 |

LCB 用 n=5 是因为题目更长、生成更慢。

## 4. SFT 训练

### 4.1 APPS human SFT

入口：

- `scripts/run_apps_sft_warmup.sh`
- `scripts/run_sft_apps_warmup.py`

用途：

- 用 APPS human solutions 对 `Qwen2.5-Coder-1.5B-Instruct` 做 warmup。
- 作为 APPS GRPO 和 APPS OPD 的共同起点。

关键参数：

| 参数 | 值 |
|---|---:|
| epoch | 2 |
| per-device batch | 2 |
| grad accumulation | 8 |
| effective batch | 16 |
| learning rate | `2e-5` |
| max seq length | 3072 |
| warmup steps | 30 |
| dtype | bf16 |
| gradient checkpointing | true |

loss：

- 使用 HuggingFace `Trainer`。
- prompt token label 全部设为 `-100`。
- 只在 response token 上计算 cross entropy。

是否 LoRA：

- 没有 LoRA/PEFT/QLoRA 相关代码。
- 脚本直接加载 `AutoModelForCausalLM` 并交给 `Trainer`。
- 因此按当前代码是全参 SFT。

### 4.2 MBPP teacher SFT

入口：

- `scripts/run_sft_teacher_warmup.py`

用途：

- 在 `teacher_pass=True` 的样本上训练 student。
- response 来自 `teacher_response_ids`。

关键默认参数：

| 参数 | 默认值 |
|---|---:|
| epoch | 2 |
| per-device batch | 4 |
| grad accumulation | 4 |
| effective batch | 16 |
| learning rate | `2e-5` |
| max seq length | 2048 |
| warmup steps | 20 |

0.5B OPD early 流程中，launcher 注释使用过：

- epoch 3
- per-device batch 8
- grad accumulation 4
- effective batch 32

是否拒绝采样：

- `run_sft_teacher_warmup.py` 会过滤 `teacher_pass=True`。
- 如果 teacher 数据来自 `precompute_teacher_logprobs.py`，每题只有一条 greedy teacher response；这更像“teacher 正确样本过滤”，不是 n 条采样中的 rejection sampling。
- 如果 teacher 数据来自 `prepare_mbpp_1_5b_teacher_sft.py`，每题采样 `n_samples=5`，选第一条通过测试的 response，这就是更明确的 rejection/filtering 过程。

### 4.3 APPS teacher SFT 的证据边界

`docs/experiment_report.md` 记录了 APPS+LCB 主实验中 Teacher SFT 明显优于 Human SFT。

本次服务器对齐新增了 `scripts/prepare_apps_teacher_sft.py`，因此可以确认：

- 可以确认 human SFT 的构造规则和训练脚本。
- 可以确认 APPS teacher-SFT 用 7B teacher 对 `apps_rl_interview.parquet` 每题采样 `n_samples=4`，执行测试，保留第一条通过所有测试的 response。
- 可以确认 Teacher SFT warmup 入口是 `scripts/run_apps_teacher_sft_warmup.sh`，默认 3 epoch。
- 可以确认 Teacher SFT + GRPO 入口是 `scripts/run_apps_teacher_grpo.sh`，默认 `rollout.n=8`、4 epoch、LR `5e-6`。

答辩时建议说：

> MBPP teacher-SFT 的过滤规则在代码里可确认：teacher_pass=True 或 n=5 中选 passing response。APPS teacher-SFT 现在也可由 `prepare_apps_teacher_sft.py` 确认：7B teacher 每题采样 4 条，按测试执行结果筛选第一条 passing response，最终只把 `teacher_pass=True` 的样本写入 SFT parquet。

## 5. GRPO 训练入口与超参

所有 GRPO 训练基本都通过：

```bash
python -m verl.trainer.main_ppo
```

核心共有设置：

- `algorithm.adv_estimator=grpo`
- `algorithm.use_kl_in_reward=False`
- `data.filter_overlong_prompts=True`
- `data.truncation=error`
- `actor_rollout_ref.rollout.name=vllm`
- `trainer.balance_batch=True`
- `trainer.n_gpus_per_node=1`
- `trainer.nnodes=1`

GRPO advantage：

- 同一 prompt 采样 N 条 response。
- 对同组 reward 做相对归一化。
- 如果一组全 0 或全 1，组内标准差为 0，advantage 约等于 0，这组基本无梯度。

当前仓库没有自定义“剔除 0 advantage group 后重采样”的逻辑。只能确认：

- 训练依赖 veRL 的 `grpo` estimator。
- launcher 设置了 `trainer.balance_batch=True`。
- 没有在本仓库 reward 或 launcher 里显式写 resample/过滤 zero-advantage group。

### 5.1 MBPP 1.7B 早期消融

代表脚本：

- `scripts/run_mbpp_v2_binary_1_7b.sh`
- `scripts/run_mbpp_v2_partial_1_7b.sh`
- `scripts/run_mbpp_v2_partial_kl_1_7b.sh`
- `scripts/run_mbpp_v2_partial_entropy_1_7b.sh`

共同参数：

| 参数 | 值 |
|---|---:|
| model | `Qwen3-1.7B` |
| train file | `mbpp_v2/mbpp_train.parquet` |
| val file | `mbpp_v2/mbpp_test.parquet` |
| train batch | 64 |
| PPO mini batch | 32 |
| PPO micro batch | 4 |
| max prompt | 512 |
| max response | 512 |
| LR | `1e-6` |
| rollout n | 5 |
| total epochs | 16 |

差异：

| 实验 | reward | 额外项 |
|---|---|---|
| binary | binary | 无 |
| partial | partial | 无 |
| partial+KL | partial | `actor.use_kl_loss=True`, `kl_loss_coef=0.001` |
| partial+entropy | partial | `entropy_coeff=0.01` |

评测结果记录：

- binary final: `35.30%/40.44%`
- partial final: `35.50%/42.27%`
- KL final: `35.24%/40.83%`

### 5.2 MBPP 1.5B 关键消融

代表脚本：

- `scripts/run_mbpp_v2_coder_1_5b_grpo_lr5e6.sh`
- `scripts/run_mbpp_v2_coder_1_5b_opd_bc_sweep.sh`

纯 GRPO 起点：

- `models/coder_1_5b_sft_warmup/final`

关键参数：

| 参数 | 值 |
|---|---:|
| train batch | 64 |
| PPO mini batch | 64 |
| PPO micro batch | 8 |
| LR | `5e-6` |
| rollout n | 8 |
| total epochs | 16 |
| reward | partial |

结果解读：

- 纯 GRPO：`45.33%/55.76%`
- OPD/BC：`45.99%/55.96%`
- 因为差距仅约 `+0.66pp/+0.20pp`，不能说 OPD 显著提升；更强结论是 LR 和 SFT warmup 本身才是主增益。

### 5.3 APPS 训练 + LCB validation

代表脚本：

- `scripts/run_apps_sft_warmup.sh`
- `scripts/run_apps_grpo_lr5e6.sh`
- `scripts/run_apps_opd_bc005.sh`

APPS GRPO 参数：

| 参数 | 值 |
|---|---:|
| model | `coder_1_5b_apps_sft_warmup/final` |
| train file | `apps_rl_interview.parquet` |
| val file | `lcb_v5_verl.parquet` |
| train batch | 32 |
| PPO mini batch | 32 |
| PPO micro batch | 4 |
| log-prob micro batch | 2 |
| max prompt | 1024 |
| max response | 2048 |
| LR | `5e-6` |
| rollout n | 8 |
| total epochs | 4 |
| reward | partial |

APPS OPD 与 GRPO 的结构差异：

- 增加 `teacher_model_path`
- 增加 `bc_loss_coef=0.05`
- 其他训练数据、起点、LR、rollout n 尽量保持一致。

实验报告结论：

- Human SFT warmup 提升 greedy 但压低 sampling pass@k，说明分布收窄。
- Teacher SFT warmup 更好，叠加 GRPO 最强。

## 6. OPD/loss 实现

### 6.1 标准 GRPO loss

GRPO 主体由 veRL 实现，本仓库通过参数开启：

```bash
algorithm.adv_estimator=grpo
```

概念上：

```text
same prompt -> N responses -> rewards -> group normalized advantage
loss = -mean(min(ratio * A, clip(ratio) * A))
```

其中 `ratio = pi_current / pi_old`，通过 PPO clip 控制更新幅度。

### 6.2 OPD as loss：Arm B

设计意图：

```text
L_total = L_GRPO + beta * L_OPD
```

OPD 作为 actor update 阶段的额外正则，不改变 GRPO 的 rollout 竞争标准。

正确梯度必须包含 live student log-prob：

```text
L_OPD = mean(log pi_student_live(token) - log pi_teacher(token).detach())
```

重要坑：

- 如果写成 `-mean(teacher_log_probs)`，teacher log-prob 与 student 参数无关，梯度为 0。
- `docs/experiment_report.md` 明确记录过历史 OPD 梯度为零问题。
- 当前 `scripts/apply_verl_opd_patch.sh` 中保留的早期 patch 片段仍有“只对 teacher_log_probs 做 agg_loss”的风险；最终正确实现应以 `docs/verl_patch_inventory.md` 和 `docs/opd_implementation_analysis.md` 的修正说明为准。

### 6.3 OPD as teacher-weighted PG：Arm C

入口：

- `scripts/patch_arm_cd.py`
- `scripts/run_mbpp_0.5b_opd_k1_reward.sh`

激活参数：

- `bc_loss_coef=0`
- `bc_reward_coef=0.05`

loss 形式：

```text
L_C = -mean(exp(log pi_teacher(token)).detach() * log pi_student_live(token))
```

区别：

- Arm B 等权对齐每个 sampled token。
- Arm C 用 teacher probability 加权，高置信 token 更新更大，低置信 token 更新更小。

### 6.4 OPD as shaped reward：Arm E

入口：

- `scripts/run_mbpp_0.5b_opd_shaped_reward.sh`
- `scripts/run_mbpp_0.5b_opd_shaped_reward_b001.sh`

激活参数：

- `bc_loss_coef=0`
- `bc_reward_coef=0`
- `bc_shaped_reward=True`
- 可选 `bc_shaped_reward_beta=0.001`

机制：

```text
token_level_rewards = task_reward + beta * opd_token_reward
advantage = GRPO_normalize(token_level_rewards)
```

与 Arm B/C 的本质区别：

- Arm B/C：OPD 在 advantage 之后加到 loss，不改变哪个 rollout 被认为更好。
- Arm E：OPD 在 advantage 之前混入 reward，改变 rollout 竞争标准。

实验结果：

- Arm E 系列崩溃，response 变长，val acc 归零。
- 归因是 teacher token preference 与 task objective 不完全一致时，GRPO 会强化更长/teacher 更偏好的 rollout，形成正反馈。

### 6.5 token_reward_direct：Arm D

文档入口：

- `docs/opd_implementation_analysis.md`

思想：

```text
rm_score_t = log pi_teacher(sampled_token_t) - log pi_student_old(sampled_token_t)
```

Arm D_direct：

```text
advantage[t] = rm_score[t]
```

Arm D_plus_grpo：

```text
advantage[t] = rm_score[t] + GRPO_task_advantage
```

结果：

- D_direct 可稳定训练。
- D_plus_grpo 最好，说明 token-level teacher 信号仍需要 task reward 约束方向。
- K=16 top-K 版本单卡 OOM，因为 PPO 阶段需要对多个 top-K `log pi_new` 反传，`gather.backward` 显存过大。

当前 caveat：

- `scripts/run_mbpp_0.5b_arm_d_k1.sh` 使用 `+actor_rollout_ref.actor.arm_d_k1_coef=0.05`。
- `docs/verl_patch_inventory.md` 已确认远端 veRL 注册了 `token_reward_direct` / `token_reward_direct_plus_grpo`，并记录了 shaped reward 注入点。
- 但本次拉到的 `verl/workers/utils/losses.py` 没有找到 active `arm_d_k1_coef` loss 分支，因此 K=1 脚本需要在运行前重新验证 patch 是否完整生效。

## 7. 重点实验对照表

### 7.1 MBPP 数据/奖励消融

| 实验 | 数据 | 模型 | 起点 | reward | LR | rollout n | 结论 |
|---|---|---|---|---|---:|---:|---|
| v1 early | MBPP v1 | Qwen3-1.7B | instruct | binary/partial | `1e-6` | 5 | 函数名缺失导致 reward 稀疏 |
| v2 binary | MBPP v2 | Qwen3-1.7B | instruct | binary | `1e-6` | 5 | RL 可提升 |
| v2 partial | MBPP v2 | Qwen3-1.7B | instruct | partial | `1e-6` | 5 | pass@5 略优 |
| v2 KL | MBPP v2 | Qwen3-1.7B | instruct | partial | `1e-6` | 5 | 延缓 collapse 但最终不最优 |
| 1.5B GRPO | MBPP v2 | Qwen2.5-Coder-1.5B | SFT warmup | partial | `5e-6` | 8 | 纯 GRPO 主结果 |
| 1.5B OPD/BC | MBPP v2 | Qwen2.5-Coder-1.5B | SFT warmup | partial | `5e-6` | 8 | 与纯 GRPO 差距不显著 |

### 7.2 APPS SFT 数据对比

| 阶段 | 数据 | 结论 |
|---|---|---|
| Human SFT warmup | APPS human `solutions`，每题最多 3 个 | greedy 可提升，但 sampling pass@k 分布收窄 |
| Human SFT + GRPO | APPS RL tests | GRPO 部分修复分布收窄 |
| Human SFT + OPD | APPS RL tests + 7B teacher BC | 与 GRPO 无显著差异 |
| Teacher SFT warmup | 实验报告记录，生成脚本不在 tracked 仓库 | 单独已强于 Human SFT+GRPO pass@1 |
| Teacher SFT + GRPO | APPS/LCB 主结果 | 最强，LCB `13.1%/25.7%` |

### 7.3 MBPP OPD 验证

| Arm | 起点 | 机制 | 注入位置 | 结果解读 |
|---|---|---|---|---|
| A | 0.5B SFT warmup | GRPO only | task advantage | 对照 |
| B | 0.5B SFT warmup | supervised KL | actor update | 稳定，但 pass@5 收窄 |
| C | 0.5B SFT warmup | teacher-weighted PG | actor update | pass@5 更接近对照 |
| E | 0.5B SFT warmup | shaped reward | advantage 之前 | 崩溃 |
| D_direct | 1.5B-aligned SFT | token_reward_direct | token advantage | 可行但无 task 约束 |
| D_plus_grpo | 1.5B-aligned SFT | rm_score + task adv | token advantage | Arm D 最优 |
| D_topK | 1.5B-aligned SFT | K=16 top-K reward | token advantage | 单卡 OOM |

## 8. 常见问答

### Q1：数据怎么生产，训练时怎么处理？

MBPP：

- `prepare_mbpp.py` 直接从 MBPP 原始数据生成函数名已注入的 veRL parquet。
- `prepare_mbpp_v2.py` 只保留旧命令兼容入口。
- 训练时 veRL 读取 `prompt/data_source/reward_model/extra_info`。
- reward 函数从 `ground_truth` 取 assert 列表执行。

APPS：

- `prepare_apps.py` 读取 APPS train jsonl。
- 默认只取 `interview` 难度。
- RL 数据写成 veRL 格式，ground truth 是最多 5 个 stdin/stdout tests。
- SFT 数据写成 prompt/response 文本对，每题最多 3 个 human solutions。

LCB：

- `prepare_lcb.py` 处理 LCB v5。
- 主评测保留 stdin/stdout 子集。
- 生成普通 eval parquet 和 veRL-format val parquet。

### Q2：reward 怎么改？

MBPP reward 从 binary 切到 partial 主要通过：

```bash
export REWARD_MODE=partial
```

代码在 `rewards/mbpp_reward.py`：

- binary：全 assert 通过为 1。
- partial：`0, 0.2, 0.6, 1.0`。

APPS/LCB reward 在 `rewards/lcb_reward.py`：

- binary：所有 stdin/stdout tests 全通过为 1。
- partial：中间通过数线性映射到 `[0.2, 0.6]`。

OPD reward/loss 没有放在 reward 函数里，而是通过 patched veRL actor update 或 advantage 计算路径注入。

### Q3：trainer/loss 怎么实现？

分三层：

1. SFT：HuggingFace `Trainer`，response-only cross entropy，prompt label mask 为 `-100`。
2. GRPO：veRL `main_ppo`，`algorithm.adv_estimator=grpo`，PPO clip loss。
3. OPD：patch veRL，在 actor update 或 advantage 前后加 teacher signal。

仓库可确认的 OPD 方式：

- `bc_loss_coef`：OPD as supervised/BC loss。
- `bc_reward_coef`：teacher-weighted PG loss。
- `bc_shaped_reward`：teacher signal 进入 reward，再做 GRPO normalize。
- `token_reward_direct` / `plus_grpo`：`docs/verl_patch_inventory.md` 记录了远端 veRL 的实现边界，当前仓库不 vendoring 完整 veRL 文件。

### Q4：SFT 训练了多少轮，样本量多少，LoRA 还是全参？

APPS human SFT：

- 2 epoch。
- 脚本注释：5611 samples。
- effective batch 16。
- 全参 SFT，无 LoRA/PEFT。

MBPP teacher SFT：

- 默认 2 epoch，部分 0.5B 流程用 3 epoch。
- 样本来自 `teacher_pass=True` 过滤后的 teacher 数据；实际数量取决于 teacher pass rate，需要读对应 parquet 或训练日志确认。
- 全参 SFT，无 LoRA/PEFT。

### Q5：训练过程中采样 n 是多少，advantage 怎么算？

MBPP 1.7B early：

- `rollout.n=5`

MBPP 1.5B key / APPS / 0.5B OPD：

- `rollout.n=8`

评测：

- MBPP eval `n=20`
- LCB eval `n=5`

GRPO advantage：

- 同一 prompt 的 N 条回答按 reward 做组内归一化。
- 高于组均值为正 advantage，低于组均值为负 advantage。
- 全组 reward 相同，advantage 约为 0，训练信号消失。

### Q6：7B teacher SFT 是否做拒绝采样？

需要分数据线说：

- MBPP `precompute_teacher_logprobs.py`：7B greedy 每题 1 条，按正式 reward 的 `extract_code` 口径记录 `teacher_pass`；后续 SFT 过滤 `teacher_pass=True`。这是正确样本过滤，但不是多样本 rejection sampling。
- MBPP `prepare_mbpp_1_5b_teacher_sft.py`：每题采样 `n=5`，选择通过测试的 response，属于明确的 passing-sample selection。
- APPS teacher SFT：`prepare_apps_teacher_sft.py` 每题采样 `n=4`，执行测试并保留第一条 passing response，属于明确的 rejection/filtering。

### Q7：GRPO 生成 query 的分布是否合理？

这里的 query/prompt 分布由训练 parquet 决定：

- MBPP v2：函数级任务，374 train，函数名注入后 reward 对齐。
- APPS：只取 interview 难度，避免太简单/太难。
- LCB：只做 val/eval，不训练。

合理性的主要证据不是“生成 query”，而是：

- prompt 与 reward/eval harness 对齐。
- rollout n 足够产生组内差异。
- batch size 足够降低全零批次概率。
- pass@5 没有明显塌缩时，说明采样分布仍保留一定多样性。

### Q8：是否对 0 advantage 样本剔除、重采样？

当前 tracked 仓库没有显式实现。

能确认的是：

- 所有重点 GRPO launcher 使用 `algorithm.adv_estimator=grpo`。
- 多数 launcher 使用 `trainer.balance_batch=True`。
- 没有看到自定义 zero-advantage group filtering 或 resampling 代码。

本次对齐后的回答是：导入的重点 launcher 没有开启 zero-advantage group filtering 或 resampling；`docs/verl_patch_inventory.md` 也只看到 veRL 的 `filter_groups` 配置能力，没有看到这些实验脚本启用它。训练日志仍可用于确认实际运行时是否临时 override。

### Q9：数据对实验影响最大的点是什么？

按影响排序：

1. MBPP v1 缺函数名导致 reward 稀疏，v2 修复后训练才有意义。
2. SFT warmup 数据质量决定 RL 上限，APPS human SFT 可能压缩 sampling 分布，teacher SFT 更好。
3. APPS difficulty 选择决定 reward 稀疏度，interview 是折中。
4. reward mode 决定 group 内是否容易出现差异，partial 对 pass@5 和早期信号略有帮助。
5. teacher/student 初始分布差距决定 OPD shaped reward 是否会干扰 GRPO rollout 选择。

## 9. 后续需要补证的清单

为了把答辩材料从“仓库级复盘”升级为“完整训练记录级复盘”，还需要补：

1. 实际 APPS/MBPP teacher parquet 行数和 `teacher_pass` 分布。
2. 每个 final checkpoint 对应的训练日志，确认实际 step 数、是否 resume、best checkpoint 选择指标。
3. veRL 当前版本中 `trainer.balance_batch=True` 的精确定义，确认它是否仅平衡 token/batch，还是包含 zero-advantage 处理。
4. `arm_d_k1_coef` 对应 loss 分支是否在最终运行环境完整生效。
