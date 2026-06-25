# 代码走读 Q&A 笔记

本文档记录读码时反复会查的问题。主线解释看 `docs/project_code_walkthrough.md`，这里保持短、可操作。

## Q1：`load_dataset(dataset_name, dataset_config)` 在做什么？

位置：`data/prepare_mbpp.py`

```python
ds = load_dataset(dataset_name, dataset_config)
```

当前默认值：

```python
dataset_name = "google-research-datasets/mbpp"
dataset_config = "full"
```

含义：

- `dataset_name` 是 Hugging Face Hub 上的数据集 ID，不是本地路径。
- `dataset_config` 是数据集配置名，对应 `load_dataset(..., name=...)`。
- 不传 `split` 时，返回 `DatasetDict`，可用 `ds["train"]`、`ds["test"]` 等访问。

MBPP `full` 的 split：

| split | 行数 |
|---|---:|
| `train` | 374 |
| `test` | 500 |
| `validation` | 90 |
| `prompt` | 10 |

`full` 原始字段：

```text
task_id
text
code
test_list
test_setup_code
challenge_test_list
```

## Q2：config 除了 `full` 还能用什么？

MBPP 当前有两个 config：

| config | 行数 | 字段特点 | 当前脚本兼容性 |
|---|---:|---|---|
| `full` | 974 | `text`, `test_list`, `test_setup_code`, `challenge_test_list` | 直接兼容 |
| `sanitized` | 427 | `prompt`, `test_imports`, `test_list` | 当前脚本不直接兼容 |

查询方式：

```python
from datasets import get_dataset_config_names

print(get_dataset_config_names("google-research-datasets/mbpp"))
```

注意：`prepare_mbpp.py` 现在读取 `item["text"]`，而 `sanitized` 的题面字段是 `prompt`。如果要支持 `sanitized`，至少需要改题面读取逻辑，并判断 `test_imports` 是否要进入 reward 执行环境。

## Q3：原始数据从哪里拿？服务器上为什么没有 `google-research-datasets/mbpp` 目录？

`google-research-datasets/mbpp` 是 Hugging Face 数据集 ID，不是服务器目录。

本项目建议优先用 Hugging Face `full` 数据复现：

```python
from datasets import load_dataset

ds = load_dataset("google-research-datasets/mbpp", "full")
print(ds["train"][0])
```

如果要追溯 Google Research 原始发布源：

```text
https://github.com/google-research/google-research/tree/master/mbpp
```

里面主要是：

```text
mbpp.jsonl
sanitized-mbpp.json
```

服务器训练时不读 Hugging Face ID，而是读已经生成好的 parquet：

| 目录 | 内容 |
|---|---|
| `$HOME/data/mbpp_v2/` | 当前 MBPP v2 主实验数据，prompt 已注入函数名 |
| `$HOME/data/mbpp/` | 旧版 MBPP v1 数据，prompt 未注入函数名 |
| `$HOME/data/apps/` | APPS 数据 |
| `$HOME/data/lcb/` | LiveCodeBench 数据 |
| `$HOME/codellmRL/data/` | 数据处理脚本，不是主要数据落盘目录 |

MBPP 主线脚本通常读取：

```bash
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/mbpp_v2/mbpp_train.parquet}
TEST_FILE=${TEST_FILE:-$HOME/data/mbpp_v2/mbpp_test.parquet}
```

## Q4：如何手动复现 MBPP 数据处理？

本地已经导出过一份原始数据：

```text
data/raw/mbpp_hf_full/train.jsonl
data/raw/mbpp_hf_full/test.jsonl
data/raw/mbpp_hf_full/validation.jsonl
data/raw/mbpp_hf_full/prompt.jsonl
```

本地也已经生成过一份复现 parquet：

```text
data/mbpp_v2_reproduce/mbpp_train.parquet
data/mbpp_v2_reproduce/mbpp_test.parquet
```

重新导出原始数据：

```bash
HF_HOME=/tmp/codellmrl_hf \
HF_DATASETS_CACHE=/tmp/codellmrl_hf/datasets \
python - <<'PY'
from datasets import load_dataset
from pathlib import Path
import json

ds = load_dataset("google-research-datasets/mbpp", "full")
out_dir = Path("data/raw/mbpp_hf_full")
out_dir.mkdir(parents=True, exist_ok=True)

for split_name, split in ds.items():
    out_path = out_dir / f"{split_name}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for item in split:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(split_name, len(split), out_path)
PY
```

重新生成 veRL parquet：

```bash
HF_HOME=/tmp/codellmrl_hf \
HF_DATASETS_CACHE=/tmp/codellmrl_hf/datasets \
python data/prepare_mbpp.py \
  --output_dir data/mbpp_v2_reproduce \
  --preview_rows 2
```

转换逻辑：

1. 读取 MBPP `full` 原始样本。
2. 从 `test_list` 的 assert 里提取函数名。
3. 把函数名写进 prompt：`Your function must be named ...`。
4. 写出 veRL 训练四列：`prompt`, `data_source`, `reward_model`, `extra_info`。

本次复现观察：

```text
train: 374 rows, function names injected: 372/374
test: 500 rows, function names injected: 500/500
```

train 中有 2 条没注入函数名，因为 assert 写法是 `assert (func(...)) == ...`，当前正则只匹配 `assert func(...)`。

## Q5：如何读取 train/test parquet？

最小读取方式：

```python
import pandas as pd

train_df = pd.read_parquet("data/mbpp_v2_reproduce/mbpp_train.parquet")
test_df = pd.read_parquet("data/mbpp_v2_reproduce/mbpp_test.parquet")

print(train_df.shape)
print(test_df.shape)
print(train_df.columns)
```

当前 MBPP v2 parquet 结构：

| 文件 | 行数 | 列 |
|---|---:|---|
| `mbpp_train.parquet` | 374 | `prompt`, `data_source`, `reward_model`, `extra_info` |
| `mbpp_test.parquet` | 500 | `prompt`, `data_source`, `reward_model`, `extra_info` |

更推荐这样拆开看一条样本：

```python
import json
import pandas as pd

df = pd.read_parquet("data/mbpp_v2_reproduce/mbpp_train.parquet")
row = df.iloc[0].to_dict()

print("task_id:", row["extra_info"]["task_id"])
print("data_source:", row["data_source"])
print("prompt:")
print(row["prompt"][0]["content"])
print("tests:")
print(json.loads(row["reward_model"]["ground_truth"]))
```

要按 `task_id` 查某一题：

```python
import pandas as pd

df = pd.read_parquet("data/mbpp_v2_reproduce/mbpp_train.parquet")
row = df[df["extra_info"].map(lambda x: x["task_id"] == 601)].iloc[0]

print(row["prompt"][0]["content"])
```

注意：`prompt` 是 chat 格式嵌套字段，通常长这样：

```python
[{"role": "user", "content": "..."}]
```

`reward_model["ground_truth"]` 不是 Python list，而是 JSON 字符串，因此查看测试用例时需要：

```python
json.loads(row["reward_model"]["ground_truth"])
```

## Q6：MBPP 原始数据里的 `code` 是标准答案吗？能像 APPS human solutions 一样做 SFT 吗？

可以把 MBPP 原始字段 `code` 理解为参考解，但它不在当前 `mbpp_v2` 训练 parquet 里。

本地快速验证：

```text
MBPP full/train: 374/374 reference code passed test_list
MBPP full/test:  498/500 reference code passed test_list
```

所以它基本可以作为 SFT target，但和 APPS human SFT 有几个区别：

| 数据 | 解法字段 | 当前项目是否已构造成 SFT parquet | 特点 |
|---|---|---|---|
| MBPP | `code` | 没有 | 每题通常 1 个参考解，函数级任务 |
| APPS | `solutions` | 有，`apps_sft_*.parquet` | 每题可取多个 human solutions，stdin/stdout 程序 |

当前 `prepare_mbpp.py` 写出的 veRL parquet 只有：

```text
prompt
data_source
reward_model
extra_info
```

没有保留原始 `code`。因此不能直接拿 `data/mbpp_v2_reproduce/mbpp_train.parquet` 做 reference-code SFT；要从 `load_dataset(...)` 或 `data/raw/mbpp_hf_full/train.jsonl` 重新构造：

```python
import json
import pandas as pd

rows = []
with open("data/raw/mbpp_hf_full/train.jsonl", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        rows.append({
            "task_id": item["task_id"],
            "prompt": json.dumps([
                {
                    "role": "user",
                    "content": (
                        "You are an expert Python programmer. "
                        "Solve the following problem by writing a Python function.\n\n"
                        f"Problem: {item['text']}\n\n"
                        "Write only the function implementation, no explanations. /no_think"
                    ),
                }
            ], ensure_ascii=False),
            "response": item["code"].strip(),
        })

pd.DataFrame(rows).to_parquet("data/mbpp_reference_sft.parquet", index=False)
```

如果要和当前 MBPP v2 prompt 完全对齐，还应该复用 `prepare_mbpp.py` 的函数名注入逻辑，把 `Your function must be named ...` 也写进 SFT prompt。

实验上要注意：只能用 train split 的 `code` 做 SFT，不能把 test split 的 `code` 放进训练，否则就是评测泄漏。

## Q7：为什么 MBPP 当时不用 `code` 做 SFT，而是用 7B teacher？APPS 却先用 human solutions？

短答案：APPS 每题有多个 human solutions 是原因之一，但不是唯一原因。更核心的区别是当时两条实验线的目标不同。

代码能确认的事实：

- APPS human SFT 明确使用 `solutions` 字段，每题最多取 3 个 Python 解法。
- APPS 脚本注释给出的理由是：human solutions 已验证、质量高、多样性好、不需要 7B inference。
- MBPP teacher SFT 使用 `teacher_response_ids`，并过滤 `teacher_pass=True`。
- MBPP OPD/CL 数据也围绕 `teacher_pass`、`teacher_logprobs`、`teacher_response_ids` 设计。

合理推断：

| 选择 | 主要考虑 |
|---|---|
| APPS 先用 human solutions | APPS 自带多解，数据量大，构造 SFT 成本低；一开始足够作为 warmup 起点 |
| MBPP 用 7B teacher SFT | MBPP 实验重点包含 OPD/teacher distillation，需要 teacher response、teacher pass、teacher logprob 这些字段 |
| 后续 APPS 又做 teacher SFT | 实验发现 APPS human SFT 让 sampling 分布变窄，Teacher SFT 在 LCB 上明显更好 |

所以不是因为 MBPP `code` 不能做 SFT。它能做，只是当时更想验证的是：

```text
teacher 生成正确解 -> 过滤 teacher_pass=True -> student SFT warmup -> GRPO/OPD
```

这条链路和 OPD 目标更一致。

如果现在补一个 MBPP reference-code SFT 对照，是合理的消融：

```text
MBPP code SFT
vs MBPP 7B teacher SFT
vs no SFT
vs SFT + GRPO
```

这个实验能回答：在 MBPP 上，SFT gain 到底来自“任何正确代码 warmup”，还是来自“teacher 风格/teacher 对齐”的额外收益。

## Q8：`precompute_teacher_logprobs.py` 里的 `compute_logprobs_for_response` 在做什么？

位置：`data/precompute_teacher_logprobs.py`

```python
def compute_logprobs_for_response(model, input_ids, response_ids) -> list[float]:
    full_ids = torch.cat([input_ids[0], response_ids], dim=0).unsqueeze(0)
    with torch.no_grad():
        logits = model(full_ids).logits

    prompt_len = input_ids.shape[1]
    resp_logits = logits[0, prompt_len - 1: prompt_len - 1 + len(response_ids), :]
    log_probs = F.log_softmax(resp_logits.float(), dim=-1)
    token_lp = log_probs[torch.arange(len(response_ids)), response_ids].cpu().tolist()
    return token_lp
```

它的作用：计算 teacher 对自己生成的 response 中每个 token 的 log probability。

输入形状：

| 变量 | 形状 | 含义 |
|---|---|---|
| `input_ids` | `[1, prompt_len]` | prompt 经过 chat template 后的 token |
| `response_ids` | `[resp_len]` | teacher greedy 生成出来的 response token |
| `full_ids` | `[1, prompt_len + resp_len]` | prompt 和 response 拼在一起 |
| `logits` | `[1, total_len, vocab]` | 模型每个位置对下一个 token 的预测 |

关键是 causal LM 的 shift：

```text
full_ids:    [prompt_0 ... prompt_last, resp_0, resp_1, ...]
logits pos:   0      ... prompt_len-1,  prompt_len, ...

logits[prompt_len - 1] 预测 resp_0
logits[prompt_len]     预测 resp_1
logits[prompt_len + 1] 预测 resp_2
```

所以 response 对齐的 logits 要从 `prompt_len - 1` 开始切：

```python
resp_logits = logits[0, prompt_len - 1: prompt_len - 1 + len(response_ids), :]
```

然后：

```python
log_probs = F.log_softmax(resp_logits.float(), dim=-1)
```

把每个位置的 vocab logits 变成 log probability。这里转成 `float32` 是为了数值稳定。

最后这一行：

```python
token_lp = log_probs[torch.arange(len(response_ids)), response_ids]
```

是在每个时间步取出“teacher 实际生成的那个 token”的 log probability。

返回值示意：

```text
response_ids:      [tok_a, tok_b, tok_c]
teacher_logprobs:  [log p(tok_a | prompt),
                    log p(tok_b | prompt, tok_a),
                    log p(tok_c | prompt, tok_a, tok_b)]
```

它不做这些事：

- 不训练模型，没有 backward。
- 不计算 reward，通过测试与否由后面的 `run_tests(...)` 决定。
- 不计算 reference `code` 的概率，只计算 teacher 生成 response 的概率。
- 不直接做 SFT；SFT 主要用 `teacher_response_ids`，OPD/蒸馏相关逻辑才需要 `teacher_logprobs`。

## Q9：`token_logprobs` 存到数据里后是用来做什么的？

`precompute_teacher_logprobs.py` 里：

```python
token_logprobs = compute_logprobs_for_response(model, input_ids, response_ids)
df.at[idx, "teacher_logprobs"] = token_logprobs
```

这里存的是 teacher 对自己生成 response 的逐 token log probability。

直觉上，它是 teacher 对每个 token 的“认可分”：

```text
越接近 0：teacher 越确信这个 token 合理
越小的负数：teacher 越不确信这个 token
```

它和另外两个字段一起构成 teacher 数据：

| 字段 | 含义 | 主要用途 |
|---|---|---|
| `teacher_response_ids` | teacher 生成的答案 token | SFT target |
| `teacher_logprobs` | teacher 对这些 token 的逐 token log prob | OPD/蒸馏权重或分析信号 |
| `teacher_pass` | teacher 答案是否通过测试 | 过滤正确 teacher 样本 |

当前仓库里最确定的用途：

- `run_sft_teacher_warmup.py` 主要用 `teacher_response_ids` 做 SFT，并用 `teacher_pass=True` 过滤样本。
- `prepare_mbpp_opd_cl.py` 会检查 `teacher_logprobs` 是否存在，把这类数据视为可用于 OPD/CL 的 teacher 数据。
- OPD 文档和 patch 里把 teacher log-prob 当作 teacher 对 token 的评分，但正确 loss 必须同时包含 live student log-prob。

重要区别：

```text
teacher_logprobs   = parquet 里离线预计算的列，teacher 对 teacher response 打分
teacher_log_probs  = veRL patch 训练时常用的 batch 字段，teacher 对当前 rollout token 打分
```

做 SFT 时，模型只需要知道“应该模仿哪些 token”，所以 `teacher_response_ids` 已经够了。

做 OPD/蒸馏时，才更关心 teacher 对 token 的概率分布。例如：

```text
teacher 高分 token -> student 应该更倾向生成
teacher 低分 token -> student 不应该被强行推太高
```

但不能写成：

```python
loss = -mean(teacher_logprobs)
```

因为 `teacher_logprobs` 是 teacher 算出来的常量，和 student 参数没有关系，梯度为 0。正确方向必须让 student 当前的 `log π_student(...)` 出现在 loss 中，例如把 teacher 分数当 target、权重或 reward，再去更新 student。

## Q10：`precompute_teacher_logprobs.py` 里 ground truth 应该从哪里取？

MBPP v2 parquet 没有顶级 `ground_truth` 列，测试用例存在：

```python
row["reward_model"]["ground_truth"]
```

所以正确写法是：

```python
reward_model = row.get("reward_model", {})
if isinstance(reward_model, dict):
    ground_truth = reward_model.get("ground_truth", "[]")
else:
    ground_truth = "[]"
tests = json.loads(ground_truth) if isinstance(ground_truth, str) else list(ground_truth)
```

如果写成：

```python
ground_truth = row.get("ground_truth", "[]")
```

就会拿到空列表，因为 row 顶层没有这个字段。后果是：

```python
passed, total = run_tests(response_text, [])
teacher_pass = (passed == total and total > 0)
```

其中 `total == 0`，所以 `teacher_pass` 会变成 `False`。如果整份数据都这样跑出来，后续 `run_sft_teacher_warmup.py` 用 `teacher_pass=True` 过滤时就会没有可训练样本。

当前本地脚本已修正为从 `reward_model["ground_truth"]` 取；服务器脚本也已同步修正。已有 teacher parquet 中的 `teacher_pass` 不是全 False，说明当前使用的数据并不是由这个错误逻辑直接全量产出的，或后续曾被重新评测/修正过。

## Q11：还有没有类似的数据字段读取问题？

本次排查结论：

| 问题 | 状态 | 影响 |
|---|---|---|
| MBPP `ground_truth` 顶层误读 | 已修复并同步服务器 | 旧写法会让 `teacher_pass` 全 False |
| MBPP teacher response 未抽取 Markdown code fence | 已修复并同步服务器 | 旧写法会把带 Markdown 代码块的正确代码误判为 `SyntaxError` |
| APPS teacher SFT 的 `prompt` 类型处理 | 已修复并同步服务器 | parquet 读出的是 `ndarray`，现在统一转成 message list |
| MBPP 1.5B teacher 文件的 `teacher_logprobs` | 需注意 | 当前 SFT 不依赖；若做逐 token OPD 需重新生成或补齐 |

APPS prompt 的修复点在 `scripts/prepare_apps_teacher_sft.py`：

```python
def normalize_messages(prompt_obj) -> list[dict]:
    if hasattr(prompt_obj, "tolist"):
        prompt_obj = prompt_obj.tolist()
    if isinstance(prompt_obj, tuple):
        prompt_obj = list(prompt_obj)
    if isinstance(prompt_obj, str):
        try:
            prompt_obj = json.loads(prompt_obj)
        except json.JSONDecodeError:
            return [{"role": "user", "content": prompt_obj}]
    if isinstance(prompt_obj, list):
        return prompt_obj
    return [{"role": "user", "content": str(prompt_obj)}]
```

服务器真实 APPS parquet 的 `prompt` 读出类型是 `ndarray`，修复后会稳定转成：

```text
[{"role": "system", ...}, {"role": "user", ...}]
```

MBPP 1.5B teacher 文件的注意点：`teacher_response_ids` 和 `teacher_pass` 可用于 SFT；但该文件里的 `teacher_logprobs` 为空，因此不能直接拿它做依赖逐 token teacher log-prob 的 OPD。

## Q12：为什么 `precompute_teacher_logprobs.py` 里曾经会把带 Markdown fence 的 teacher response 判错？

7B teacher 生成 MBPP 代码时，很多 response 是这种形式：

````text
```python
def prime_num(n):
    ...
```
````

如果直接把整段 `response_text` 传给裸 `exec()`，第一行的 ```python 会触发：

```text
SyntaxError: invalid syntax
```

正式 reward 路径 `rewards/mbpp_reward.py` 会先调用 `extract_code()`，去掉 `<think>...</think>` 和 Markdown code fence，再执行 assert。因此 `precompute_teacher_logprobs.py` 的 `run_tests()` 也必须复用同一套抽取逻辑。

当前修复后的关键逻辑是：

```python
from rewards.mbpp_reward import extract_code

def run_tests(code: str, tests: list[str]) -> tuple[int, int]:
    code = extract_code(code)
    ...
```

历史上曾经出现过这样的时间线：

- 原始 `precompute_teacher_logprobs.py` 跑出 `teacher_pass=0/374`。
- 后续用临时补算流程重新评测，得到 `teacher_pass=True` 共 192 条。
- 本次修复把临时补算口径沉淀回正式预计算脚本，避免从零重跑时再次得到全 0 pass。

## Q13：`apply_chat_template` 和 `model.generate` 在 teacher 生成里分别做什么？

位置：`data/precompute_teacher_logprobs.py`

```python
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(args.device)

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
```

`messages` 是 chat-format 数据，例如：

```python
[
    {"role": "user", "content": "You are an expert Python programmer..."}
]
```

模型不能直接吃 Python list。`tokenizer.apply_chat_template(...)` 会按当前模型 tokenizer 里的 `chat_template` 把 messages 转成模型训练时熟悉的对话格式，再 tokenize 成 `input_ids`。

常用参数含义：

| 参数 | 含义 |
|---|---|
| `messages` | chat 消息列表，每条通常有 `role` 和 `content` |
| `add_generation_prompt=True` | 在末尾加上 assistant 开始回答的标记，告诉模型现在该生成答案 |
| `return_tensors="pt"` | 返回 PyTorch tensor，形状通常是 `[1, prompt_len]` |
| `tokenize=False` | 不返回 token ids，而是返回格式化后的 prompt 字符串；vLLM 批量推理常用 |

可以用下面的方式查看模型真正看到的格式化文本：

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print(text)
```

`model.generate(...)` 是自回归生成入口。它从 `input_ids` 开始，每一步：

1. 模型 forward，得到下一个 token 的 logits。
2. 根据 decoding 策略选出一个 next token。
3. 把 next token 拼回序列。
4. 继续生成，直到 EOS 或达到 `max_new_tokens`。

这里返回的 `output_ids` 是完整序列：

```text
prompt_ids + generated_response_ids
```

所以后面需要切掉 prompt：

```python
response_ids = output_ids[0, input_ids.shape[1]:]
response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
```

`model.generate` 常见参数：

| 参数 | 含义 |
|---|---|
| `max_new_tokens` | 最多生成多少新 token，不包含 prompt 长度 |
| `do_sample=False` | 不采样；配合默认 `num_beams=1` 就是 greedy decoding |
| `pad_token_id` | padding token id；decoder-only 模型常用 `eos_token_id` 代替 |
| `attention_mask` | 标记哪些 token 是有效输入；单样本无 padding 时通常影响不大，批量 padding 时应显式传 |
| `return_dict_in_generate=True` | 返回包含序列、scores 等字段的结构化对象 |
| `output_scores=True` | 返回每一步生成时的分数，调试采样/概率时有用 |

这段 teacher 生成实现 greedy 的关键是：

```python
do_sample=False
# num_beams 未设置，默认是 1
```

即每一步都选当前概率最高的 token：

```text
next_token = argmax(logits)
```

因此 7B teacher 在 MBPP 预计算中是每题一条 deterministic greedy response，不是多样本采样。

## Q14：multinomial sampling、beam search、temperature/top-k/top-p/repetition 分别是什么？

模型每一步都会输出一个词表大小的 logits，softmax 后是：

```text
p(next_token | 当前上下文)
```

不同 decoding 策略的区别，就是面对这个分布时怎么选下一个 token。

| 策略 | 做法 | 特点 |
|---|---|---|
| Greedy | 每步选概率最高 token | 稳定、保守、可复现 |
| Multinomial sampling | 按概率随机抽 token | 有多样性，适合 RL rollout 生成多条 response |
| Beam search | 同时维护多条高概率路径 | 非随机，搜索全局高概率序列，但更慢、更模板化 |

采样常见配置：

```python
model.generate(
    input_ids,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
)
```

它可以理解成：

1. 模型输出 logits。
2. `temperature=0.7` 调整 logits 分布，低于 1 会让分布更尖锐、更保守。
3. `top_k=50` 只保留排名最高的 50 个 token，其余 token 置为不可选。
4. `top_p=0.95` 在候选集合里按概率从高到低排序，保留累计概率达到 95% 的核心集合。
5. 对最终保留的 token 重新归一化。
6. 按概率随机抽一个 token。

同时设置 `top_k=50` 和 `top_p=0.95` 时，最终候选集合通常体现为二者的交集效果：

```text
最多不超过 50 个 token，同时只保留累计概率核心区域。
```

实现细节上，不同框架的 logits warper 顺序可能略有差异；在 Hugging Face `generate` 的使用心智模型里，可以记成“先经过 temperature/top-k/top-p 等过滤和变形，再从剩余分布中采样”。

重复控制用于降低模型反复生成同一片段的概率，例如：

```python
repetition_penalty=1.1
no_repeat_ngram_size=3
```

含义：

| 参数 | 含义 |
|---|---|
| `repetition_penalty` | 惩罚已经出现过的 token，使其后续概率下降 |
| `no_repeat_ngram_size=3` | 不允许相同的连续 3-token 片段再次出现 |

代码生成里重复控制要谨慎，因为代码天然会重复变量名、括号、缩进、`return`、`for` 等 token；惩罚太强可能伤害正确性。

## Q15：对大模型来说，`system` 和 `user` 有什么区别？不都是 `message_list` 吗？

是的，数据结构上它们都在同一个 `message_list` 里：

```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
]
```

但经过 `apply_chat_template()` 后，不同 role 会被包装成不同的模板标记。模型实际看到的是类似：

```text
<system-role-marker>
全局规则
<user-role-marker>
当前问题
<assistant-generation-marker>
```

具体 marker 由 tokenizer 的 `chat_template` 决定。

语义上：

| role | 常见用途 |
|---|---|
| `system` | 全局行为约束、角色、输出协议、格式要求 |
| `user` | 当前任务输入、具体题目、具体请求 |

APPS 使用 system + user：

```python
[
    {"role": "system", "content": "Write a complete Python 3 program that reads from stdin and writes to stdout..."},
    {"role": "user", "content": "## Problem\n\n..."},
]
```

原因是 APPS/LCB 是完整 stdin/stdout 竞赛题，输出协议比较强：必须写完整 Python 3 程序、读 stdin、写 stdout、不要 debug 输出、最好包在 Markdown code block 里。把这些稳定协议放进 `system`，具体题面放进 `user`，更符合 chat 模型的指令组织方式。

MBPP 只有单条 user：

```python
[
    {"role": "user", "content": "Problem: ... Your function must be named `xxx`. Write only the function implementation..."}
]
```

这不是因为 MBPP 不能用 system，而是因为 MBPP 是短函数任务，约束少：只需要函数名正确、只输出函数实现。单条 user prompt 已经足够表达任务。

## 原始数据示例

以下示例来自 `data/raw/mbpp_hf_full/train.jsonl`，为阅读方便省略了 `code` 全文。

### 示例 1：普通函数任务

```json
{
  "task_id": 601,
  "text": "Write a function to find the longest chain which can be formed from the given set of pairs.",
  "code": "class Pair(object): ... def max_chain_length(arr, n): ...",
  "test_list": [
    "assert max_chain_length([Pair(5, 24), Pair(15, 25),Pair(27, 40), Pair(50, 60)], 4) == 3",
    "assert max_chain_length([Pair(1, 2), Pair(3, 4),Pair(5, 6), Pair(7, 8)], 4) == 4",
    "assert max_chain_length([Pair(19, 10), Pair(11, 12),Pair(13, 14), Pair(15, 16), Pair(31, 54)], 5) == 5"
  ],
  "test_setup_code": "",
  "challenge_test_list": []
}
```

这个样本会被提取出函数名 `max_chain_length`，并在 prompt 中注入：

```text
Your function must be named `max_chain_length`.
```

### 示例 2：简单字符串任务

```json
{
  "task_id": 602,
  "text": "Write a python function to find the first repeated character in a given string.",
  "code": "def first_repeated_char(str1): ...",
  "test_list": [
    "assert first_repeated_char(\"abcabc\") == \"a\"",
    "assert first_repeated_char(\"abc\") == \"None\"",
    "assert first_repeated_char(\"123123\") == \"1\""
  ],
  "test_setup_code": "",
  "challenge_test_list": []
}
```

这个样本会被提取出函数名 `first_repeated_char`。

### 示例 3：未注入函数名的边界样本

```json
{
  "task_id": 769,
  "text": "Write a python function to get the difference between two lists.",
  "code": "def Diff(li1,li2): ...",
  "test_list": [
    "assert (Diff([10, 15, 20, 25, 30, 35, 40], [25, 40, 35])) == [10, 20, 30, 15]",
    "assert (Diff([1,2,3,4,5], [6,7,1])) == [2,3,4,5,6,7]",
    "assert (Diff([1,2,3], [6,7,1])) == [2,3,6,7]"
  ],
  "test_setup_code": "",
  "challenge_test_list": []
}
```

当前正则没有匹配 `assert (Diff(...)) == ...` 这种带括号写法，所以这条不会注入函数名。

### 示例 4：带测试前置代码的边界样本

```json
{
  "task_id": 927,
  "text": "Write a function to calculate the height of the given binary tree.",
  "code": "class Node: ... def max_height(node): ...",
  "test_list": [
    "assert (max_height(root)) == 3",
    "assert (max_height(root1)) == 5 ",
    "assert (max_height(root2)) == 4"
  ],
  "test_setup_code": "root = Node(1) ... root2.left.left.right = Node(7)",
  "challenge_test_list": []
}
```

这条也因为 assert 外层括号没有注入函数名。它还展示了另一个重要字段：`test_setup_code`，里面会构造测试用的树节点。

## 参考链接

- Hugging Face MBPP：<https://huggingface.co/datasets/google-research-datasets/mbpp>
- `load_dataset` 文档：<https://huggingface.co/docs/datasets/package_reference/loading_methods#datasets.load_dataset>
- Google Research MBPP：<https://github.com/google-research/google-research/tree/master/mbpp>
