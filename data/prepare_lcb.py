"""
LiveCodeBench 数据准备脚本
==========================

【什么是 LiveCodeBench？】
LiveCodeBench 是一个"防污染"代码评测基准（https://livecodebench.github.io）。
它的核心设计理念：**持续从真实竞赛平台（LeetCode/Codeforces/AtCoder）爬取新题**。
每隔几个月出一个新版本（v1→v5），新版本只用模型 knowledge cutoff 之后的题，
这样就规避了"训练集里见过这道题"的数据污染问题。

【livecodebench/code_generation_lite 是什么？】
官方在 HuggingFace 上发布了若干数据集：
  - code_generation_lite：题目较少（~400题），适合快速评测
  - code_generation：完整版（~800题+）
每道题的关键字段：
  - question_id, title, difficulty（Easy/Medium/Hard）
  - platform（LeetCode/Codeforces/AtCoder）
  - question_content：题目描述（HTML 格式）
  - starter_code：函数签名（LeetCode 题有，其他平台通常为空）
  - public_tests：公开样例，格式 [{"input": "...", "output": "..."}, ...]

【LeetCode 题 vs Codeforces/AtCoder 题的本质差异】
─────────────────────────────────────────────────────────
  LeetCode（函数式）:
    输入: nums = [1,2,3], target = 2
    期望输出: 1
    → 模型需要写 class Solution 并实现某个方法
    → 执行时需要"把输入解析成函数参数，调用函数，比对返回值"
    → 这需要知道方法签名，格式复杂

  Codeforces / AtCoder（stdin/stdout 式）:
    输入（stdin）: 3\n1 2 3\n
    期望输出（stdout）: 2
    → 模型需要写一个完整的 Python 程序，读 stdin，打印 stdout
    → 执行时: subprocess 传入 input，捕获 stdout，字符串比对
    → 这是标准的 OJ（Online Judge）风格，更简单、统一

【本脚本的策略】
我们做全量下载，但在 parquet 里标注 platform 和 execution_type。
eval 脚本会根据 execution_type 选择不同的执行方式：
  - "stdin_stdout": Codeforces/AtCoder，subprocess 模式
  - "leetcode_fn": LeetCode，函数调用模式（目前跳过，实现复杂）
对于 baseline 对比，Codeforces/AtCoder 的题足够说明问题。

用法：
  python data/prepare_lcb.py --output_dir data/lcb --split v5
  # split: v4（2024年）, v5（2025年前几个月）
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


def clean_html(html: str) -> str:
    """
    LCB 的 question_content 是 HTML 格式，需要转成纯文本给模型。
    简单处理：去掉 HTML 标签，处理 &lt; &gt; 等实体。
    """
    # 常见 HTML 实体
    text = html.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # 去掉所有 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    # 去掉多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_execution_type(platform: str, starter_code: str) -> str:
    """
    判断这道题需要什么执行方式。

    LeetCode 题的 starter_code 通常是：
        class Solution:
            def twoSum(self, nums: List[int], target: int) -> List[int]:

    Codeforces/AtCoder 的 starter_code 通常是空字符串。
    """
    if platform == "LeetCode":
        return "leetcode_fn"
    else:
        # Codeforces, AtCoder
        return "stdin_stdout"


def make_prompt_messages(title: str, content_text: str, execution_type: str,
                         starter_code: str) -> list:
    """
    构造 chat messages（List[Dict]），送给 tokenizer.apply_chat_template。

    【为什么要写"详细的指令"？】
    竞赛题比 MBPP 复杂得多：
    - 可能有多行输入
    - 输出要严格匹配（大小写、空格都算）
    - 不能有多余的 print（否则输出会多一行）
    所以我们在 system prompt 里明确告诉模型应该怎么写代码。

    stdin_stdout 格式的 prompt：
      "用 Python 写一个完整程序，从 stdin 读取输入，输出答案到 stdout。
       不要 print 任何 debug 信息。将代码放在 ```python ... ``` 块中。"

    leetcode_fn 格式（暂时不需要，先留 placeholder）：
      "写 class Solution，实现方法 xxx。"
    """
    if execution_type == "stdin_stdout":
        system = (
            "You are an expert competitive programmer. "
            "Write a complete Python 3 program that reads from stdin and writes to stdout. "
            "Do NOT include any debug prints or extra output. "
            "Wrap your code in a ```python ... ``` block."
        )
        user = f"## Problem: {title}\n\n{content_text}"
    else:
        # leetcode_fn - 使用 starter_code
        system = (
            "You are an expert programmer. "
            "Complete the given Python function. "
            "Wrap your solution in a ```python ... ``` block."
        )
        user = f"## Problem: {title}\n\n{content_text}"
        if starter_code.strip():
            user += f"\n\n## Starter Code\n```python\n{starter_code.strip()}\n```"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_tests(item: dict) -> list:
    """
    从 LCB v5 JSONL 题目中提取测试用例。

    实测字段名（v5）：
      - public_test_cases:  JSON 字符串，格式 [{"input": "...", "output": "...", "testtype": "stdin"}, ...]
      - private_test_cases: base64/zlib 压缩，无法直接解析，跳过

    testtype 目前只见过 "stdin"，代表 stdin/stdout 执行模式。
    LeetCode 题的 public_test_cases 也是这个格式，但 input 是 Python 字面量（如 "nums = [1,2,3]"），
    不能直接用 subprocess stdin 执行，需要额外处理。
    """
    raw = item.get("public_test_cases", None)
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    tests = []
    if isinstance(raw, list):
        for t in raw:
            if isinstance(t, dict) and "input" in t and "output" in t:
                tests.append({
                    "input": str(t["input"]),
                    "output": str(t["output"]),
                    "testtype": t.get("testtype", "stdin"),
                })
    return tests


def process_items(items: list, split_tag: str) -> pd.DataFrame:
    """
    把 JSONL 记录列表转成 DataFrame。

    字段映射（v5 实测）：
      JSONL 字段          → DataFrame 列
      question_title      → title
      question_content    → (解析为 prompt 的 content_text)
      platform            → platform
      question_id         → question_id
      difficulty          → difficulty
      starter_code        → (用于构造 prompt)
      public_test_cases   → tests
    """
    rows = []
    skipped_no_tests = 0

    for item in items:
        platform = item.get("platform", "Unknown")
        starter_code = item.get("starter_code", "")
        execution_type = detect_execution_type(platform, starter_code)

        tests = parse_tests(item)
        # 只保留 stdin 类型的测试用例（过滤 LeetCode 的函数调用格式）
        stdin_tests = [t for t in tests if t.get("testtype", "stdin") == "stdin"]

        if not stdin_tests:
            skipped_no_tests += 1
            continue

        content_html = item.get("question_content", "")
        content_text = clean_html(content_html)
        title = item.get("question_title", "")  # v5 用 question_title

        prompt_messages = make_prompt_messages(
            title=title,
            content_text=content_text,
            execution_type=execution_type,
            starter_code=starter_code,
        )

        rows.append({
            "question_id": item.get("question_id", ""),
            "title": title,
            "difficulty": item.get("difficulty", "Unknown"),
            "platform": platform,
            "execution_type": execution_type,
            "prompt": json.dumps(prompt_messages, ensure_ascii=False),
            "tests": json.dumps(stdin_tests, ensure_ascii=False),
            "split_tag": split_tag,
        })

    print(f"  Processed {len(rows)} problems, skipped {skipped_no_tests} (no stdin tests)")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare LiveCodeBench dataset")
    parser.add_argument("--output_dir", default="data/lcb",
                        help="Output directory for parquet files")
    parser.add_argument("--split", default="v5",
                        choices=["v1", "v2", "v3", "v4", "v5", "v6"],
                        help="LCB release version")
    parser.add_argument("--jsonl_path", default=None,
                        help="直接指定本地 JSONL 文件路径，跳过下载（用于离线环境）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # 【为什么用 hf_hub_download 而不是 load_dataset？】
    # datasets >= 3.0 不再支持 dataset scripts（.py 格式的加载器），
    # livecodebench/code_generation_lite 恰好使用了 script 格式。
    # 直接下载 JSONL 原始文件是最简单的绕过方式。
    # ─────────────────────────────────────────────────────────────────────
    if args.jsonl_path:
        jsonl_path = Path(args.jsonl_path)
        print(f"Using local JSONL: {jsonl_path}")
    else:
        # split 数字对应 HF repo 里的文件名：v5 → test5.jsonl
        split_num = args.split.lstrip("v")
        filename = f"test{split_num}.jsonl"
        print(f"Downloading {filename} from livecodebench/code_generation_lite ...")
        jsonl_path = Path(hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename=filename,
            repo_type="dataset",
            local_dir=str(output_dir / "raw"),
        ))
        print(f"  Downloaded to {jsonl_path}")

    # 读取 JSONL
    items = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"  Loaded {len(items)} problems from JSONL")

    df = process_items(items, split_tag=args.split)

    # 打印分布
    print("\n--- Dataset Statistics ---")
    print(f"Total problems (with tests): {len(df)}")
    print("\nBy platform:")
    print(df["platform"].value_counts().to_string())
    print("\nBy difficulty:")
    print(df["difficulty"].value_counts().to_string())
    print("\nBy execution_type:")
    print(df["execution_type"].value_counts().to_string())

    # 保存完整集合
    full_path = output_dir / f"lcb_{args.split}_all.parquet"
    df.to_parquet(full_path, index=False)
    print(f"\nSaved full dataset: {full_path} ({len(df)} problems)")

    # 保存 stdin_stdout 子集（我们的主要评测集）
    df_io = df[df["execution_type"] == "stdin_stdout"].reset_index(drop=True)
    io_path = output_dir / f"lcb_{args.split}_stdin_stdout.parquet"
    df_io.to_parquet(io_path, index=False)
    print(f"Saved stdin_stdout subset: {io_path} ({len(df_io)} problems)")

    # 保存 leetcode 子集（暂时仅保存，eval 脚本暂不支持）
    df_lc = df[df["execution_type"] == "leetcode_fn"].reset_index(drop=True)
    lc_path = output_dir / f"lcb_{args.split}_leetcode.parquet"
    df_lc.to_parquet(lc_path, index=False)
    print(f"Saved leetcode subset: {lc_path} ({len(df_lc)} problems)")

    print("\nDone! Use lcb_{split}_stdin_stdout.parquet for evaluation.")

    # ─────────────────────────────────────────────────────────────────────
    # 额外保存 veRL 训练格式（用作 val_data 在 GRPO 训练中）
    # 格式要求：和 MBPP parquet 完全一致
    #   prompt:       list of dicts（numpy array）
    #   data_source:  "lcb"
    #   reward_model: {"ground_truth": JSON string of tests}
    #   extra_info:   {"question_id": ..., "difficulty": ...}
    # ─────────────────────────────────────────────────────────────────────
    import numpy as np
    verl_rows = []
    for _, row in df_io.iterrows():
        messages = json.loads(row["prompt"])
        verl_rows.append({
            "prompt": np.array(messages, dtype=object),
            "data_source": "lcb",
            "reward_model": {
                "ground_truth": row["tests"],  # 已经是 JSON 字符串
            },
            "extra_info": {
                "question_id": row["question_id"],
                "difficulty": row["difficulty"],
            },
        })
    df_verl = pd.DataFrame(verl_rows)
    verl_path = output_dir / f"lcb_{args.split}_verl.parquet"
    df_verl.to_parquet(verl_path, index=False)
    print(f"Saved veRL-format val dataset: {verl_path} ({len(df_verl)} problems)")


if __name__ == "__main__":
    main()
