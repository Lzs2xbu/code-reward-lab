"""
APPS 数据集准备脚本（用于 LCB 实验的 RL 训练数据）
===================================================

【APPS 是什么？】
APPS（Automated Programming Progress Standard）是 2021 年发布的竞赛编程数据集，
包含约 10,000 道题（5000 训练 / 5000 测试）。
来源：Codeforces、Kattis、AtCoder 等，和 LCB 的题目来源高度重叠。
参考：https://github.com/hendrycks/apps

【为什么用 APPS 的人工解法做 SFT，而不是用 7B 生成？】
APPS 自带 `solutions` 字段，是人类程序员提交的正确解法（已验证通过测试）。
  优点：
  1. 质量高，逻辑清晰，不需要 7B inference（省时省资源）
  2. 多样性好（一题多种写法）
  3. 已经过验证，不需要沙箱过滤
  缺点：
  1. 不是"7B 风格"——SFT 不是在模仿 teacher，而是在学人类写法
  2. 有些解法比较老（Python 2 风格，奇怪的变量名）
  对于我们的目标（SFT warmup + GRPO），人类解法足够好。

【APPS 数据格式】
  - id: 题目 ID
  - question: 题目描述（纯文本，不是 HTML）
  - solutions: JSON 字符串，格式 ["python_code_1", "python_code_2", ...]
  - input_output: JSON 字符串，格式 {"inputs": ["inp1", "inp2"], "outputs": ["out1", "out2"]}
    注意：inputs/outputs 是并行列表，每个元素是一个完整测试用例的输入/输出
  - difficulty: "introductory" / "interview" / "competition"
  - starter_code: 通常为空

【本脚本输出两个文件】
  1. apps_train_rl.parquet  → RL 训练用（和 MBPP 格式一致）
  2. apps_sft.parquet       → SFT warmup 用（包含解法文本）

在本地运行，生成后 scp 上传到服务器：
  python data/prepare_apps.py --output_dir <tmp_dir>/apps_data
  scp -P <port> <tmp_dir>/apps_data/apps_*.parquet <user>@<host>:<remote_data_dir>/apps/
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


# ──────────────────────────────────────────────────────────────────────
# 测试用例解析
# ──────────────────────────────────────────────────────────────────────

def normalize_io_value(val) -> str:
    """
    APPS 的 input/output 值有两种格式：
      格式A（字符串）: "5\n1 2 3\n"              → 直接用
      格式B（列表）:   ["5", "1 2 3"]            → join 成 "5\n1 2 3\n"

    【为什么 APPS 有两种格式？】
    APPS 是从多个平台爬取的，不同平台的数据预处理方式不一致。
    格式B 的列表每个元素是一行输入（不带换行），需要拼接成 stdin 格式。
    """
    if isinstance(val, list):
        return "\n".join(str(x) for x in val) + "\n"
    return str(val)


def parse_input_output(io_str: str) -> list:
    """
    解析 APPS input_output 字段，兼容字符串格式和列表格式。
    统一输出格式：[{"input": "5\n1 2 3\n", "output": "YES\n"}, ...]
    """
    if not io_str:
        return []
    try:
        io = json.loads(io_str)
    except Exception:
        return []

    inputs = io.get("inputs", [])
    outputs = io.get("outputs", [])
    if not inputs or not outputs:
        return []

    tests = []
    for inp, out in zip(inputs, outputs):
        tests.append({
            "input": normalize_io_value(inp),
            "output": normalize_io_value(out),
        })
    return tests


# ──────────────────────────────────────────────────────────────────────
# Prompt 构造
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert competitive programmer. "
    "Write a complete Python 3 program that reads from stdin and writes to stdout. "
    "Do NOT include any debug prints or extra output. "
    "Wrap your code in a ```python ... ``` block."
)


def make_prompt_messages(question: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"## Problem\n\n{question}"},
    ]


def make_sft_text(question: str, solution: str) -> str:
    """
    SFT 数据格式：把解法包装成 ```python ... ``` 块。
    SFT 训练时，模型看到 question 后需要预测整个 response（含代码块）。
    """
    return f"```python\n{solution.strip()}\n```"


# ──────────────────────────────────────────────────────────────────────
# 解法提取
# ──────────────────────────────────────────────────────────────────────

def get_python_solutions(solutions_str: str, max_per_problem: int = 3) -> list:
    """
    从 APPS solutions 字段提取 Python 3 解法。
    过滤规则：
    1. 不包含 `input()` 调用（因为是 stdin 读取，必须有）
       实际上 APPS 的题都是 stdin/stdout，有 input() 是正常的
    2. 不包含明显的 Python 2 语法（print "..." 形式）
    3. 取前 max_per_problem 个（避免同一题太多重复）
    """
    if not solutions_str:
        return []
    try:
        solutions = json.loads(solutions_str)
    except Exception:
        return []

    python3_solutions = []
    for sol in solutions:
        if not isinstance(sol, str) or not sol.strip():
            continue
        # 简单的 Python 2 过滤：有 print "..." 格式
        if 'print "' in sol or "print '" in sol:
            continue
        # 过滤掉非常短的解法（可能是错误/空实现）
        if len(sol.strip()) < 50:
            continue
        python3_solutions.append(sol.strip())
        if len(python3_solutions) >= max_per_problem:
            break

    return python3_solutions


# ──────────────────────────────────────────────────────────────────────
# 主处理逻辑
# ──────────────────────────────────────────────────────────────────────

def process_train_file(jsonl_path: Path, difficulty_filter: list = None,
                       max_problems: int = None) -> tuple:
    """
    处理 APPS train.jsonl，返回 (rl_df, sft_df)。

    difficulty_filter: 只保留指定难度，None 表示全部
      可选值: ["introductory", "interview", "competition"]
      - introductory: 入门级，大约 2639 题
      - interview:    中级竞赛题，大约 5000 题  ← 最适合我们
      - competition:  竞赛级（非常难），大约 2361 题

    为什么主要用 interview 难度？
      - introductory 对 1.5B 太简单，和 LCB 难度不匹配
      - competition 太难，7B 也常常做不出来，GRPO reward 会接近 0
      - interview 难度和 LCB 的 medium/hard AtCoder 题接近
    """
    rl_rows = []
    sft_rows = []
    skipped = {"no_tests": 0, "no_solutions": 0, "difficulty_filtered": 0}

    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if max_problems and i >= max_problems:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            difficulty = item.get("difficulty", "")
            if difficulty_filter and difficulty not in difficulty_filter:
                skipped["difficulty_filtered"] += 1
                continue

            question = item.get("question", "").strip()
            if not question:
                continue

            # RL 训练数据
            tests = parse_input_output(item.get("input_output", ""))
            if tests:
                prompt_messages = make_prompt_messages(question)
                # ── veRL 要求的固定列格式（和 MBPP parquet 完全一致）──
                # prompt:        list of dicts，格式 [{"role": ..., "content": ...}, ...]
                # data_source:   字符串，用于 reward 函数路由和 metric 命名
                # reward_model:  dict，{"ground_truth": JSON 字符串}
                # extra_info:    dict，额外信息（question_id、difficulty 等）
                rl_rows.append({
                    "prompt": prompt_messages,             # list of dicts（不是 JSON 字符串）
                    "data_source": "apps",
                    "reward_model": {
                        "ground_truth": json.dumps(tests[:5], ensure_ascii=False),
                    },
                    "extra_info": {
                        "question_id": str(item.get("id", i)),
                        "difficulty": difficulty,
                    },
                })
            else:
                skipped["no_tests"] += 1

            # SFT 数据：每题取最多3个不同解法
            solutions = get_python_solutions(item.get("solutions", ""))
            if not solutions:
                skipped["no_solutions"] += 1
                continue

            prompt_messages = make_prompt_messages(question)
            for sol in solutions:
                sft_rows.append({
                    "question_id": str(item.get("id", i)),
                    "difficulty": difficulty,
                    "prompt": json.dumps(prompt_messages, ensure_ascii=False),
                    "response": make_sft_text(question, sol),
                })

    print(f"  RL rows: {len(rl_rows)}, SFT rows: {len(sft_rows)}")
    print(f"  Skipped: {skipped}")

    # 打乱顺序（SFT 训练需要随机顺序）
    random.shuffle(rl_rows)
    random.shuffle(sft_rows)

    return pd.DataFrame(rl_rows), pd.DataFrame(sft_rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare APPS dataset")
    parser.add_argument("--output_dir", default="data/apps")
    parser.add_argument("--jsonl_path", default=None,
                        help="本地 train.jsonl 路径（跳过下载）")
    parser.add_argument("--difficulty", nargs="+",
                        default=["interview"],
                        choices=["introductory", "interview", "competition"],
                        help="要包含的难度级别")
    parser.add_argument("--max_problems", type=int, default=None,
                        help="调试用：只处理前 N 题")
    args = parser.parse_args()

    random.seed(42)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.jsonl_path:
        jsonl_path = Path(args.jsonl_path)
        print(f"Using local JSONL: {jsonl_path}")
    else:
        print("Downloading APPS train.jsonl from HuggingFace...")
        jsonl_path = Path(hf_hub_download(
            repo_id="codeparrot/apps",
            filename="train.jsonl",
            repo_type="dataset",
            local_dir=str(output_dir / "raw"),
        ))
        print(f"  Downloaded to {jsonl_path}")

    print(f"\nProcessing (difficulty={args.difficulty})...")
    rl_df, sft_df = process_train_file(
        jsonl_path,
        difficulty_filter=args.difficulty,
        max_problems=args.max_problems,
    )

    # 打印统计
    print("\n--- Statistics ---")
    if not rl_df.empty:
        print(f"RL dataset: {len(rl_df)} problems")
        diffs = [r["difficulty"] for r in rl_df["extra_info"]]
        from collections import Counter
        print("  Difficulty:", dict(Counter(diffs)))
    if not sft_df.empty:
        print(f"SFT dataset: {len(sft_df)} samples")

    # 保存
    diff_tag = "_".join(sorted(args.difficulty))
    rl_path = output_dir / f"apps_rl_{diff_tag}.parquet"
    sft_path = output_dir / f"apps_sft_{diff_tag}.parquet"

    rl_df.to_parquet(rl_path, index=False)
    sft_df.to_parquet(sft_path, index=False)
    print(f"\nSaved RL data:  {rl_path} ({len(rl_df)} problems)")
    print(f"Saved SFT data: {sft_path} ({len(sft_df)} samples)")
    print("\nNext step: scp these files to server data/apps/")


if __name__ == "__main__":
    main()
