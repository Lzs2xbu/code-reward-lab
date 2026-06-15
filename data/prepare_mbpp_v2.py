"""
准备 MBPP v2 数据集：在 prompt 中加入函数名，修复 reward 极度稀疏的问题。

问题背景：
  v1 数据集的 prompt 只有功能描述，没有函数名。Ground truth 的 assert 语句
  要求精确的函数名（如 first_repeated_char），模型只能靠猜，导致 97.6% 的
  样本 reward=0，GRPO 无法产生有效梯度信号。

修复方案：
  从 ground truth 的第一条 assert 中提取函数名，注入 prompt：
    "Your function must be named `first_repeated_char`."

用法：
  python data/prepare_mbpp_v2.py \
      --input_dir data/mbpp \
      --output_dir data/mbpp_v2

输出：
  data/mbpp_v2/mbpp_train.parquet
  data/mbpp_v2/mbpp_test.parquet
"""

import argparse
import json
import os
import re

import pandas as pd


PROMPT_TEMPLATE = (
    "You are an expert Python programmer. Solve the following problem by writing a Python function.\n\n"
    "Problem: {problem}\n\n"
    "Your function must be named `{func_name}`.\n\n"
    "Write only the function implementation, no explanations. /no_think"
)


def extract_func_name(ground_truth_json: str) -> str | None:
    """从 ground truth JSON 的第一条 assert 中提取函数名。"""
    try:
        test_list = json.loads(ground_truth_json)
        for t in test_list:
            m = re.match(r"assert\s+(\w+)\s*\(", t.strip())
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


def extract_problem(old_content: str) -> str:
    """从旧 prompt 中提取 Problem: ... 那行。"""
    m = re.search(r"Problem:\s*(.+?)(?:\n\n|$)", old_content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return old_content.strip()


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    skipped = 0
    for _, row in df.iterrows():
        gt_json = row["reward_model"]["ground_truth"]
        func_name = extract_func_name(gt_json)

        if func_name is None:
            # 极少数样本 assert 格式不标准，保留原 prompt，跳过注入
            skipped += 1
            rows.append(row.to_dict())
            continue

        old_content = row["prompt"][0]["content"]
        problem = extract_problem(old_content)
        new_content = PROMPT_TEMPLATE.format(problem=problem, func_name=func_name)

        new_row = row.to_dict()
        new_row["prompt"] = [{"role": "user", "content": new_content}]
        rows.append(new_row)

    if skipped:
        print(f"  警告：{skipped} 条样本无法提取函数名，保留原始 prompt")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/mbpp",
                        help="v1 parquet 所在目录")
    parser.add_argument("--output_dir", default="data/mbpp_v2",
                        help="v2 parquet 输出目录")
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for split in ("train", "test"):
        src = os.path.join(input_dir, f"mbpp_{split}.parquet")
        dst = os.path.join(output_dir, f"mbpp_{split}.parquet")

        if not os.path.exists(src):
            print(f"跳过 {src}（不存在）")
            continue

        print(f"处理 {src} ...")
        df = pd.read_parquet(src)
        print(f"  原始样本数: {len(df)}")

        df_v2 = process_dataframe(df)
        df_v2.to_parquet(dst, index=False)
        print(f"  已保存到 {dst}")

        # 抽样展示几条对比
        print(f"\n  === 前3条 prompt 对比 ===")
        for i in range(min(3, len(df_v2))):
            old_c = df.iloc[i]["prompt"][0]["content"]
            new_c = df_v2.iloc[i]["prompt"][0]["content"]
            gt = json.loads(df_v2.iloc[i]["reward_model"]["ground_truth"])
            func_name = extract_func_name(df_v2.iloc[i]["reward_model"]["ground_truth"])
            print(f"  Row {i}: func_name={func_name}")
            print(f"    OLD prompt末尾: ...{old_c[-80:]!r}")
            print(f"    NEW prompt末尾: ...{new_c[-80:]!r}")
            print(f"    assert[0]: {gt[0][:80]}")
        print()


if __name__ == "__main__":
    main()
