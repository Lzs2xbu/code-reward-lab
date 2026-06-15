# data/prepare_mbpp.py
# 本地运行：python data/prepare_mbpp.py
# 输出：data/mbpp_train.parquet
#       data/mbpp_test.parquet

import json
import pandas as pd
from datasets import load_dataset

PROMPT_TEMPLATE = """You are an expert Python programmer. Solve the following problem by writing a Python function.

Problem: {text}

Write only the function implementation, no explanations. /no_think"""


def make_row(item):
    content = PROMPT_TEMPLATE.format(text=item["text"].strip())
    # veRL 要求 prompt 是 chat format（list of dicts），与 GSM8K parquet 格式一致
    prompt = [{"role": "user", "content": content}]
    ground_truth = json.dumps(item["test_list"])  # JSON 序列化，服务器端 json.loads 还原
    return {
        "prompt": prompt,
        "data_source": "mbpp",
        "reward_model": {"ground_truth": ground_truth},
        "extra_info": {"task_id": item["task_id"]},
    }


def main():
    ds = load_dataset("google-research-datasets/mbpp", "full")

    for split_name, split_key in [("train", "train"), ("test", "test")]:
        split = ds[split_key]
        rows = [make_row(item) for item in split]
        df = pd.DataFrame(rows)
        out_path = f"data/mbpp_{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Saved {len(df)} rows to {out_path}")
        # 打印前两行验证结构
        print(df.iloc[0].to_dict())
        print()


if __name__ == "__main__":
    main()
