"""
Prepare MBPP data for veRL code-RL experiments.

This is the canonical MBPP preprocessing entrypoint. It downloads MBPP from
Hugging Face and writes veRL-compatible parquet files with the required function
name injected into each prompt.

Why function-name injection matters:
  MBPP tests call a specific function name in assert statements. If the prompt
  only describes the task but does not tell the model the expected function
  name, correct-looking code can still receive reward=0 because the function
  name is wrong.

Usage:
  python data/prepare_mbpp.py --output_dir data/mbpp_v2

Outputs:
  data/mbpp_v2/mbpp_train.parquet
  data/mbpp_v2/mbpp_test.parquet
"""

import argparse
import json
import os
import re
from typing import Iterable

import pandas as pd
from datasets import load_dataset


PROMPT_TEMPLATE = """You are an expert Python programmer. Solve the following problem by writing a Python function.

Problem: {text}

Your function must be named `{func_name}`.

Write only the function implementation, no explanations. /no_think"""

LEGACY_PROMPT_TEMPLATE = """You are an expert Python programmer. Solve the following problem by writing a Python function.

Problem: {text}

Write only the function implementation, no explanations. /no_think"""


def extract_func_name(test_list: Iterable[str]) -> str | None:
    """Extract the tested function name from MBPP assert statements."""
    for test in test_list:
        match = re.match(r"\s*assert\s+([A-Za-z_]\w*)\s*\(", str(test).strip())
        if match:
            return match.group(1)
    return None


def build_prompt(problem_text: str, func_name: str | None, legacy_no_function_name: bool) -> str:
    """Build the user prompt content for one MBPP task."""
    text = problem_text.strip()
    if legacy_no_function_name or func_name is None:
        return LEGACY_PROMPT_TEMPLATE.format(text=text)
    return PROMPT_TEMPLATE.format(text=text, func_name=func_name)


def make_row(item: dict, legacy_no_function_name: bool = False) -> tuple[dict, bool]:
    """Convert one MBPP item into the veRL parquet row format.

    Returns:
        A tuple of (row, injected), where injected is False when no function
        name was found or when legacy mode is enabled.
    """
    test_list = list(item["test_list"])
    func_name = extract_func_name(test_list)
    content = build_prompt(
        problem_text=item["text"],
        func_name=func_name,
        legacy_no_function_name=legacy_no_function_name,
    )
    prompt = [{"role": "user", "content": content}]
    ground_truth = json.dumps(test_list)
    row = {
        "prompt": prompt,
        "data_source": "mbpp",
        "reward_model": {"ground_truth": ground_truth},
        "extra_info": {"task_id": item["task_id"]},
    }
    return row, (func_name is not None and not legacy_no_function_name)


def build_dataframe(split, legacy_no_function_name: bool = False) -> tuple[pd.DataFrame, int]:
    """Build one split and return the dataframe plus injection count."""
    rows = []
    injected_count = 0
    for item in split:
        row, injected = make_row(item, legacy_no_function_name=legacy_no_function_name)
        rows.append(row)
        injected_count += int(injected)
    return pd.DataFrame(rows), injected_count


def write_mbpp_dataset(
    output_dir: str,
    dataset_name: str = "google-research-datasets/mbpp",
    dataset_config: str = "full",
    legacy_no_function_name: bool = False,
    preview_rows: int = 2,
) -> None:
    """Download MBPP and write train/test parquet files."""
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset(dataset_name, dataset_config)

    for split_name, split_key in [("train", "train"), ("test", "test")]:
        df, injected_count = build_dataframe(
            ds[split_key],
            legacy_no_function_name=legacy_no_function_name,
        )
        out_path = os.path.join(output_dir, f"mbpp_{split_name}.parquet")
        df.to_parquet(out_path, index=False)

        print(f"Saved {len(df)} rows to {out_path}")
        print(f"  function names injected: {injected_count}/{len(df)}")
        if injected_count < len(df) and not legacy_no_function_name:
            print(f"  warning: {len(df) - injected_count} rows kept without function-name injection")

        for i in range(min(preview_rows, len(df))):
            sample = df.iloc[i].to_dict()
            prompt_tail = sample["prompt"][0]["content"][-120:]
            tests = json.loads(sample["reward_model"]["ground_truth"])
            print(f"  preview row {i}: task_id={sample['extra_info']['task_id']}")
            print(f"    prompt tail: ...{prompt_tail!r}")
            print(f"    assert[0]: {tests[0][:100]}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/mbpp_v2")
    parser.add_argument("--dataset_name", default="google-research-datasets/mbpp")
    parser.add_argument("--dataset_config", default="full")
    parser.add_argument(
        "--legacy-no-function-name",
        action="store_true",
        help="Reproduce the early v1 prompt format without function-name injection.",
    )
    parser.add_argument("--preview_rows", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_mbpp_dataset(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        legacy_no_function_name=args.legacy_no_function_name,
        preview_rows=args.preview_rows,
    )


if __name__ == "__main__":
    main()
