"""
MBPP pass@1 / pass@5 评测脚本
用法：
  python eval/eval_mbpp.py \
      --model_path models/Qwen3-1.7B \
      --test_file data/mbpp/mbpp_test.parquet \
      --output_file <repo>/eval_results/baseline.json \
      --n_samples 20 \
      --temperature 0.8 \
      --label baseline
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent.parent))
from rewards.mbpp_reward import extract_code, run_in_sandbox


def pass_at_k(n: int, c: int, k: int) -> float:
    """无偏估计公式：pass@k = 1 - C(n-c, k) / C(n, k)"""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def evaluate(model_path: str, test_file: str, output_file: str,
             n_samples: int, temperature: float, label: str):
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    df = pd.read_parquet(test_file)
    print(f"Test set: {len(df)} problems")

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        max_tokens=1024,
    )

    # prompt 是 chat format list-of-dicts，需要 apply_chat_template 转成字符串
    raw_prompts = df["prompt"].tolist()
    prompts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in raw_prompts
    ]
    print(f"Generating {n_samples} samples per problem...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    pass1_list = []
    pass5_list = []

    for i, (output, row) in enumerate(zip(outputs, df.itertuples())):
        test_list = json.loads(row.reward_model["ground_truth"])
        task_id = row.extra_info.get("task_id", i)

        # 对每条生成的回答跑沙箱测试
        c = 0  # 通过全部测试用例的回答数
        per_sample = []
        for sample_output in output.outputs:
            code = extract_code(sample_output.text)
            passed_all = all(run_in_sandbox(code, t) for t in test_list)
            if passed_all:
                c += 1
            per_sample.append(passed_all)

        p1 = pass_at_k(n_samples, c, 1)
        p5 = pass_at_k(n_samples, c, 5)
        pass1_list.append(p1)
        pass5_list.append(p5)

        results.append({
            "task_id": task_id,
            "c": c,
            "n": n_samples,
            "pass@1": p1,
            "pass@5": p5,
            "per_sample": per_sample,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(df)}] running pass@1={sum(pass1_list)/len(pass1_list):.3f}")

    avg_pass1 = sum(pass1_list) / len(pass1_list)
    avg_pass5 = sum(pass5_list) / len(pass5_list)

    summary = {
        "label": label,
        "model_path": model_path,
        "n_samples": n_samples,
        "temperature": temperature,
        "num_problems": len(df),
        "pass@1": avg_pass1,
        "pass@5": avg_pass5,
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({"summary": summary, "per_problem": results}, f, indent=2)

    print(f"\nResults for {label}:")
    print(f"  pass@1 = {avg_pass1:.4f} ({avg_pass1*100:.1f}%)")
    print(f"  pass@5 = {avg_pass5:.4f} ({avg_pass5*100:.1f}%)")
    print(f"  Saved to {output_file}")
    return summary


def print_comparison(summaries: list):
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Label':<20} {'pass@1':>10} {'pass@5':>10}")
    print("-" * 42)
    baseline_p1 = None
    baseline_p5 = None
    for s in summaries:
        label = s["label"]
        p1 = s["pass@1"]
        p5 = s["pass@5"]
        if label == "baseline":
            baseline_p1, baseline_p5 = p1, p5
            print(f"{label:<20} {p1*100:>9.1f}% {p5*100:>9.1f}%")
        else:
            d1 = f"(+{(p1-baseline_p1)*100:.1f}%)" if baseline_p1 else ""
            d5 = f"(+{(p5-baseline_p5)*100:.1f}%)" if baseline_p5 else ""
            print(f"{label:<20} {p1*100:>9.1f}% {p5*100:>9.1f}%  {d1} {d5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_file", default="data/mbpp/mbpp_test.parquet")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    summary = evaluate(
        model_path=args.model_path,
        test_file=args.test_file,
        output_file=args.output_file,
        n_samples=args.n_samples,
        temperature=args.temperature,
        label=args.label,
    )
