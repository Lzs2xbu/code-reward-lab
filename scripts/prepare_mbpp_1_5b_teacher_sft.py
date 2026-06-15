"""
生成 1.5B GRPO best 模型的 MBPP 解法，作为 SFT warmup 数据。

目的：替换原来的 7B teacher 数据，使 SFT warmup 与 OPD teacher（1.5B）对齐，
      减小初始 teacher-student 分布差距，改善 OPD shaped reward 实验。

输出：data/mbpp_v2/mbpp_train_1_5b_teacher.parquet
      格式与 mbpp_train_with_teacher.parquet 相同（teacher_response_ids, teacher_pass 等）

用法：
  python3 <repo>/scripts/prepare_mbpp_1_5b_teacher_sft.py \
      --model models/coder_1_5b_mbpp_grpo_best \
      --train_file data/mbpp_v2/mbpp_train.parquet \
      --output data/mbpp_v2/mbpp_train_1_5b_teacher.parquet \
      --n_samples 5 \
      --temperature 0.8
"""

import argparse, os, sys, json, re, subprocess, tempfile, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default=os.path.expanduser("models/coder_1_5b_mbpp_grpo_best"))
    p.add_argument("--train_file",  default=os.path.expanduser("data/mbpp_v2/mbpp_train.parquet"))
    p.add_argument("--reward_fn",   default=os.path.expanduser("<repo>/rewards/mbpp_reward.py"))
    p.add_argument("--output",      default=os.path.expanduser("data/mbpp_v2/mbpp_train_1_5b_teacher.parquet"))
    p.add_argument("--n_samples",   type=int,   default=5,    help="samples per problem")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_tokens",  type=int,   default=512)
    p.add_argument("--batch_size",  type=int,   default=32)
    return p.parse_args()


def load_reward_fn(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("reward", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_score


def run_solution(solution_code, extra_info):
    """Run solution and check if it passes using subprocess sandbox."""
    try:
        test_cases = extra_info.get("test_cases", []) if isinstance(extra_info, dict) else []
        if not test_cases:
            return False
        # Simple execution check
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(solution_code)
            f.write('\n')
            # Write a simple test
            f.write('import sys\n')
            passed = 0
            for tc in test_cases[:3]:  # test first 3 cases
                inp = tc.get("input", "")
                expected = tc.get("output", "")
                f.write(f'try:\n    result = str({inp})\n    assert result == {repr(expected)}\n    sys.stdout.write("PASS\\n")\nexcept Exception as e:\n    sys.stdout.write(f"FAIL: {{e}}\\n")\n')
            fname = f.name
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=5
        )
        os.unlink(fname)
        output = result.stdout
        pass_count = output.count("PASS")
        total = min(len(test_cases), 3)
        return pass_count == total and total > 0
    except Exception:
        return False


def main():
    args = parse_args()
    print(f"Loading model: {args.model}")
    print(f"Train file: {args.train_file}")

    # Load vLLM for fast generation
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.7,
        max_model_len=1024,
        enforce_eager=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load reward function
    import importlib.util
    spec = importlib.util.spec_from_file_location("reward", args.reward_fn)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    compute_score = mod.compute_score

    # Load training data
    df = pd.read_parquet(args.train_file)
    print(f"Loaded {len(df)} training problems")

    results = []
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.n_samples,
        stop=["```\n\n", "\n\n\n"],
    )

    # Build prompts
    all_prompts = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        if isinstance(prompt, (list, np.ndarray)):
            prompt = list(prompt)
        else:
            prompt = [{"role": "user", "content": str(prompt)}]
        formatted = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        all_prompts.append(formatted)

    print(f"Generating {args.n_samples} solutions per problem for {len(all_prompts)} problems...")
    outputs = llm.generate(all_prompts, sampling)
    print("Generation done. Evaluating solutions...")

    for idx, (row_tuple, output) in enumerate(zip(df.itertuples(), outputs)):
        row = df.iloc[idx]
        extra_info = row.get("extra_info", {})
        if isinstance(extra_info, str):
            try: extra_info = json.loads(extra_info)
            except: extra_info = {}

        best_response_text = None
        best_response_ids  = None
        best_logprobs      = None
        any_pass           = False
        pass_count         = 0

        for sample in output.outputs:
            response_text = sample.text
            # Evaluate this solution
            try:
                reward_model = row.get("reward_model", {})
                if hasattr(reward_model, "tolist"): reward_model = reward_model.tolist()
                if isinstance(reward_model, str): reward_model = json.loads(reward_model)
                ground_truth = reward_model.get("ground_truth", "[]") if isinstance(reward_model, dict) else "[]"
                data_source = row.get("data_source", "mbpp")
                score = compute_score(data_source, response_text, ground_truth, extra_info)
                passed = (score > 0.5)
            except Exception as e:
                passed = False

            if passed:
                pass_count += 1
                if best_response_text is None:
                    best_response_text = response_text
                    # Get token IDs
                    encoded = tokenizer.encode(response_text, add_special_tokens=False)
                    best_response_ids = encoded
                    best_logprobs = sample.cumulative_logprob / max(len(encoded), 1)

        any_pass = pass_count > 0
        pass_ratio = pass_count / args.n_samples

        results.append({
            "prompt":              row["prompt"],
            "data_source":         row.get("data_source", "mbpp"),
            "reward_model":        row.get("reward_model", {}),
            "extra_info":          row.get("extra_info", {}),
            "teacher_response_ids": np.array(best_response_ids) if best_response_ids else None,
            "teacher_logprobs":    float(best_logprobs) if best_logprobs else None,
            "teacher_pass":        any_pass,
            "teacher_pass_ratio":  float(pass_ratio),
            "teacher_model":       "coder_1_5b_mbpp_grpo_best",
        })

        if (idx + 1) % 50 == 0:
            n_pass = sum(1 for r in results if r["teacher_pass"])
            print(f"[{idx+1}/{len(df)}] pass rate so far: {n_pass}/{idx+1} = {n_pass/(idx+1):.1%}")

    # Save
    out_df = pd.DataFrame(results)
    n_pass = out_df["teacher_pass"].sum()
    print(f"\nDone. {n_pass}/{len(out_df)} problems have passing solutions ({n_pass/len(out_df):.1%})")
    out_df.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
