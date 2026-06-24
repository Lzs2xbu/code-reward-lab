"""
prepare_apps_teacher_sft.py

用 7B teacher 模型对 APPS interview 题目生成解法，过滤出 teacher_pass=True 的样本，
输出为 SFT 训练数据。

用法：
    python scripts/prepare_apps_teacher_sft.py \
        --teacher_model models/Qwen2.5-Coder-7B-Instruct \
        --apps_data    data/apps/apps_rl_interview.parquet \
        --output_file  data/apps/apps_teacher_sft.parquet \
        --n_samples 4 \
        --temperature 0.8 \
        --max_tokens 2048

逻辑：
    1. 对每道题生成 n_samples 个解法
    2. 逐一执行测试，取第一个通过所有测试的解法作为 SFT 样本
    3. 保存 teacher_pass=True 的行；同时输出统计（各难度通过率）
    4. 支持断点续跑（--checkpoint_file）
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd


# ─── 参数解析 ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_model", default=os.path.expanduser(
        "models/Qwen2.5-Coder-7B-Instruct"))
    p.add_argument("--apps_data", default=os.path.expanduser(
        "data/apps/apps_rl_interview.parquet"))
    p.add_argument("--output_file", default=os.path.expanduser(
        "data/apps/apps_teacher_sft.parquet"))
    p.add_argument("--checkpoint_file", default=os.path.expanduser(
        "data/apps/apps_teacher_sft_checkpoint.jsonl"))
    p.add_argument("--n_samples", type=int, default=4,
                   help="每道题生成 n 个候选，取第一个通过测试的")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--timeout", type=int, default=10,
                   help="每个测试用例的执行超时秒数")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    return p.parse_args()


# ─── 代码提取 ─────────────────────────────────────────────────────────────────

def extract_code(solution_str: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", solution_str, flags=re.DOTALL)
    text = text.strip()
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


# ─── 代码执行 & 输出比对 ──────────────────────────────────────────────────────

def run_code(code: str, test_input: str, timeout: int) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        r = subprocess.run(
            ["python3", tmp],
            input=test_input,
            capture_output=True, text=True, timeout=timeout,
        )
        return True, r.stdout
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"ERROR:{e}"
    finally:
        Path(tmp).unlink(missing_ok=True)


def compare_output(actual: str, expected: str) -> bool:
    actual, expected = actual.strip(), expected.strip()
    if actual == expected:
        return True
    al = [l.rstrip() for l in actual.splitlines()]
    el = [l.rstrip() for l in expected.splitlines()]
    if al == el:
        return True
    if len(al) == len(el):
        try:
            return all(
                abs(float(a) - float(e)) <= max(1e-6, 1e-6 * abs(float(e)))
                for a, e in zip(al, el)
            )
        except (ValueError, ZeroDivisionError):
            pass
    return False


def eval_solution(code_str: str, ground_truth_json: str, timeout: int) -> bool:
    """返回该解法是否通过 ground_truth 中的所有测试用例。"""
    try:
        test_cases = json.loads(ground_truth_json)
    except Exception:
        return False
    code = extract_code(code_str)
    if not code:
        return False
    for tc in test_cases:
        ok, actual = run_code(code, tc["input"], timeout)
        if not ok or not compare_output(actual, tc["output"]):
            return False
    return True


# ─── 断点续跑 ─────────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_file: str) -> dict:
    """返回 {question_id -> result_dict} 的已完成结果。"""
    done = {}
    p = Path(checkpoint_file)
    if p.exists():
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    done[obj["question_id"]] = obj
        print(f"[checkpoint] Loaded {len(done)} previously completed problems")
    return done


def save_checkpoint(checkpoint_file: str, result: dict):
    with open(checkpoint_file, "a") as f:
        f.write(json.dumps(result) + "\n")


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"Loading APPS data from {args.apps_data}")
    df = pd.read_parquet(args.apps_data)
    print(f"  {len(df)} problems")

    # 提取 extra_info 中的 question_id 和 difficulty
    df["question_id"] = df["extra_info"].apply(lambda x: x["question_id"])
    df["difficulty"]  = df["extra_info"].apply(lambda x: x["difficulty"])
    df["ground_truth"] = df["reward_model"].apply(lambda x: x["ground_truth"])

    # 断点续跑
    done = load_checkpoint(args.checkpoint_file)
    remaining = df[~df["question_id"].isin(done.keys())]
    print(f"  Remaining to process: {len(remaining)}")

    if len(remaining) == 0:
        print("All problems already processed, skipping inference.")
    else:
        # vLLM 推理
        print(f"\nLoading vLLM with {args.teacher_model} ...")
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.teacher_model,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=4096,
        )
        tokenizer = llm.get_tokenizer()
        sampling = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n=args.n_samples,
        )

        # 构建 prompt 列表（chat template）
        prompts_text = []
        for _, row in remaining.iterrows():
            msgs = row["prompt"]
            if isinstance(msgs, (list, tuple)):
                msgs = list(msgs)
            text = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
            prompts_text.append(text)

        print(f"Running vLLM inference on {len(prompts_text)} problems × n={args.n_samples} ...")
        t0 = time.time()
        outputs = llm.generate(prompts_text, sampling)
        print(f"  Inference done in {time.time()-t0:.1f}s")

        # 释放 GPU（评测在 CPU subprocess 里）
        del llm
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 评测每道题
        print("\nEvaluating solutions against test cases ...")
        for i, (_, row) in enumerate(remaining.iterrows()):
            qid = row["question_id"]
            gt  = row["ground_truth"]
            candidates = [o.text for o in outputs[i].outputs]  # n_samples 个候选

            teacher_pass = False
            best_response = ""
            for cand in candidates:
                if eval_solution(cand, gt, args.timeout):
                    teacher_pass = True
                    best_response = cand
                    break

            result = {
                "question_id": qid,
                "difficulty":  row["difficulty"],
                "teacher_pass": teacher_pass,
                "response": best_response,
                "prompt": json.dumps(
                    list(row["prompt"]) if not isinstance(row["prompt"], str)
                    else row["prompt"]
                ),
            }
            save_checkpoint(args.checkpoint_file, result)
            done[qid] = result

            if (i + 1) % 50 == 0:
                n_pass = sum(1 for v in done.values() if v["teacher_pass"])
                print(f"  [{i+1}/{len(remaining)}] teacher_pass so far: {n_pass}/{len(done)}")

    # ─── 汇总并保存 ────────────────────────────────────────────────────────────
    records = list(done.values())
    df_out = pd.DataFrame(records)

    n_pass = df_out["teacher_pass"].sum()
    n_total = len(df_out)
    print(f"\n=== Results ===")
    print(f"Total problems:   {n_total}")
    print(f"teacher_pass=True: {n_pass}  ({n_pass/n_total:.1%})")
    print(f"\nBy difficulty:")
    for diff, grp in df_out.groupby("difficulty"):
        p = grp["teacher_pass"].sum()
        print(f"  {diff}: {p}/{len(grp)} ({p/len(grp):.1%})")

    # 只保留 teacher_pass=True 的行（prompt 已经是 JSON string）
    df_sft = df_out[df_out["teacher_pass"] == True][
        ["question_id", "difficulty", "prompt", "response"]
    ].reset_index(drop=True)
    df_sft.to_parquet(args.output_file, index=False)
    print(f"\nSaved {len(df_sft)} teacher_pass samples to {args.output_file}")


if __name__ == "__main__":
    main()
