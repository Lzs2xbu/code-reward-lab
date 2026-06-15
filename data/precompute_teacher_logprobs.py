"""
precompute_teacher_logprobs.py

对 MBPP 训练集的每个 prompt，用教师模型（7B）生成一条 greedy 响应，
并计算教师模型在该响应上的 token-level log-probs。

输出: parquet 文件，新增列：
  - teacher_response_ids   : List[int]   — 教师生成的 response token ids
  - teacher_logprobs       : List[float] — 每个 response token 的 log-prob（来自教师）
  - teacher_pass           : bool        — 教师响应是否通过所有测试（由外部 eval 填充）

用法（CPU，GPU 被占用时）:
  python precompute_teacher_logprobs.py \
      --teacher_model models/Qwen2.5-Coder-7B-Instruct \
      --train_file data/mbpp_v2/mbpp_train.parquet \
      --output_file data/mbpp_v2/mbpp_train_with_teacher.parquet \
      --device cpu --dtype float16 --batch_size 1

用法（GPU 空闲时，更快）:
  python precompute_teacher_logprobs.py \
      --teacher_model models/Qwen2.5-Coder-7B-Instruct \
      --train_file data/mbpp_v2/mbpp_train.parquet \
      --output_file data/mbpp_v2/mbpp_train_with_teacher.parquet \
      --device cuda --dtype bfloat16 --batch_size 4

注意：
  - CPU + float16 约需 14GB RAM，372 条样本估计 2-4 小时
  - 支持断点续跑：已有输出文件时自动跳过已处理的行
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_model", default=os.path.expanduser("models/Qwen2.5-Coder-7B-Instruct"))
    p.add_argument("--train_file",    default=os.path.expanduser("data/mbpp_v2/mbpp_train.parquet"))
    p.add_argument("--output_file",   default=os.path.expanduser("data/mbpp_v2/mbpp_train_with_teacher.parquet"))
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--device",  default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype",   default="float16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size for generation. CPU: keep 1; GPU: 4-8 safe.")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip already-processed rows if output file exists (default: True)")
    return p.parse_args()


def extract_func_name(ground_truth_str: str) -> str | None:
    """从 ground_truth JSON 中提取被测函数名（用于执行验证）"""
    try:
        tests = json.loads(ground_truth_str)
        for t in tests:
            m = re.match(r'assert\s+(\w+)\s*\(', t)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


def run_tests(code: str, tests: list[str]) -> tuple[int, int]:
    """在隔离子进程中执行代码 + 测试，返回 (passed, total)"""
    script = code + "\n" + "\n".join(
        f"try:\n    {t}\nexcept Exception:\n    pass" for t in tests
    )
    passed = 0
    for test in tests:
        full = code + f"\ntry:\n    {test}\n    print('OK')\nexcept Exception as e:\n    print('FAIL', e)\n"
        try:
            result = subprocess.run(
                [sys.executable, "-c", full],
                capture_output=True, text=True, timeout=5
            )
            if "OK" in result.stdout:
                passed += 1
        except subprocess.TimeoutExpired:
            pass
    return passed, len(tests)


def get_prompt_messages(row) -> list[dict]:
    """从 parquet row 中提取 chat messages"""
    prompt = row["prompt"]
    if isinstance(prompt, list):
        # 已经是 message list
        return list(prompt)
    if isinstance(prompt, np.ndarray):
        return list(prompt.tolist())
    # 字符串 fallback
    return [{"role": "user", "content": str(prompt)}]


def compute_logprobs_for_response(
    model, input_ids: torch.Tensor, response_ids: torch.Tensor
) -> list[float]:
    """
    给定 prompt input_ids 和 response_ids，
    计算教师模型在 response 上每个 token 的 log-prob。

    input_ids:   [1, prompt_len]
    response_ids:[resp_len]
    返回: [resp_len] 的 float list
    """
    full_ids = torch.cat([input_ids[0], response_ids], dim=0).unsqueeze(0)  # [1, total_len]
    with torch.no_grad():
        logits = model(full_ids).logits  # [1, total_len, vocab]

    prompt_len = input_ids.shape[1]
    # response 对应的 logits：位置 prompt_len-1 到 total_len-2（shift by 1）
    resp_logits = logits[0, prompt_len - 1: prompt_len - 1 + len(response_ids), :]  # [resp_len, vocab]
    log_probs = F.log_softmax(resp_logits.float(), dim=-1)  # cast to float32 for stability
    token_lp = log_probs[torch.arange(len(response_ids)), response_ids].cpu().tolist()
    return token_lp


def main():
    args = parse_args()
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # ---------- 加载训练数据 ----------
    df = pd.read_parquet(args.train_file)
    print(f"Loaded {len(df)} training samples from {args.train_file}")

    # ---------- 断点续跑：加载已有输出 ----------
    already_done = set()
    if args.resume and Path(args.output_file).exists():
        done_df = pd.read_parquet(args.output_file)
        if "teacher_response_ids" in done_df.columns:
            already_done = set(done_df.index[done_df["teacher_response_ids"].notna()])
            print(f"Resuming: {len(already_done)} rows already processed, skipping them.")
            # 用已有结果初始化输出列
            for col in ["teacher_response_ids", "teacher_logprobs", "teacher_pass"]:
                if col in done_df.columns:
                    df[col] = done_df[col]

    # 初始化输出列（若不存在）
    for col in ["teacher_response_ids", "teacher_logprobs", "teacher_pass"]:
        if col not in df.columns:
            df[col] = None

    # ---------- 加载教师模型 ----------
    print(f"Loading teacher model from {args.teacher_model}")
    print(f"  device={args.device}, dtype={args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

    # ---------- 逐条处理 ----------
    to_process = [i for i in df.index if i not in already_done]
    print(f"Processing {len(to_process)} samples...")

    save_every = 20  # 每 20 条保存一次（断点续跑粒度）

    for count, idx in enumerate(tqdm(to_process)):
        row = df.loc[idx]
        messages = get_prompt_messages(row)

        # --- 构建 input_ids ---
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(args.device)  # [1, prompt_len]

        # --- 教师 greedy 生成 ---
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )  # [1, prompt_len + resp_len]

        response_ids = output_ids[0, input_ids.shape[1]:]  # [resp_len]
        response_ids_cpu = response_ids.cpu()

        # --- 计算 log-probs ---
        token_logprobs = compute_logprobs_for_response(model, input_ids, response_ids)

        # --- 运行测试（可选，用于后续分析）---
        response_text = tokenizer.decode(response_ids_cpu, skip_special_tokens=True)
        try:
            # ground_truth 存在 reward_model['ground_truth'] 里，不是顶级列
            reward_model = row.get("reward_model", {})
            if isinstance(reward_model, dict):
                ground_truth = reward_model.get("ground_truth", "[]")
            else:
                ground_truth = "[]"
            if isinstance(ground_truth, np.ndarray):
                ground_truth = ground_truth.tolist()
            tests = json.loads(ground_truth) if isinstance(ground_truth, str) else list(ground_truth)
            passed, total = run_tests(response_text, tests)
            teacher_pass = (passed == total and total > 0)
        except Exception:
            teacher_pass = None

        # --- 存入 df ---
        df.at[idx, "teacher_response_ids"] = response_ids_cpu.tolist()
        df.at[idx, "teacher_logprobs"]     = token_logprobs
        df.at[idx, "teacher_pass"]         = teacher_pass

        # --- 定期保存 ---
        if (count + 1) % save_every == 0:
            df.to_parquet(args.output_file, index=True)
            passed_count = df["teacher_pass"].sum() if df["teacher_pass"].notna().any() else 0
            total_count  = df["teacher_pass"].notna().sum()
            print(f"\n[checkpoint] {count+1}/{len(to_process)} done | "
                  f"teacher pass rate so far: {passed_count}/{total_count} "
                  f"({100*passed_count/max(total_count,1):.1f}%)")

    # ---------- 最终保存 ----------
    df.to_parquet(args.output_file, index=True)

    # ---------- 统计 ----------
    total_done  = df["teacher_pass"].notna().sum()
    total_pass  = df["teacher_pass"].sum()
    print(f"\n=== Done ===")
    print(f"Output: {args.output_file}")
    print(f"Teacher pass rate: {total_pass}/{total_done} ({100*total_pass/max(total_done,1):.1f}%)")
    print(f"Avg response length: {df['teacher_response_ids'].dropna().apply(len).mean():.1f} tokens")


if __name__ == "__main__":
    main()
