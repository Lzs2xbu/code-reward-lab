"""
prepare_mbpp_opd_cl.py

在 teacher 预计算结果基础上，生成 OPD + Curriculum Learning 训练集。

输入：data/mbpp_v2/mbpp_train_with_teacher.parquet
      （precompute_teacher_logprobs.py 的输出，含 teacher_pass/teacher_response_ids/teacher_logprobs）

输出：
  data/mbpp_v2/mbpp_train_opd_cl.parquet     — OPD+CL 训练集（过滤 + 权重列）
  data/mbpp_v2/mbpp_train_opd_cl_stats.json  — 统计摘要

Curriculum Learning 策略：
  - teacher_pass == True  → "Medium"（7B能解，有正确解可蒸馏），保留，weight=2.0
  - teacher_pass == False → "Hard"（7B也解不了），保留但降权，weight=0.5
  - teacher_pass == None  → 预计算失败，按原始权重，weight=1.0

训练脚本可通过以下方式利用 cl_weight 列：
  1. 过滤模式：只保留 cl_weight >= 1.0 的样本（早期 epoch 更干净的信号）
  2. 加权模式：在 batch 采样时按 cl_weight 加权（更完整利用数据）

注意：蒸馏 loss 只对 teacher_pass==True 的样本有意义（只有这些样本有正确的教师解）。
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file",  default=os.path.expanduser("data/mbpp_v2/mbpp_train_with_teacher.parquet"))
    p.add_argument("--output_file", default=os.path.expanduser("data/mbpp_v2/mbpp_train_opd_cl.parquet"))
    p.add_argument("--stats_file",  default=os.path.expanduser("data/mbpp_v2/mbpp_train_opd_cl_stats.json"))
    p.add_argument("--medium_weight", type=float, default=2.0,  help="teacher_pass=True 样本的采样权重")
    p.add_argument("--hard_weight",   type=float, default=0.5,  help="teacher_pass=False 样本的采样权重")
    p.add_argument("--filter_hard",   action="store_true",       help="若设置，直接排除 teacher_pass=False 样本")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 加载 ──
    df = pd.read_parquet(args.input_file)
    print(f"Loaded {len(df)} samples from {args.input_file}")

    # ── 检查必要列 ──
    required = ["teacher_pass", "teacher_response_ids", "teacher_logprobs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"WARNING: Missing columns {missing}. Run precompute_teacher_logprobs.py first.")
        print("Proceeding without teacher data (OPD distillation will be unavailable).")

    # ── 统计 ──
    n_total = len(df)
    n_medium = (df["teacher_pass"] == True).sum()   if "teacher_pass" in df.columns else 0
    n_hard   = (df["teacher_pass"] == False).sum()  if "teacher_pass" in df.columns else 0
    n_unknown= df["teacher_pass"].isna().sum()       if "teacher_pass" in df.columns else n_total

    print(f"\nDifficulty distribution:")
    print(f"  Medium (teacher_pass=True):  {n_medium} ({100*n_medium/n_total:.1f}%)")
    print(f"  Hard   (teacher_pass=False): {n_hard}   ({100*n_hard/n_total:.1f}%)")
    print(f"  Unknown (not yet computed):  {n_unknown} ({100*n_unknown/n_total:.1f}%)")

    # ── 过滤 hard 样本（可选）──
    if args.filter_hard and "teacher_pass" in df.columns:
        df_filtered = df[df["teacher_pass"] != False].copy()
        print(f"\nFiltered out {n_hard} hard samples → {len(df_filtered)} remain")
        df = df_filtered

    # ── 添加 cl_weight 列 ──
    if "teacher_pass" in df.columns:
        df["cl_weight"] = df["teacher_pass"].map({
            True:  args.medium_weight,
            False: args.hard_weight,
        }).fillna(1.0).astype(float)
    else:
        df["cl_weight"] = 1.0

    # ── 添加 has_teacher_solution 列（方便训练时判断是否计算蒸馏 loss）──
    if "teacher_response_ids" in df.columns:
        df["has_teacher_solution"] = df["teacher_response_ids"].notna() & (df["teacher_pass"] == True)
    else:
        df["has_teacher_solution"] = False

    # ── 保存 ──
    df.to_parquet(args.output_file, index=False)
    print(f"\nSaved {len(df)} samples to {args.output_file}")

    # ── 统计摘要 ──
    stats = {
        "total_samples": len(df),
        "medium_samples": int((df.get("teacher_pass") == True).sum()),
        "hard_samples":   int((df.get("teacher_pass") == False).sum()),
        "has_teacher_solution": int(df.get("has_teacher_solution", pd.Series([False]*len(df))).sum()),
        "teacher_pass_rate": float(n_medium / n_total) if n_total > 0 else 0.0,
        "medium_weight": args.medium_weight,
        "hard_weight":   args.hard_weight,
        "filter_hard":   args.filter_hard,
    }
    Path(args.stats_file).write_text(json.dumps(stats, indent=2))
    print(f"\nStats saved to {args.stats_file}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
