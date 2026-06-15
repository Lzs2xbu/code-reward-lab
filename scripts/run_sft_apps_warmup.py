"""
APPS SFT Warmup 脚本
====================
对 Coder-1.5B 用 APPS 人工解法做 SFT warmup，作为后续 GRPO/OPD 训练的起点。

和 run_sft_teacher_warmup.py 的区别：
  原脚本：使用 teacher_response_ids（token ID 列表），专为 MBPP teacher 数据设计
  本脚本：使用 response 文本字段，适用于 APPS 的纯文本解法

用法（在服务器上运行）：
  python <repo>/scripts/run_sft_apps_warmup.py \
      --base_model models/Qwen2.5-Coder-1.5B-Instruct \
      --sft_data data/apps/apps_sft_interview.parquet \
      --output_dir models/coder_1_5b_apps_sft_warmup \
      --num_train_epochs 2
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default=os.path.expanduser(
        "models/Qwen2.5-Coder-1.5B-Instruct"))
    p.add_argument("--sft_data", default=os.path.expanduser(
        "data/apps/apps_sft_interview.parquet"))
    p.add_argument("--output_dir", default=os.path.expanduser(
        "models/coder_1_5b_apps_sft_warmup"))
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8,
                   help="effective_batch = 2*8 = 16，和 MBPP SFT 一致")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--max_seq_length", type=int, default=3072,
                   help="APPS 题目 + 解法比 MBPP 长，用 3072")
    p.add_argument("--warmup_steps", type=int, default=30)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=10)
    return p.parse_args()


class AppsSFTDataset(Dataset):
    """
    APPS SFT 数据集。

    【SFT 的训练目标是什么？】
    给定 prompt（题目描述），预测 response（解法代码）。
    在 transformer 的 loss 计算里：
      - prompt token 的 label 设为 -100（不计算 loss，只作为 context）
      - response token 的 label 设为实际 token ID（计算 cross-entropy loss）
    这样模型只学"如何生成解法"，不学"如何重复题目"。

    【为什么不直接用 transformers 的 DataCollatorForLanguageModeling？】
    DataCollatorForLanguageModeling 对整个序列（包括 prompt）都计算 loss，
    SFT 时需要用 DataCollatorForSeq2Seq 配合 label=-100 的 mask。
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_seq_length: int = 3072):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []
        skipped = 0

        for _, row in df.iterrows():
            # prompt 是 JSON 序列化的 messages list
            messages = json.loads(row["prompt"]) if isinstance(row["prompt"], str) else row["prompt"]
            response_text = row["response"]

            # 用 tokenizer 的 chat template 拼 prompt
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )

            # tokenize response（不加 special tokens，避免重复 BOS）
            response_ids = tokenizer.encode(
                response_text,
                add_special_tokens=False,
            )
            # 加上 EOS，让模型学会何时停止生成
            response_ids = response_ids + [tokenizer.eos_token_id]

            total_len = len(prompt_ids) + len(response_ids)
            if total_len > max_seq_length:
                # 过长的样本截断 response（保留 prompt 完整）
                max_response_len = max_seq_length - len(prompt_ids)
                if max_response_len < 50:
                    # prompt 本身就太长，跳过
                    skipped += 1
                    continue
                response_ids = response_ids[:max_response_len]

            input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.long)
            labels = torch.tensor(
                [-100] * len(prompt_ids) + response_ids,
                dtype=torch.long,
            )
            self.samples.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            })

        print(f"  Dataset: {len(self.samples)} samples (skipped {skipped} too-long)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    args = parse_args()

    print(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    print(f"Loading SFT data from {args.sft_data}")
    df = pd.read_parquet(args.sft_data)
    print(f"  {len(df)} samples, difficulty: {df['difficulty'].value_counts().to_dict()}")

    train_dataset = AppsSFTDataset(df, tokenizer, max_seq_length=args.max_seq_length)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    total_steps = (len(train_dataset) // effective_batch) * args.num_train_epochs
    print(f"\n  Effective batch size: {effective_batch}")
    print(f"  Estimated total steps: {total_steps}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print(f"\nStarting APPS SFT warmup...")
    trainer.train()

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nSaved to {final_dir}")


if __name__ == "__main__":
    main()
