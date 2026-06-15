"""
run_sft_teacher_warmup.py

在 teacher_pass=True 的样本上对 partial_v2_step80 模型做 SFT warmup。
目标：让 1.7B 学生从 7B 教师的 greedy 解法中学习，改善 GRPO 的起点。

用法：
    python scripts/run_sft_teacher_warmup.py \
        --base_model models/eval_merged/partial_v2_final_step80 \
        --teacher_data data/mbpp_v2/mbpp_train_with_teacher.parquet \
        --output_dir models/opd_sft_warmup \
        --num_train_epochs 2 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-5 \
        --max_seq_length 2048
"""

import argparse
import os
import json
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from torch.utils.data import Dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",  default=os.path.expanduser("models/eval_merged/partial_v2_final_step80"))
    p.add_argument("--teacher_data", default=os.path.expanduser("data/mbpp_v2/mbpp_train_with_teacher.parquet"))
    p.add_argument("--output_dir",  default=os.path.expanduser("models/opd_sft_warmup"))
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=5)
    return p.parse_args()


class TeacherSFTDataset(Dataset):
    """Dataset of (prompt, teacher_response) pairs for SFT training."""

    def __init__(self, df, tokenizer, max_seq_length=2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []

        # Filter: only teacher_pass=True samples
        df_filtered = df[df["teacher_pass"] == True].copy()
        print(f"Using {len(df_filtered)} teacher_pass=True samples out of {len(df)} total")

        for _, row in df_filtered.iterrows():
            # Get prompt messages
            prompt = row["prompt"]
            if isinstance(prompt, np.ndarray):
                prompt = prompt.tolist()
            messages = list(prompt) if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]

            # Get teacher response ids
            teacher_ids = row["teacher_response_ids"]
            if isinstance(teacher_ids, np.ndarray):
                teacher_ids = teacher_ids.tolist()
            if teacher_ids is None:
                continue

            self.samples.append((messages, teacher_ids))

        print(f"Final dataset size: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages, teacher_response_ids = self.samples[idx]

        # Tokenize prompt (without generation prompt first, then with)
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )[0]  # [prompt_len]

        # Append EOS token after teacher response
        eos = self.tokenizer.eos_token_id
        response_ids = torch.tensor(teacher_response_ids + [eos], dtype=torch.long)

        # Concatenate
        input_ids = torch.cat([prompt_ids, response_ids])

        # Labels: -100 for prompt tokens (don't compute loss), actual ids for response
        labels = torch.cat([
            torch.full((len(prompt_ids),), -100, dtype=torch.long),
            response_ids,
        ])

        # Truncate to max_seq_length
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }


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
    model.config.use_cache = False  # required for gradient checkpointing

    print(f"Loading teacher data from {args.teacher_data}")
    df = pd.read_parquet(args.teacher_data)
    print(f"teacher_pass distribution:\n{df['teacher_pass'].value_counts(dropna=False)}")

    train_dataset = TeacherSFTDataset(df, tokenizer, max_seq_length=args.max_seq_length)

    if len(train_dataset) == 0:
        print("ERROR: No teacher_pass=True samples found! Check the data file.")
        return

    # Data collator pads to longest in batch
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

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

    print(f"\nStarting SFT warmup training...")
    print(f"  base model: {args.base_model}")
    print(f"  samples: {len(train_dataset)}")
    print(f"  epochs: {args.num_train_epochs}")
    print(f"  batch size: {args.per_device_train_batch_size} * {args.gradient_accumulation_steps} = {args.per_device_train_batch_size * args.gradient_accumulation_steps} effective")
    print(f"  output: {args.output_dir}")

    trainer.train()

    print(f"\nSaving final model to {args.output_dir}/final")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    print("Done!")


if __name__ == "__main__":
    main()
