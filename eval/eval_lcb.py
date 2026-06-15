"""
LiveCodeBench pass@k 评测脚本
==============================

【eval harness 是什么？】
"eval harness"（评测框架）指的是：把模型生成的代码字符串变成 pass/fail 信号的全套机制。
主要解决以下问题：
  1. 代码提取：从 ```python...``` 块里把代码取出来
  2. 代码执行：安全、有超时地运行代码
  3. 输出比对：判断输出是否"正确"（字符串匹配 or 数值近似）
  4. 错误处理：运行时错误、超时、内存溢出

【为什么 LCB 的 harness 比 MBPP 复杂？】

MBPP 的执行模型（assert 风格）：
  ```
  exec("def my_func(x): return x+1\nassert my_func(1)==2")
  ```
  → 在同一个 Python 进程里 exec，异常=失败，无异常=通过
  → 简单，但：不支持 stdin/stdout，不能处理无限循环（需要超时）

LCB/竞赛题的执行模型（stdin/stdout 风格）：
  ```
  输入: "5\n3 1 4 1 5\n"   （题目输入，多行）
  期望输出: "3\n"           （OJ 期望的标准答案）
  实际执行: subprocess.run(["python3", ...], input="5\n3 1 4 1 5\n")
  比对: actual_stdout.strip() == expected.strip()
  ```
  → 需要用 subprocess 启动独立进程（不是 exec！）
  → 为什么用 subprocess 而不是 exec？
      - exec 在同一进程，无法捕获 stdin/stdout
      - exec 的超时控制复杂（无法强制 kill）
      - subprocess 天然隔离：子进程崩溃不影响父进程

【输出比对的细节——为什么不能直接 == ？】
竞赛题输出比对有几个常见坑：
  1. 末尾换行：expected="3\n", actual="3" → strip() 后相同，应该通过
  2. 末尾空格：" 3 " vs "3" → 通常要去掉行末空格
  3. 浮点数：expected="0.333333" vs actual="0.333333333" → 需要近似比较
  4. 大小写：通常大小写敏感，除非题目说明不区分
本脚本的策略：先做 strip() 的精确比对，再做浮点数近似比对（tolerance=1e-6）。

用法：
  python eval/eval_lcb.py \
      --model_path models/Qwen2.5-Coder-1.5B-Instruct \
      --test_file data/lcb/lcb_v5_stdin_stdout.parquet \
      --output_file <repo>/eval_results/lcb_coder_1_5b_instruct.json \
      --n_samples 5 \
      --temperature 0.8 \
      --label coder_1_5b_instruct
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ──────────────────────────────────────────────────────────────────────
# 代码提取
# ──────────────────────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    """
    从模型输出中提取 Python 代码块。
    与 MBPP 版本相同逻辑，但竞赛题模型通常不会有 <think> 块（用的是 instruct 模型）。
    """
    import re
    # 去掉 thinking 块（Qwen3 等推理模型会有）
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    # 提取 ```python ... ``` 或 ``` ... ```
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # 如果没有代码块标记，返回整个文本（部分模型直接输出代码）
    return text.strip()


# ──────────────────────────────────────────────────────────────────────
# stdin/stdout 执行沙箱
# ──────────────────────────────────────────────────────────────────────

def run_stdin_stdout(code: str, test_input: str, timeout: int = 10) -> tuple[bool, str]:
    """
    用 subprocess 执行代码，传入 stdin，捕获 stdout，返回 (是否通过, 实际输出)。

    【为什么把代码写到临时文件而不是 python -c "code"？】
    python -c 有个问题：代码里如果有单引号/双引号，shell 转义会很麻烦。
    写到临时文件，用 python <tmp_dir>/xxx.py 执行，完全没有转义问题。

    【超时机制】
    subprocess.run(..., timeout=10) 会在 10s 后抛 TimeoutExpired 异常，
    并自动 kill 子进程。这是比 MBPP 里用 multiprocessing 更简洁的方式。

    【安全性限制】
    这里没有像 MBPP 那样设 setrlimit（内存限制），
    因为 subprocess 本身就是独立进程，崩溃不影响父进程。
    对于评测（而非训练），这个安全级别足够了。
    如果要更严格的沙箱（防止读写文件等），需要 Docker 或 seccomp，这里不做。
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        actual_output = result.stdout
        return True, actual_output  # 成功执行（返回码可能非0，但至少跑完了）
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"ERROR: {e}"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def compare_output(actual: str, expected: str) -> bool:
    """
    比对实际输出和期望输出。

    策略（从宽到严）：
    1. 精确比对（strip 后）：绝大多数情况适用
    2. 逐行比对（去行末空格）：处理中间行有多余空格的情况
    3. 浮点数近似比对：如果每一行都能解析为浮点数，用 1e-6 容差比对

    【为什么要做浮点数比对？】
    有些题答案是浮点数，如"输出最短路径长度（保留6位小数）"。
    expected="1.234567", actual="1.23456700001" → 用 == 会失败，需要近似比对。
    """
    actual = actual.strip()
    expected = expected.strip()

    # 1. 精确比对
    if actual == expected:
        return True

    # 2. 逐行比对（去每行末尾空格）
    actual_lines = [l.rstrip() for l in actual.splitlines()]
    expected_lines = [l.rstrip() for l in expected.splitlines()]
    if actual_lines == expected_lines:
        return True

    # 3. 浮点数近似比对
    if len(actual_lines) == len(expected_lines):
        try:
            all_float = True
            for a_line, e_line in zip(actual_lines, expected_lines):
                a_vals = [float(x) for x in a_line.split()]
                e_vals = [float(x) for x in e_line.split()]
                if len(a_vals) != len(e_vals):
                    all_float = False
                    break
                for av, ev in zip(a_vals, e_vals):
                    if abs(av - ev) > max(1e-6, 1e-6 * abs(ev)):
                        all_float = False
                        break
                if not all_float:
                    break
            if all_float:
                return True
        except (ValueError, ZeroDivisionError):
            pass

    return False


def run_test_case(code: str, test_input: str, test_output: str, timeout: int = 10) -> bool:
    """
    对单个测试用例运行代码，返回是否通过。
    """
    success, actual_output = run_stdin_stdout(code, test_input, timeout)
    if not success:
        return False
    return compare_output(actual_output, test_output)


# ──────────────────────────────────────────────────────────────────────
# pass@k 计算
# ──────────────────────────────────────────────────────────────────────

def pass_at_k(n: int, c: int, k: int) -> float:
    """无偏估计：pass@k = 1 - C(n-c, k) / C(n, k)"""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ──────────────────────────────────────────────────────────────────────
# 主评测逻辑
# ──────────────────────────────────────────────────────────────────────

def evaluate(model_path: str, test_file: str, output_file: str,
             n_samples: int, temperature: float, label: str,
             max_problems: int = None):
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    df = pd.read_parquet(test_file)
    if max_problems:
        df = df.head(max_problems)
        print(f"[DEBUG] Using first {max_problems} problems only")
    print(f"Test set: {len(df)} problems")

    # 按难度分组打印
    if "difficulty" in df.columns:
        print("Difficulty distribution:", df["difficulty"].value_counts().to_dict())

    # 加载模型（vLLM）
    # 【为什么用 vLLM 而不是 transformers generate？】
    # vLLM 实现了 PagedAttention，对于 n_samples 个并行采样（如 n=20），
    # 比 transformers 的朴素实现快 5-10x。对于 400 题 × 5 样本，速度差异显著。
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=4096,   # LCB 题比 MBPP 长得多，需要更大的 context
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 【max_tokens 设为 2048 而不是 1024】
    # 竞赛题的解法代码通常比 MBPP 长（需要处理输入、主逻辑、输出），
    # 1024 tokens 很容易截断导致代码不完整。
    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        max_tokens=2048,
    )

    # 构造 prompts
    raw_prompts = df["prompt"].tolist()
    prompts = []
    for raw in raw_prompts:
        messages = json.loads(raw) if isinstance(raw, str) else raw
        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    print(f"Generating {n_samples} samples per problem (total {len(prompts)*n_samples} generations)...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    pass1_list = []
    pass5_list = []

    # 难度分组统计
    by_difficulty = {}

    for i, (output, row) in enumerate(zip(outputs, df.itertuples())):
        tests = json.loads(row.tests)
        question_id = row.question_id
        difficulty = getattr(row, "difficulty", "Unknown")

        c = 0  # 通过所有测试用例的样本数
        per_sample = []

        for sample_output in output.outputs:
            code = extract_code(sample_output.text)
            # 对每个测试用例都要通过
            passed_all = all(
                run_test_case(code, t["input"], t["output"])
                for t in tests
            )
            if passed_all:
                c += 1
            per_sample.append(passed_all)

        p1 = pass_at_k(n_samples, c, 1)
        p5 = pass_at_k(n_samples, c, min(5, n_samples))
        pass1_list.append(p1)
        pass5_list.append(p5)

        # 按难度分组
        if difficulty not in by_difficulty:
            by_difficulty[difficulty] = []
        by_difficulty[difficulty].append(p1)

        results.append({
            "question_id": question_id,
            "title": getattr(row, "title", ""),
            "difficulty": difficulty,
            "platform": getattr(row, "platform", ""),
            "c": c,
            "n": n_samples,
            "pass@1": p1,
            "pass@5": p5,
            "per_sample": per_sample,
        })

        if (i + 1) % 20 == 0:
            running_avg = sum(pass1_list) / len(pass1_list)
            print(f"  [{i+1}/{len(df)}] running pass@1 = {running_avg:.3f}")

    avg_pass1 = sum(pass1_list) / len(pass1_list)
    avg_pass5 = sum(pass5_list) / len(pass5_list)

    # 按难度分组统计
    difficulty_stats = {
        diff: sum(vals) / len(vals)
        for diff, vals in by_difficulty.items()
        if vals
    }

    summary = {
        "label": label,
        "model_path": model_path,
        "n_samples": n_samples,
        "temperature": temperature,
        "num_problems": len(df),
        "pass@1": avg_pass1,
        "pass@5": avg_pass5,
        "by_difficulty": difficulty_stats,
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({"summary": summary, "per_problem": results}, f, indent=2)

    print(f"\nResults for {label}:")
    print(f"  pass@1 = {avg_pass1:.4f} ({avg_pass1*100:.1f}%)")
    print(f"  pass@5 = {avg_pass5:.4f} ({avg_pass5*100:.1f}%)")
    print(f"  By difficulty: " + " | ".join(
        f"{d}: {v*100:.1f}%" for d, v in sorted(difficulty_stats.items())
    ))
    print(f"  Saved to {output_file}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--n_samples", type=int, default=5,
                        help="LCB 题目比 MBPP 更难更长，n=5 已足够估计 pass@1，减少计算量")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max_problems", type=int, default=None,
                        help="调试时用，只跑前 N 道题")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        test_file=args.test_file,
        output_file=args.output_file,
        n_samples=args.n_samples,
        temperature=args.temperature,
        label=args.label,
        max_problems=args.max_problems,
    )
