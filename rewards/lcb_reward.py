# rewards/lcb_reward.py
"""
LCB/APPS stdin/stdout 代码执行奖励函数，通过 veRL custom_reward_function 接入。

【和 mbpp_reward.py 的核心区别】

mbpp_reward.py（assert 风格）：
  exec("def func(x): return x+1\nassert func(1)==2")
  → 在同一进程内 exec，assert 失败抛 AssertionError → reward=0
  → 简单，但只适用于函数级别的断言测试

lcb_reward.py（stdin/stdout 风格）：
  subprocess.run(["python3", tmp_file], input=test_input, capture_output=True)
  → 独立子进程，比对 stdout 和期望输出
  → 适用于竞赛题：模型需要写完整的读入/输出程序

【为什么训练时不用 run_in_sandbox（multiprocessing 版本）？】
  mbpp_reward.py 用 multiprocessing.Process 是因为测试用例是 Python exec 执行，
  需要在子进程里运行避免影响主进程。
  lcb_reward.py 用 subprocess，天然是独立进程，不需要额外包装。

函数签名符合 veRL NaiveRewardManager 要求：
    compute_score(data_source, solution_str, ground_truth, extra_info) -> float
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path


def extract_code(solution_str: str) -> str:
    """从模型输出中提取 Python 代码块（与 mbpp_reward.py 逻辑相同）。"""
    import re
    text = re.sub(r"<think>.*?</think>", "", solution_str, flags=re.DOTALL)
    text = text.strip()
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def run_stdin_stdout(code: str, test_input: str, timeout: int = 10) -> tuple:
    """
    用 subprocess 执行代码，返回 (成功执行, 实际stdout)。
    成功执行不等于输出正确——后者需要 compare_output 判断。
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
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"ERROR: {e}"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def compare_output(actual: str, expected: str) -> bool:
    """三层输出比对：精确 → 逐行去末尾空格 → 浮点近似。"""
    actual = actual.strip()
    expected = expected.strip()
    if actual == expected:
        return True
    actual_lines = [l.rstrip() for l in actual.splitlines()]
    expected_lines = [l.rstrip() for l in expected.splitlines()]
    if actual_lines == expected_lines:
        return True
    if len(actual_lines) == len(expected_lines):
        try:
            for a_line, e_line in zip(actual_lines, expected_lines):
                a_vals = [float(x) for x in a_line.split()]
                e_vals = [float(x) for x in e_line.split()]
                if len(a_vals) != len(e_vals):
                    return False
                for av, ev in zip(a_vals, e_vals):
                    if abs(av - ev) > max(1e-6, 1e-6 * abs(ev)):
                        return False
            return True
        except (ValueError, ZeroDivisionError):
            pass
    return False


def compute_score_binary(code: str, test_list: list) -> float:
    """全部测试用例通过=1.0，否则=0.0。"""
    for t in test_list:
        success, actual = run_stdin_stdout(code, t["input"])
        if not success or not compare_output(actual, t["output"]):
            return 0.0
    return 1.0


def compute_score_partial(code: str, test_list: list) -> float:
    """
    按通过比例给部分 reward。
    竞赛题的测试用例往往有单项通过的意义（如边界条件），
    partial reward 让模型有"至少通过简单用例"的梯度信号。

    映射规则（和 mbpp 保持一致的设计哲学）：
      0/n → 0.0
      1..n-1 → 线性 [0.2, 0.6]
      n/n → 1.0
    """
    n = len(test_list)
    if n == 0:
        return 0.0
    passed = 0
    for t in test_list:
        success, actual = run_stdin_stdout(code, t["input"])
        if success and compare_output(actual, t["output"]):
            passed += 1
    if passed == 0:
        return 0.0
    if passed == n:
        return 1.0
    # 部分通过：线性映射到 [0.2, 0.6]
    return 0.2 + 0.4 * (passed - 1) / (n - 1)


def compute_score(data_source: str, solution_str: str, ground_truth: str,
                  extra_info: dict = None) -> float:
    """
    veRL custom_reward_function 入口。
      data_source: "apps" 或 "lcb"（由 parquet 的 data_source 字段决定）
      solution_str: 模型生成的代码（含 ```python...``` 包装）
      ground_truth: JSON 序列化的 test_list，格式 [{"input": "...", "output": "..."}, ...]
      extra_info: 可选
    """
    if data_source not in ("apps", "lcb"):
        raise ValueError(f"lcb_reward.py: unexpected data_source={data_source!r}")

    code = extract_code(solution_str)
    if not code.strip():
        return 0.0

    test_list = json.loads(ground_truth)
    if not test_list:
        return 0.0

    reward_mode = os.environ.get("REWARD_MODE", "binary")
    if reward_mode == "binary":
        return compute_score_binary(code, test_list)
    elif reward_mode == "partial":
        return compute_score_partial(code, test_list)
    else:
        raise ValueError(f"Unknown REWARD_MODE={reward_mode!r}")
