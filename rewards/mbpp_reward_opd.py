"""
mbpp_reward_opd.py — OPD + CL 版本的 MBPP reward function

与 mbpp_reward.py 的区别：
1. 支持 teacher_pass 列（Curriculum Learning 过滤已在数据集层完成）
2. 保留 partial reward 逻辑不变
3. 蒸馏 loss（distillation loss）在 actor update 层实现（见 docs/opd_experiment_plan.md Phase 2）
   reward fn 本身不计算蒸馏项——这里只负责 execution reward

set 环境变量 REWARD_MODE=partial 启用 partial reward，否则用 binary。
"""

import json
import multiprocessing
import os
import re
import resource
import subprocess
import sys
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# 代码提取（兼容 Qwen3 的 <think> 标签和 markdown 代码块）
# ─────────────────────────────────────────────────────────────────────────────

def extract_code(solution_str: str) -> str:
    """从模型输出中提取纯 Python 代码。"""
    # Step 1：去掉 <think>...</think>
    code = re.sub(r"<think>.*?</think>", "", solution_str, flags=re.DOTALL)
    code = code.strip()
    # Step 2：提取 ```python ... ``` 或 ``` ... ```
    m = re.search(r"```(?:python)?\s*\n(.*?)```", code, flags=re.DOTALL)
    if m:
        code = m.group(1).strip()
    return code


# ─────────────────────────────────────────────────────────────────────────────
# 沙箱执行（子进程隔离，防止无限循环 / 内存炸弹）
# ─────────────────────────────────────────────────────────────────────────────

def _sandbox_worker(code: str, test_case: str, result_queue: multiprocessing.Queue):
    try:
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
        exec(compile(code + "\n" + test_case, "<string>", "exec"), {})
        result_queue.put(True)
    except Exception:
        result_queue.put(False)


def run_in_sandbox(code: str, test_case: str, timeout: int = 5) -> bool:
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_sandbox_worker, args=(code, test_case, q))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        p.join()
        return False
    return q.get() if not q.empty() else False


# ─────────────────────────────────────────────────────────────────────────────
# Partial reward 映射（与 mbpp_reward.py 保持一致）
# ─────────────────────────────────────────────────────────────────────────────

def partial_reward(passed: int, total: int) -> float:
    if total == 0:
        return 0.0
    ratio = passed / total
    if ratio == 0:
        return 0.0
    elif ratio < 0.5:
        return 0.2
    elif ratio < 1.0:
        return 0.6
    else:
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 主 reward 函数（veRL 调用接口）
# ─────────────────────────────────────────────────────────────────────────────

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
) -> float:
    """
    veRL custom_reward_function 接口。

    参数：
        data_source   : 数据集名称（mbpp_v2）
        solution_str  : 模型生成的完整文本
        ground_truth  : JSON string，包含 assert 语句列表
        extra_info    : batch 中的其他列（含 teacher_pass, teacher_response_ids 等）

    返回：
        float — execution reward（0.0 ~ 1.0）
        蒸馏 loss 由 actor update 层单独计算，不在此处返回。
    """
    reward_mode = os.environ.get("REWARD_MODE", "binary").lower()

    # ── 1. 解析测试用例 ──
    try:
        if isinstance(ground_truth, str):
            tests = json.loads(ground_truth)
        elif hasattr(ground_truth, "tolist"):
            tests = ground_truth.tolist()
        else:
            tests = list(ground_truth)
        if not tests:
            return 0.0
    except Exception:
        return 0.0

    # ── 2. 提取代码 ──
    code = extract_code(solution_str)
    if not code:
        return 0.0

    # ── 3. 执行测试 ──
    passed = sum(run_in_sandbox(code, t) for t in tests)
    total = len(tests)

    # ── 4. 计算 execution reward ──
    if reward_mode == "partial":
        return partial_reward(passed, total)
    else:
        return 1.0 if passed == total else 0.0
