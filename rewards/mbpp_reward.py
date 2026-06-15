# rewards/mbpp_reward.py
"""
MBPP 代码执行奖励函数，通过 veRL 的 custom_reward_function 机制接入。
环境变量 REWARD_MODE=binary|partial 控制奖励类型。

函数签名符合 veRL NaiveRewardManager 要求：
    compute_score(data_source, solution_str, ground_truth, extra_info) -> float
"""

import json
import multiprocessing
import os
import resource


def _worker(code: str, test_case: str, result_queue: multiprocessing.Queue) -> None:
    """在子进程中执行代码 + 测试用例，结果放入队列。"""
    import sys, os
    # 限制虚拟地址空间为 512MB，防止内存炸弹
    mem_limit = 512 * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
    except (ValueError, resource.error):
        pass  # 某些系统不支持，忽略

    # 重定向 stdout/stderr，防止生成代码的 print() 污染 Ray 日志
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    sys.stderr = devnull

    try:
        exec(compile(code + "\n" + test_case, "<string>", "exec"), {})
        result_queue.put(True)
    except Exception:
        result_queue.put(False)


def run_in_sandbox(code: str, test_case: str, timeout: int = 5) -> bool:
    """
    在隔离的子进程中执行 code + test_case。
    - 超时后主动 kill，不留僵尸进程
    - 子进程内存限制 512MB
    返回：True（测试通过）/ False（异常/超时/断言失败）
    """
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(code, test_case, q))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        p.join()
        return False
    return q.get() if not q.empty() else False


def compute_score_binary(code: str, test_list: list) -> float:
    """全部测试用例通过 = 1.0，否则 = 0.0"""
    results = [run_in_sandbox(code, t) for t in test_list]
    return 1.0 if all(results) else 0.0


def compute_score_partial(code: str, test_list: list) -> float:
    """
    非线性 partial credit：
    0/3 -> 0.0, 1/3 -> 0.2, 2/3 -> 0.6, 3/3 -> 1.0
    最后一步跨度 (+0.4) 是第一步 (+0.2) 的两倍，强化冲击全通过的动机。
    """
    results = [run_in_sandbox(code, t) for t in test_list]
    passed = sum(results)
    reward_map = {0: 0.0, 1: 0.2, 2: 0.6, 3: 1.0}
    return reward_map[passed]


def extract_code(solution_str: str) -> str:
    """
    从模型输出中提取可执行 Python 代码。
    处理两种常见包装：
    1. Qwen3 thinking 模式：<think>...</think>\n实际代码
    2. Markdown 代码块：```python\n代码\n```
    """
    import re
    # Step 1: 去掉所有 <think>...</think> 块（包括跨行）
    code = re.sub(r'<think>.*?</think>', '', solution_str, flags=re.DOTALL)
    code = code.strip()
    # Step 2: 提取 markdown 代码块内容（```python ... ``` 或 ``` ... ```）
    m = re.search(r'```(?:python)?\s*\n(.*?)```', code, flags=re.DOTALL)
    if m:
        code = m.group(1).strip()
    return code


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """
    veRL custom_reward_function 入口。
    - data_source: 应为 "mbpp"（由 parquet 的 data_source 字段决定）
    - solution_str: 模型生成的代码字符串（可能包含 Qwen3 的 <think>...</think> 块）
    - ground_truth: JSON 序列化的 test_list，e.g. '["assert func(1)==2", ...]'
    - extra_info: 可选的额外信息（本实验不使用）
    """
    if data_source != "mbpp":
        raise ValueError(f"mbpp_reward.py: unexpected data_source={data_source!r}")

    code = extract_code(solution_str)
    test_list = json.loads(ground_truth)
    reward_mode = os.environ.get("REWARD_MODE", "binary")

    if reward_mode == "binary":
        return compute_score_binary(code, test_list)
    elif reward_mode == "partial":
        return compute_score_partial(code, test_list)
    else:
        raise ValueError(f"Unknown REWARD_MODE={reward_mode!r}, expected 'binary' or 'partial'")
