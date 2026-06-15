# LCB baseline 评测：Coder-1.5B-Instruct vs Coder-7B-Instruct
#
# 跑完后可以看到：
#   1. 两个模型在 LCB 上的绝对性能
#   2. 1.5B vs 7B 的 gap（这决定 OPD 有多大的知识迁移空间）
#
# 用法：
#   bash scripts/eval_lcb_baseline.sh
#   # 快速 debug（只跑20题）：
#   bash scripts/eval_lcb_baseline.sh --debug

set -euo pipefail

DEBUG=${1:-""}
PYTHON=${PYTHON:-python}
DATA_DIR=$HOME/data/lcb
OUT_DIR=$HOME/codellmRL/eval_results
LCB_FILE=$DATA_DIR/lcb_v5_stdin_stdout.parquet

CODER_1_5B=$HOME/models/Qwen2.5-Coder-1.5B-Instruct
CODER_7B=$HOME/models/Qwen2.5-Coder-7B-Instruct

mkdir -p $OUT_DIR

# ─────────────────────────────────────────────────────────────────────
# Step 0: 确认数据存在
# 注意：服务器无法访问外网，数据必须在本地准备好后 scp 上传
# 上传命令：scp -P <port> <tmp_dir>/lcb_data/lcb_v5_stdin_stdout.parquet <user>@<host>:<remote_data_dir>/lcb/
# ─────────────────────────────────────────────────────────────────────
if [ ! -f "$LCB_FILE" ]; then
    echo "ERROR: LCB 数据不存在：$LCB_FILE"
    echo "请在本地运行 prepare_lcb.py 生成 parquet 后 scp 上传到服务器。"
    exit 1
fi

echo "[Step 0] 数据已存在：$LCB_FILE"
$PYTHON -c "
import pandas as pd
df = pd.read_parquet('$LCB_FILE')
print(f'  共 {len(df)} 道题')
print('  难度分布:', df['difficulty'].value_counts().to_dict())
print('  平台分布:', df['platform'].value_counts().to_dict())
"

# debug 模式：只跑前20题
MAX_PROBLEMS_FLAG=""
if [ "$DEBUG" = "--debug" ]; then
    echo "[DEBUG MODE] 只评测前20题"
    MAX_PROBLEMS_FLAG="--max_problems 20"
fi

# ─────────────────────────────────────────────────────────────────────
# Step 1: 评测 Coder-1.5B-Instruct
# 【为什么 n_samples=5 而不是 20？】
# LCB 题比 MBPP 难得多，模型生成的 token 更多（max_tokens=2048 vs 1024）。
# 每题生成一次的时间约是 MBPP 的 2-3x。
# n=5 → SE ≈ sqrt(p*(1-p)/N) ≈ 0.022（和 MBPP n=20 相近），够用了。
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "[Step 1] 评测 Coder-1.5B-Instruct on LCB"
echo "=============================================="
$PYTHON $HOME/codellmRL/eval/eval_lcb.py \
    --model_path $CODER_1_5B \
    --test_file $LCB_FILE \
    --output_file $OUT_DIR/lcb_coder_1_5b_instruct.json \
    --n_samples 5 \
    --temperature 0.8 \
    --label coder_1_5b_instruct \
    $MAX_PROBLEMS_FLAG \
    2>&1 | tee $HOME/codellmRL/eval_lcb_1_5b.log

echo "[Step 1] Done"

# ─────────────────────────────────────────────────────────────────────
# Step 2: 评测 Coder-7B-Instruct
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "[Step 2] 评测 Coder-7B-Instruct on LCB"
echo "=============================================="
$PYTHON $HOME/codellmRL/eval/eval_lcb.py \
    --model_path $CODER_7B \
    --test_file $LCB_FILE \
    --output_file $OUT_DIR/lcb_coder_7b_instruct.json \
    --n_samples 5 \
    --temperature 0.8 \
    --label coder_7b_instruct \
    $MAX_PROBLEMS_FLAG \
    2>&1 | tee $HOME/codellmRL/eval_lcb_7b.log

echo "[Step 2] Done"

# ─────────────────────────────────────────────────────────────────────
# Step 3: 打印对比结果
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "FINAL COMPARISON: LCB Baseline"
echo "=============================================="
$PYTHON - <<'PYEOF'
import json
import os
import sys

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("<repo>/eval_results")

files = [
    (OUT_DIR + '/lcb_coder_1_5b_instruct.json', 'Coder-1.5B-Instruct'),
    (OUT_DIR + '/lcb_coder_7b_instruct.json',   'Coder-7B-Instruct'),
]

header = "%-25s %8s %8s  By Difficulty" % ("Model", "pass@1", "pass@5")
print(header)
print('-' * 75)
results = []
for fpath, name in files:
    try:
        with open(fpath) as f:
            d = json.load(f)
        s = d['summary']
        diff = s.get('by_difficulty', {})
        diff_str = ' | '.join("%s: %.1f%%" % (k, v*100) for k, v in sorted(diff.items()))
        print("%-25s %7.1f%% %7.1f%%  %s" % (name, s['pass@1']*100, s['pass@5']*100, diff_str))
        results.append(s)
    except FileNotFoundError:
        print("%-25s (not found)" % name)

if len(results) == 2:
    gap = results[1]['pass@1'] - results[0]['pass@1']
    print()
    print("7B vs 1.5B gap: %.1fpp (绝对值)" % (gap*100))
    print("  gap > 15pp -> OPD 有显著知识迁移空间")
    print("  gap < 5pp  -> LCB 也太简单了，考虑换更难的 benchmark")
PYEOF
