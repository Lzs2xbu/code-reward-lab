set -euo pipefail

PYTHON=${PYTHON:-python}
REPO_DIR=${REPO_DIR:-$PWD}
MODEL_DIR=${MODEL_DIR:-$HOME/models}
DATA_DIR=${DATA_DIR:-$HOME/data}

MODEL=${MODEL:-$MODEL_DIR/coder_1_5b_apps_teacher_sft_warmup/final}
LCB_FILE=${LCB_FILE:-$DATA_DIR/lcb/lcb_v5_stdin_stdout.parquet}
OUT_DIR=${OUT_DIR:-$REPO_DIR/eval_results}

mkdir -p "$OUT_DIR"

echo "=== Evaluating teacher SFT model on LCB ==="

echo "--- [1/2] Greedy eval ---"
"$PYTHON" "$REPO_DIR/eval/eval_lcb.py" \
    --model_path   "$MODEL" \
    --test_file    "$LCB_FILE" \
    --output_file  "$OUT_DIR/lcb_teacher_sft_greedy.json" \
    --n_samples    1 \
    --temperature  0.0 \
    --label        teacher_sft_greedy

echo "--- [2/2] Sampling eval (n=5, T=0.8) ---"
"$PYTHON" "$REPO_DIR/eval/eval_lcb.py" \
    --model_path   "$MODEL" \
    --test_file    "$LCB_FILE" \
    --output_file  "$OUT_DIR/lcb_teacher_sft.json" \
    --n_samples    5 \
    --temperature  0.8 \
    --label        teacher_sft

echo "=== Comparison ==="
"$PYTHON" - "$OUT_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])


def load_summary(name):
    path = out_dir / name
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f).get("summary", {})


rows = [
    ("1.5B baseline", load_summary("lcb_coder_1_5b_instruct.json")),
    ("Human SFT", load_summary("lcb_sft_warmup.json")),
    ("Teacher SFT greedy", load_summary("lcb_teacher_sft_greedy.json")),
    ("Teacher SFT (n=5,T=0.8)", load_summary("lcb_teacher_sft.json")),
]

print(f"{'Model':<32} {'p@1':>6} {'p@5':>6}  by_difficulty")
print("-" * 80)
for name, summary in rows:
    if summary is None:
        print(f"{name:<32} {'MISS':>6} {'MISS':>6}")
        continue
    by_difficulty = summary.get("by_difficulty", {})
    difficulty_text = " | ".join(
        f"{k}:{v:.1%}" for k, v in sorted(by_difficulty.items())
    )
    pass_at_1 = summary.get("pass@1", math.nan)
    pass_at_5 = summary.get("pass@5", math.nan)
    print(f"{name:<32} {pass_at_1:>5.1%} {pass_at_5:>6.1%}  {difficulty_text}")

baseline = rows[0][1] or {}
teacher_sft = rows[-1][1] or {}
baseline_p5 = baseline.get("pass@5", 0.0)
teacher_p5 = teacher_sft.get("pass@5", 0.0)
print()
if teacher_p5 >= baseline_p5:
    print("OK: no narrowing")
else:
    print(f"WARNING: narrowing detected (teacher sft {teacher_p5:.1%} < baseline {baseline_p5:.1%})")
PY
