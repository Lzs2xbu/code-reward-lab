# BC sweep eval: 传入 BC_TAG（如 "02" "03"）自动 merge + eval
set -euo pipefail

BC_TAG=${1:?"Usage: auto_eval_opd_bc_sweep.sh <bc_tag>  e.g. 02 or 03"}
EXP_NAME=coder_1_5b_mbpp_v2_opd_bc${BC_TAG}
PYTHON=${PYTHON:-python}
CKPT_BASE=$HOME/verl/checkpoints/verl_grpo_mbpp_v2/$EXP_NAME
MERGE_BASE=$HOME/models/eval_merged
TEST=$HOME/data/mbpp_v2/mbpp_test.parquet
EVAL=$HOME/codellmRL/eval/eval_mbpp.py
OUT=$HOME/codellmRL/eval_results

echo "=========================================="
echo "Auto eval: $EXP_NAME"
echo "=========================================="
mkdir -p $OUT $MERGE_BASE

echo "[1] Merging best checkpoint..."
mkdir -p $MERGE_BASE/coder_1_5b_opd_bc${BC_TAG}_best
$PYTHON -m verl.model_merger merge --backend fsdp \
    --local_dir "$CKPT_BASE/best_checkpoint/actor" \
    --target_dir "$MERGE_BASE/coder_1_5b_opd_bc${BC_TAG}_best" \
    2>&1 | tee $HOME/codellmRL/merge_opd_bc${BC_TAG}_best.log

FINAL_CKPT=$(ls -d "$CKPT_BASE"/global_step_*/actor 2>/dev/null | sort -t_ -k3 -n | tail -1)
FINAL_STEP=$(echo $FINAL_CKPT | grep -oP "global_step_\K[0-9]+")
echo "[2] Merging final checkpoint (step $FINAL_STEP)..."
mkdir -p $MERGE_BASE/coder_1_5b_opd_bc${BC_TAG}_step${FINAL_STEP}
$PYTHON -m verl.model_merger merge --backend fsdp \
    --local_dir "$FINAL_CKPT" \
    --target_dir "$MERGE_BASE/coder_1_5b_opd_bc${BC_TAG}_step${FINAL_STEP}" \
    2>&1 | tee $HOME/codellmRL/merge_opd_bc${BC_TAG}_step${FINAL_STEP}.log

echo "[3] Evaluating best..."
$PYTHON $EVAL --model_path "$MERGE_BASE/coder_1_5b_opd_bc${BC_TAG}_best" \
    --test_file $TEST --output_file $OUT/coder_1_5b_opd_bc${BC_TAG}_best.json \
    --n_samples 20 --temperature 0.8 --label coder_1_5b_opd_bc${BC_TAG}_best

echo "[4] Evaluating final (step $FINAL_STEP)..."
$PYTHON $EVAL --model_path "$MERGE_BASE/coder_1_5b_opd_bc${BC_TAG}_step${FINAL_STEP}" \
    --test_file $TEST --output_file $OUT/coder_1_5b_opd_bc${BC_TAG}_step${FINAL_STEP}.json \
    --n_samples 20 --temperature 0.8 --label coder_1_5b_opd_bc${BC_TAG}_step${FINAL_STEP}

rm -rf $MERGE_BASE/coder_1_5b_opd_bc${BC_TAG}_best $MERGE_BASE/coder_1_5b_opd_bc${BC_TAG}_step${FINAL_STEP}
rm -rf "$CKPT_BASE"

echo ""
echo "=========================================="
echo "EVAL SUMMARY (BC=${BC_TAG})"
$PYTHON -c "
import json, glob
for f in sorted(glob.glob('$OUT/coder_1_5b_opd_bc${BC_TAG}*.json')):
    d = json.load(open(f))
    s = d.get('summary', d)
    print(f\"{s.get('label','?')}: pass@1={s.get('pass@1',0):.4f}  pass@5={s.get('pass@5',0):.4f}\")
"
echo "Reference: BC=0(Exp15)=0.4533 | BC=0.01(Exp11c)=0.4496 | BC=0.05(Exp11b)=0.4599"
echo "=========================================="
