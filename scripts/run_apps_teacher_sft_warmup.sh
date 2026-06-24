# Exp-LCB-C：Coder-1.5B + APPS teacher SFT warmup（7B 生成解法）
# 用法：bash <repo>/scripts/run_apps_teacher_sft_warmup.sh

set -euo pipefail

PYTHON=${PYTHON:-python}
REPO_DIR=${REPO_DIR:-$PWD}
MODEL_DIR=${MODEL_DIR:-$HOME/models}
DATA_DIR=${DATA_DIR:-$HOME/data}

BASE_MODEL=${BASE_MODEL:-$MODEL_DIR/Qwen2.5-Coder-1.5B-Instruct}
SFT_DATA=${SFT_DATA:-$DATA_DIR/apps/apps_teacher_sft.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-$MODEL_DIR/coder_1_5b_apps_teacher_sft_warmup}

echo "[SFT] Base model:  $BASE_MODEL"
echo "[SFT] Data:        $SFT_DATA"
echo "[SFT] Output:      $OUTPUT_DIR"

"$PYTHON" "$REPO_DIR/scripts/run_sft_apps_warmup.py" \
    --base_model     $BASE_MODEL \
    --sft_data       $SFT_DATA \
    --output_dir     $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate  2e-5 \
    --max_seq_length 3072 \
    --warmup_steps   20 \
    --save_steps     200 \
    --logging_steps  5

echo "[SFT] Done. Model saved to $OUTPUT_DIR/final"
