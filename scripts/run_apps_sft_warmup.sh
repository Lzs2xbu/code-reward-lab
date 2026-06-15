# APPS SFT Warmup：用 APPS interview 人工解法对 Coder-1.5B 做 SFT
# 这是 Exp-LCB-A（GRPO）和 Exp-LCB-B（OPD）的共同起点
#
# 预期时间：约 60-90 分钟（5611 样本 × 2 epoch，effective batch=16）
#
# 启动：
#   nohup bash <repo>/scripts/run_apps_sft_warmup.sh \
#     > <repo>/train_apps_sft_warmup.log 2>&1 &

set -euo pipefail

PYTHON=${PYTHON:-python}
BASE_MODEL=$HOME/models/Qwen2.5-Coder-1.5B-Instruct
SFT_DATA=$HOME/data/apps/apps_sft_interview.parquet
OUTPUT_DIR=$HOME/models/coder_1_5b_apps_sft_warmup

echo "=================================================="
echo "[SFT Warmup] Starting APPS SFT warmup at $(date)"
echo "  Base model: $BASE_MODEL"
echo "  SFT data: $SFT_DATA"
echo "  Output: $OUTPUT_DIR"
echo "=================================================="

$PYTHON $HOME/codellmRL/scripts/run_sft_apps_warmup.py \
    --base_model $BASE_MODEL \
    --sft_data $SFT_DATA \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --max_seq_length 3072 \
    --warmup_steps 30 \
    --save_steps 100 \
    --logging_steps 10 \
    2>&1 | tee $HOME/codellmRL/train_apps_sft_warmup.log

echo ""
echo "=================================================="
echo "[SFT Warmup] Done at $(date)"
echo "Model saved to: $OUTPUT_DIR/final"
echo "Next: run run_apps_grpo_lr5e6.sh and run_apps_opd_bc005.sh in parallel"
echo "=================================================="
