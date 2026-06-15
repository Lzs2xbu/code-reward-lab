# Arm E v2：用 1.5B GRPO best 生成 SFT warmup 数据，重训 Arm E（方差归一化）
#
# 实验假设：
#   原始 Arm E 失败的原因之一是 SFT warmup 用 7B 数据（与 OPD teacher=1.5B 不匹配）
#   → 初始 rm_scores=-6.8/token（大，方向错）→ 崩溃
#
#   修复：用 OPD teacher（1.5B GRPO best）的成功解法做 SFT warmup
#   → 学生从起点就对齐 OPD teacher 分布
#   → 期望：初始 rm_scores 更小，OPD shaped reward 更稳定
#
# 链式执行（可 nohup 运行）：
#   Step 1: 生成 1.5B 教师解法 → mbpp_train_1_5b_teacher.parquet
#   Step 2: 0.5B SFT warmup（使用 1.5B 解法）
#   Step 3: Arm E（方差归一化，β 自动调整）
#   Step 4: Arm C（对照，用相同 SFT warmup）
#
# 启动：
#   nohup bash <repo>/scripts/run_arm_e_v2_chain.sh \
#     > <repo>/arm_e_v2_chain.log 2>&1 &

set -euo pipefail
PYTHON=${PYTHON:-python}
LOG_DIR=$HOME/codellmRL

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ─────────────────────────────────────────────
# Step 1: 生成 1.5B 教师解法
# ─────────────────────────────────────────────
log "=== Step 1: 生成 1.5B GRPO best 解法 ==="
OUTPUT_TEACHER=$HOME/data/mbpp_v2/mbpp_train_1_5b_teacher.parquet

if [ -f "$OUTPUT_TEACHER" ]; then
    log "已存在 $OUTPUT_TEACHER，跳过生成"
else
    $PYTHON $HOME/codellmRL/scripts/prepare_mbpp_1_5b_teacher_sft.py \
        --model $HOME/models/coder_1_5b_mbpp_grpo_best \
        --train_file $HOME/data/mbpp_v2/mbpp_train.parquet \
        --output $OUTPUT_TEACHER \
        --n_samples 5 \
        --temperature 0.8 \
        2>&1 | tee $LOG_DIR/prepare_1_5b_teacher.log
    log "1.5B 教师解法生成完毕"
fi

# ─────────────────────────────────────────────
# Step 2: 0.5B SFT warmup（用 1.5B 解法）
# ─────────────────────────────────────────────
log "=== Step 2: 0.5B SFT warmup (1.5B teacher data) ==="
SFT_OUTPUT=$HOME/models/coder_0.5b_sft_warmup_1_5b

if [ -f "$SFT_OUTPUT/final/config.json" ]; then
    log "SFT warmup 模型已存在，跳过"
else
    $PYTHON $HOME/codellmRL/scripts/run_sft_teacher_warmup.py \
        --base_model $HOME/models/Qwen2.5-Coder-0.5B-Instruct \
        --teacher_data $OUTPUT_TEACHER \
        --output_dir $SFT_OUTPUT \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-5 \
        2>&1 | tee $LOG_DIR/sft_0.5b_1_5b_warmup.log
    log "0.5B SFT warmup (1.5B teacher) 完毕"
fi

# ─────────────────────────────────────────────
# Step 3: Arm E v2（方差归一化，1.5B SFT warmup）
# ─────────────────────────────────────────────
log "=== Step 3: Arm E v2（shaped reward，方差归一化）==="
export MODEL_PATH=$SFT_OUTPUT/final
export EXPERIMENT_NAME=coder_0.5b_mbpp_v2_opd_shaped_varnorm_1_5b_warmup_lr5e6
bash $HOME/codellmRL/scripts/run_mbpp_0.5b_opd_shaped_reward_b001.sh \
    2>&1 | tee $LOG_DIR/train_arm_e_v2.log
log "Arm E v2 完毕"

# ─────────────────────────────────────────────
# Step 4: Arm C v2（对照，相同 SFT warmup）
# ─────────────────────────────────────────────
log "=== Step 4: Arm C v2（teacher-weighted loss，1.5B SFT warmup）==="
export MODEL_PATH=$SFT_OUTPUT/final
export EXPERIMENT_NAME=coder_0.5b_mbpp_v2_opd_k1_reward_1_5b_warmup_lr5e6
bash $HOME/codellmRL/scripts/run_mbpp_0.5b_opd_k1_reward.sh \
    2>&1 | tee $LOG_DIR/train_arm_c_v2.log
log "Arm C v2 完毕"

log "=== 全部完成，请运行 eval ==="
