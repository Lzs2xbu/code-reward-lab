# Exp-LCB-B：Coder-1.5B + APPS SFT warmup + OPD（BC=0.05，LR=5e-6）
#
# 对照关系：
#   Exp-LCB-A（run_apps_grpo_lr5e6.sh）：纯 GRPO，无 BC loss，无 teacher
#   Exp-LCB-B（本脚本）：             OPD，BC=0.05，teacher=Coder-7B-Instruct
#
# 【OPD 与 GRPO 的唯一差异（结构上）】
#   新增两个参数：
#     actor_rollout_ref.actor.teacher_model_path  → 7B teacher 模型路径
#     actor_rollout_ref.actor.bc_loss_coef=0.05   → BC loss 权重
#   总 loss = GRPO loss + 0.05 × BC loss
#   BC loss = KL(student_logits || teacher_logits)，在 student rollout 上计算
#
# 【BC=0.05 的来源】
#   MBPP 实验（Exp 11b vs 11c）：
#     BC=0.01 → entropy collapse → pass@1 下降
#     BC=0.05 → stable → 当前 MBPP 最优（0.4599）
#   在 APPS 上沿用 BC=0.05 作为首选值。
#
# 【为什么 entropy_from_logits_with_chunking=true？】
#   OPD 模式下，actor 需要同时保存 student 和 teacher 的 logits 用于 BC loss 计算。
#   chunking 把 forward pass 切成小块，避免同时存两份完整 logits 导致 OOM。
#
# 启动（SFT warmup 完成后运行，可与 Exp-LCB-A 同时 nohup）：
#   nohup bash <repo>/scripts/run_apps_opd_bc005.sh \
#     > <repo>/train_apps_opd_bc005.log 2>&1 &

set -xeuo pipefail

echo "[pre-flight] Stopping Ray cluster and killing GPU processes..."
${RAY_BIN:-ray} stop --force 2>/dev/null || true
sleep 5
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$') || true
if [ -n "$GPU_PIDS" ]; then
    echo "[pre-flight] Killing GPU-occupying PIDs: $GPU_PIDS"
    echo "$GPU_PIDS" | xargs -r kill -9 2>/dev/null || true
fi
pkill -9 -f "verl.trainer" 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
sleep 15
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$GPU_MEM_USED" -gt 500 ]; then
    echo "[pre-flight] ERROR: GPU not clean (${GPU_MEM_USED} MiB). Aborting."
    exit 1
fi

export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export VLLM_USE_V1=1
export REWARD_MODE=partial

MODEL_PATH=${MODEL_PATH:-$HOME/models/coder_1_5b_apps_sft_warmup/final}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/apps/apps_rl_interview.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/lcb/lcb_v5_verl.parquet}
REWARD_FN_PATH=$HOME/codellmRL/rewards/lcb_reward.py
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-$HOME/models/Qwen2.5-Coder-7B-Instruct}

TRAIN_BATCH_SIZE=32
PPO_MINI_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE_PER_GPU=4
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=2048
PPO_MAX_TOKEN_LEN_PER_GPU=4096

ACTOR_LR=5e-6
ROLLOUT_TP=1
ROLLOUT_GPU_MEM_UTIL=0.35
ROLLOUT_N=8
TOTAL_EPOCHS=4
SAVE_FREQ=5
TEST_FREQ=5
MAX_CKPT_TO_KEEP=2

BC_LOSS_COEF=0.05

PROJECT_NAME=verl_grpo_apps_lcb
EXPERIMENT_NAME=coder_1_5b_apps_opd_bc005
LOG_FILE=$HOME/codellmRL/train_apps_opd_bc005.log

cd "${VERL_DIR:-verl}"

${PYTHON:-python} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    "+actor_rollout_ref.model.override_config.attn_implementation=eager" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.teacher_model_path=${TEACHER_MODEL_PATH} \
    actor_rollout_ref.actor.bc_loss_coef=${BC_LOSS_COEF} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.max_model_len=3072 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.custom_reward_function.path=${REWARD_FN_PATH} \
    reward.custom_reward_function.name=compute_score \
    trainer.balance_batch=True \
    "trainer.logger=[console]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.max_actor_ckpt_to_keep=${MAX_CKPT_TO_KEEP} \
    "+trainer.save_best_metric=val-core/lcb/acc/mean@1" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    transfer_queue.enable=False \
    trainer.resume_mode=disable \
    2>&1 | tee ${LOG_FILE}
