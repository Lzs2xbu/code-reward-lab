# GRPO | Qwen3-1.7B | MBPP v2 | OPD (Online Policy Distillation) + Partial Reward + KL
#
# 在 partial + KL 基础上加入教师蒸馏 loss：
#   L_total = L_pg + kl_coef * KL(student || ref_initial)   [防止走远]
#             + distill_coef * KL(student || teacher_7B)     [引导向教师靠拢]
#
# 教师 log-probs 已离线预计算（reverse KL，token-level）：
#   data/mbpp_v2/mbpp_train_with_teacher.parquet
#   列：teacher_response_ids, teacher_logprobs, teacher_pass
#
# 依赖：mbpp_reward.py 中的 compute_score 需要支持 OPD distillation loss
#      训练数据需要使用 mbpp_train_with_teacher.parquet
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
echo "[pre-flight] GPU state after cleanup:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$GPU_MEM_USED" -gt 500 ]; then
    echo "[pre-flight] ERROR: GPU not clean (${GPU_MEM_USED} MiB). Aborting."
    exit 1
fi
echo "[pre-flight] GPU clean (${GPU_MEM_USED} MiB). Proceeding."

export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export VLLM_USE_V1=1
export REWARD_MODE=partial
export OPD_MODE=1           # 告知 reward fn 开启 OPD distillation log

MODEL_PATH=${MODEL_PATH:-$HOME/models/Qwen3-1.7B}
# OPD 专用训练集（含 teacher_response_ids / teacher_logprobs / teacher_pass）
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/mbpp_v2/mbpp_train_with_teacher.parquet}
TEST_FILE=${TEST_FILE:-$HOME/data/mbpp_v2/mbpp_test.parquet}
REWARD_FN_PATH=$HOME/codellmRL/rewards/mbpp_reward_opd.py   # OPD 版 reward fn

TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE_PER_GPU=4
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=512
PPO_MAX_TOKEN_LEN_PER_GPU=4096

ACTOR_LR=1e-6
KL_LOSS_COEF=0.001          # 与 ref_initial 的 KL（防止走远）
DISTILL_COEF=0.1             # 与 teacher 的蒸馏 loss 系数（待 grid search）
ROLLOUT_TP=1
ROLLOUT_GPU_MEM_UTIL=0.25
ROLLOUT_N=5
TOTAL_EPOCHS=16
SAVE_FREQ=10
TEST_FREQ=10
MAX_CKPT_TO_KEEP=2

PROJECT_NAME=verl_grpo_mbpp_v2
EXPERIMENT_NAME=qwen3_1_7b_mbpp_v2_opd_distill${DISTILL_COEF}_$(date +%Y%m%d_%H%M)

cd "${VERL_DIR:-verl}"

${PYTHON:-python} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
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
    "+trainer.save_best_metric=val-core/mbpp/acc/mean@1" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    transfer_queue.enable=False \
    trainer.resume_mode=disable \
    2>&1 | tee <repo>/train_mbpp_v2_opd_1_7b.log
