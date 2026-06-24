# Exp-LCB-C：Coder-1.5B + APPS SFT warmup + 纯 GRPO（LR=5e-6，无 OPD）
#
# 对照关系（和 MBPP Exp 15 类比）：
#   MBPP Exp 15：SFT warmup + 纯GRPO + LR=5e-6 → 0.4533 (+10.5%)
#   Exp-LCB-C：  同样配置，但在 APPS 上训练、LCB 上评测
#
# 【为什么用 APPS 训练而用 LCB 评测？】
# APPS 是训练集（有足够多的题 + 测试用例）。
# LCB 是评测集（防污染，持续更新，不用于训练）。
# 这是标准的"train on A, eval on B"设置，更能体现模型的泛化能力。
#
# 【关键参数变化（相比 MBPP 脚本）】
# MAX_PROMPT_LENGTH: 512→1024  (竞赛题描述更长)
# MAX_RESPONSE_LENGTH: 512→2048 (竞赛题解法更长)
# ROLLOUT_GPU_MEM_UTIL: 0.5→0.35 (更长的 context 需要更多 KV cache)
#
# 启动（SFT warmup 完成后运行）：
#   nohup bash <repo>/scripts/run_apps_teacher_grpo.sh \
#     > <repo>/train_apps_teacher_grpo.log 2>&1 &

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

REPO_DIR=${REPO_DIR:-$PWD}
MODEL_DIR=${MODEL_DIR:-$HOME/models}
DATA_DIR=${DATA_DIR:-$HOME/data}

MODEL_PATH=${MODEL_PATH:-$MODEL_DIR/coder_1_5b_apps_teacher_sft_warmup/final}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/apps/apps_rl_interview.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/lcb/lcb_v5_verl.parquet}
REWARD_FN_PATH=${REWARD_FN_PATH:-$REPO_DIR/rewards/lcb_reward.py}

TRAIN_BATCH_SIZE=32          # APPS 题比 MBPP 长，batch size 适当减小
PPO_MINI_BATCH_SIZE=32
PPO_MICRO_BATCH_SIZE_PER_GPU=4
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2
MAX_PROMPT_LENGTH=1024       # 竞赛题描述更长
MAX_RESPONSE_LENGTH=2048     # 竞赛题解法更长
PPO_MAX_TOKEN_LEN_PER_GPU=4096

ACTOR_LR=5e-6
ROLLOUT_TP=1
ROLLOUT_GPU_MEM_UTIL=0.35    # 更长序列需要更多 KV cache 空间
ROLLOUT_N=8
TOTAL_EPOCHS=4               # 1692题 × 4 epoch = ~6800 样本，约 200 个 step（加速版）
SAVE_FREQ=5
TEST_FREQ=5
MAX_CKPT_TO_KEEP=2

PROJECT_NAME=verl_grpo_apps_lcb
EXPERIMENT_NAME=coder_1_5b_apps_teacher_grpo
LOG_FILE=${LOG_FILE:-$REPO_DIR/train_apps_teacher_grpo.log}

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
