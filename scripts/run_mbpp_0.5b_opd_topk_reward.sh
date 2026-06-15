# Arm D: 0.5B + SFT warmup + forward_kl_topk OPD as 3D Policy Gradient Reward
#
# ── 实现方式（参考代码 forward_kl_topk 模式）──
#   engine_workers.py _compute_topk_rm_scores():
#     rm_scores = teacher_renorm_logp  (B, T, K=16) — teacher 归一化 log-prob 作为 reward
#   losses.py：
#     k1_reward_loss = -mean( exp(rm_scores) × log_student_topk_live )
#   梯度：以 teacher 分布概率为权重，鼓励 student 在 teacher 概率高的 token 上增加概率
#
# ── 与 Arm C (k1) 的区别 ──
#   Arm C：单采样 token（1 个），以教师概率加权
#   Arm D：K=16 个 top-K token，以教师归一化分布加权（低方差，更精确的梯度估计）
#
# ── 注意：Arm D 是参考代码 forward_kl_topk 的近似实现 ──
#   参考代码：student top-K 来自 rollout（detached），teacher gather 到 student 位置
#   本实现：teacher top-K 作为 token 集合，live student log-prob gather 到 teacher 位置
#   差异：token 集合的选择策略不同（teacher-guided vs student-guided）
#
# ── 激活参数 ──
#   actor.bc_topk_reward=16        → top-K size K=16
#   actor.bc_topk_reward_coef=0.05 → reward coefficient
#   actor.bc_loss_coef=0           → 关闭 supervised loss
#
# 启动（patch_arm_cd.py 已运行后）：
#   nohup bash <repo>/scripts/run_mbpp_0.5b_opd_topk_reward.sh \
#     > <repo>/train_mbpp_0.5b_opd_topk_reward.log 2>&1 &

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

MODEL_PATH=${MODEL_PATH:-$HOME/models/coder_0.5b_sft_warmup/final}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/mbpp_v2/mbpp_train.parquet}
TEST_FILE=${TEST_FILE:-$HOME/data/mbpp_v2/mbpp_test.parquet}
REWARD_FN_PATH=$HOME/codellmRL/rewards/mbpp_reward.py

TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-$HOME/models/coder_1_5b_mbpp_grpo_best}
BC_TOPK_REWARD=16
BC_TOPK_REWARD_COEF=0.05

TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=64
PPO_MICRO_BATCH_SIZE_PER_GPU=8
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=512
PPO_MAX_TOKEN_LEN_PER_GPU=8192

ACTOR_LR=5e-6
ROLLOUT_TP=1
ROLLOUT_GPU_MEM_UTIL=0.30
ROLLOUT_N=8
TOTAL_EPOCHS=16
SAVE_FREQ=10
TEST_FREQ=10
MAX_CKPT_TO_KEEP=2

PROJECT_NAME=verl_grpo_mbpp_v2
EXPERIMENT_NAME=coder_0.5b_mbpp_v2_opd_topk_reward_lr5e6

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
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.teacher_model_path=${TEACHER_MODEL_PATH} \
    actor_rollout_ref.actor.bc_loss_coef=0 \
    actor_rollout_ref.actor.bc_topk_reward=${BC_TOPK_REWARD} \
    actor_rollout_ref.actor.bc_topk_reward_coef=${BC_TOPK_REWARD_COEF} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.max_model_len=1024 \
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
    "+trainer.save_best_metric=val-core/mbpp/acc/mean@1" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    transfer_queue.enable=False \
    trainer.resume_mode=auto \
    2>&1 | tee <repo>/train_mbpp_0.5b_opd_topk_reward.log
