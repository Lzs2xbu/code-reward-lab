# OPD Exp 9: Qwen2.5-Coder-0.5B 学生模型 + SFT warmup → GRPO
#
# 模型来源：本地下载后上传到服务器
#   huggingface-cli download Qwen/Qwen2.5-Coder-0.5B-Instruct \
#       --local-dir models/Qwen2.5-Coder-0.5B-Instruct
#   rsync -avz --progress -e 'ssh -p <port>' \
#       models/Qwen2.5-Coder-0.5B-Instruct \
#       <user>@<host>:models/
#
# 两阶段流程（同 Exp 8）：
#   阶段1 SFT warmup：在 teacher_pass=True 样本上 SFT
#     python <repo>/scripts/run_sft_teacher_warmup.py \
#         --base_model models/Qwen2.5-Coder-0.5B-Instruct \
#         --teacher_data data/mbpp_v2/mbpp_train_with_teacher.parquet \
#         --output_dir models/coder_0.5b_sft_warmup \
#         --num_train_epochs 3 \
#         --per_device_train_batch_size 8 \
#         --gradient_accumulation_steps 4 \
#         --learning_rate 2e-5
#   阶段2 GRPO：本脚本
#
# 基准：partial_v2 1.7B pass@1=0.3550

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

# ── 0.5B 学生模型，从 SFT warmup 起训 ──
MODEL_PATH=${MODEL_PATH:-$HOME/models/coder_0.5b_sft_warmup/final}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/mbpp_v2/mbpp_train.parquet}
TEST_FILE=${TEST_FILE:-$HOME/data/mbpp_v2/mbpp_test.parquet}
REWARD_FN_PATH=$HOME/codellmRL/rewards/mbpp_reward.py

# ── OPD: 7B 在线教师模型（float16 在线蒸馏） ──
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-$HOME/models/Qwen2.5-Coder-7B-Instruct}
BC_LOSS_COEF=${BC_LOSS_COEF:-0.1}

# 0.5B 模型显存占用约 1GB，有大量余量
# 相比 1.7B 脚本：batch size 更大，rollout GPU mem util 更高，log_prob_max_token_len 更大
TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=64
PPO_MICRO_BATCH_SIZE_PER_GPU=8
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=512
PPO_MAX_TOKEN_LEN_PER_GPU=8192

ACTOR_LR=1e-6
ROLLOUT_TP=1
ROLLOUT_GPU_MEM_UTIL=0.25   # 有 7B teacher 常驻 GPU，降低 vLLM 占用给 teacher 留空间
ROLLOUT_N=8                 # 1.7B 用 5，更多采样提升 GRPO 梯度质量
TOTAL_EPOCHS=16
SAVE_FREQ=10
TEST_FREQ=10
MAX_CKPT_TO_KEEP=2

PROJECT_NAME=verl_grpo_mbpp_v2
EXPERIMENT_NAME=${EXPERIMENT_NAME:-coder_0.5b_mbpp_v2_opd_sft_grpo_$(date +%Y%m%d_%H%M)}

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
    actor_rollout_ref.actor.bc_loss_coef=${BC_LOSS_COEF} \
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
    2>&1 | tee <repo>/train_mbpp_v2_coder_0.5b_opd_grpo.log
