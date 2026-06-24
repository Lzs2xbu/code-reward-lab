# Arm D plus GRPO: token-level OPD reward plus outcome GRPO task advantage.
#
# Server snapshot note: the original launcher used `bc_shaped_reward_mode=plus_grpo`
# but still set `algorithm.adv_estimator=token_reward_direct`. The patched veRL
# tree registers `token_reward_direct_plus_grpo`, so this public launcher uses
# that estimator to make the task-reward branch active.
#
# Launch:
#   nohup bash <repo>/scripts/run_mbpp_0.5b_arm_d_plus_grpo.sh \
#     > <repo>/train_mbpp_0.5b_arm_d_plus_grpo.log 2>&1 &

set -xeuo pipefail

echo "[pre-flight] Stopping Ray and cleaning GPU..."
${RAY_BIN:-ray} stop --force 2>/dev/null || true
sleep 5
pkill -9 -f "verl.trainer" 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
sleep 10

GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$GPU_MEM_USED" -gt 5000 ]; then
    echo "[pre-flight] ERROR: GPU not clean (${GPU_MEM_USED} MiB). Aborting."
    exit 1
fi

export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export VLLM_USE_V1=1
export REWARD_MODE=partial

REPO_DIR=${REPO_DIR:-$PWD}
MODEL_DIR=${MODEL_DIR:-$HOME/models}
DATA_DIR=${DATA_DIR:-$HOME/data}

MODEL_PATH=${MODEL_PATH:-$MODEL_DIR/coder_0.5b_sft_warmup_1_5b/final}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-$MODEL_DIR/coder_1_5b_mbpp_grpo_best}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/mbpp_v2/mbpp_train.parquet}
TEST_FILE=${TEST_FILE:-$DATA_DIR/mbpp_v2/mbpp_test.parquet}
REWARD_FN_PATH=${REWARD_FN_PATH:-$REPO_DIR/rewards/mbpp_reward.py}
LOG_FILE=${LOG_FILE:-$REPO_DIR/train_mbpp_0.5b_arm_d_plus_grpo.log}

TRAIN_BATCH_SIZE=64
PPO_MINI_BATCH_SIZE=64
PPO_MICRO_BATCH_SIZE_PER_GPU=8
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=512
PPO_MAX_TOKEN_LEN_PER_GPU=8192

ACTOR_LR=5e-6
ROLLOUT_GPU_MEM_UTIL=0.30
ROLLOUT_N=8
TOTAL_EPOCHS=16
SAVE_FREQ=10
TEST_FREQ=10
MAX_CKPT_TO_KEEP=2
GRPO_OUTCOME_WEIGHT=${GRPO_OUTCOME_WEIGHT:-1.0}

PROJECT_NAME=verl_grpo_mbpp_v2
EXPERIMENT_NAME=${EXPERIMENT_NAME:-coder_0.5b_mbpp_v2_arm_d_plus_grpo_lr5e6}

cd "${VERL_DIR:-verl}"

${PYTHON:-python} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=token_reward_direct_plus_grpo \
    algorithm.grpo_outcome_weight=${GRPO_OUTCOME_WEIGHT} \
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
    +actor_rollout_ref.actor.bc_reward_coef=0 \
    +actor_rollout_ref.actor.bc_shaped_reward=True \
    +actor_rollout_ref.actor.bc_shaped_reward_mode=plus_grpo \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
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
    2>&1 | tee "${LOG_FILE}"
