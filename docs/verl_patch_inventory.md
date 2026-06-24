# veRL Patch Inventory

This document records the source-level veRL changes observed in the remote
training environment and maps them to the local experiment launch scripts.
It is intentionally an inventory, not a vendored veRL fork.

## Modified veRL Areas

The remote veRL checkout has local modifications in these areas:

- `verl/workers/config/actor.py`: adds OPD-related actor config fields.
- `verl/workers/engine_workers.py`: loads the teacher model, computes teacher sampled-token log-probs, teacher top-K data, and shaped rewards.
- `verl/workers/utils/losses.py`: preserves `teacher_log_probs` in `ppo_loss` and adds sampled-token BC / reward losses.
- `verl/trainer/distillation/fsdp/losses.py`: adds student-top-K KL reward computation.
- `verl/trainer/distillation/losses.py`: routes `student_topk_kl` and `student_topk_kl_reward`.
- `verl/trainer/ppo/ray_trainer.py`: injects teacher shaped reward before advantage computation.
- `verl/trainer/ppo/core_algos.py`: registers extra advantage estimators.
- `verl/trainer/config/algorithm.py` and `verl/trainer/config/actor/dp_actor.yaml`: expose new config knobs.
- Rollout/attention/transfer queue files contain runtime compatibility and memory-management changes.

## Actor Config Knobs

Observed OPD-related fields:

- `teacher_model_path`: enables teacher loading inside the actor worker.
- `bc_loss_coef`: sampled-token OPD as a supervised KL-style loss.
- `bc_topk`: student-top-K OPD as direct distillation loss.
- `bc_reward_coef`: Arm C, sampled-token teacher-probability-weighted policy-gradient reward.
- `bc_topk_reward` and `bc_topk_reward_coef`: Arm D, student-top-K OPD as policy-gradient reward.
- `bc_shaped_reward`: Arm E, injects teacher reward before advantage computation.
- `bc_shaped_reward_mode`: `legacy`, `direct`, or `plus_grpo`.
- `bc_shaped_reward_beta`: scaling factor for legacy shaped reward.
- `arm_d_k1_coef`: intended sampled-token KL reward coefficient for a K=1 Arm D variant.

Important caveat: the pulled `verl/workers/utils/losses.py` contains the Arm C
branch, but I did not find an active `arm_d_k1_coef` loss branch in that file.
Treat `scripts/run_mbpp_0.5b_arm_d_k1.sh` as requiring a fresh verification
that the corresponding veRL patch is actually applied.

## Teacher Data Flow

When `teacher_model_path` is set:

1. The actor worker loads the teacher model in bfloat16 with eager attention.
2. During actor update, the teacher is moved to GPU only for teacher scoring and
   then offloaded back to CPU before the student backward pass.
3. For sampled-token OPD, `_compute_teacher_log_probs_bc` reconstructs padded
   `input_ids` from veRL packed tensors, slices response logits, casts logits to
   float32 before `log_softmax`, and stores `teacher_log_probs`.
4. For student-top-K OPD, `_compute_teacher_topk_data` stores teacher top-K log
   probabilities and token ids. The distillation loss then looks up teacher
   probabilities at the student's current top-K tokens.
5. For shaped reward, `compute_teacher_shaped_reward` returns
   `teacher_rm_scores = clamp(teacher_lp - old_student_lp, -10, 10)`.

## Loss And Reward Modes

### Sampled-Token BC Loss

Enabled by `actor_rollout_ref.actor.bc_loss_coef > 0`.

The actor loss adds:

```text
mean(log p_student(sampled_token) - log p_teacher(sampled_token))
```

weighted by `bc_loss_coef`. This is the simplest OPD mode and is used by
`scripts/run_mbpp_0.5b_opd.sh`.

### Arm C K=1 Reward

Enabled by `actor_rollout_ref.actor.bc_reward_coef > 0`.

The actor loss adds a policy-gradient-style term:

```text
-mean(exp(log p_teacher(sampled_token)) * log p_student(sampled_token))
```

This weight is always non-negative, so it only pushes probability up on sampled
tokens that the teacher likes.

### Arm D Student-Top-K Reward

Enabled by `bc_topk_reward > 0` with a registered
`student_topk_kl_reward` distillation loss.

The worker computes student top-K tokens, looks up teacher probabilities at
those same tokens, computes a detached negative-KL reward, and applies:

```text
-mean(reward * sum(log p_student(student_top_k_tokens)))
```

This is more faithful to top-K OPD, but it is memory-sensitive.

### Arm E Shaped Reward

Enabled by `bc_shaped_reward=True`.

The trainer calls `compute_teacher_shaped_reward` before advantage computation.
The returned token reward is either:

- `legacy`: variance-scaled addition to the task reward.
- `direct`: replace token-level rewards with teacher reward and use
  `algorithm.adv_estimator=token_reward_direct`.
- `plus_grpo`: store the original task reward as `true_reward_score`, replace
  token-level rewards with teacher reward, and use
  `algorithm.adv_estimator=token_reward_direct_plus_grpo`.

The remote `scripts/run_mbpp_0.5b_arm_d_plus_grpo.sh` snapshot used
`bc_shaped_reward_mode=plus_grpo` but still set
`algorithm.adv_estimator=token_reward_direct`. The local public launcher uses
`token_reward_direct_plus_grpo`, because that is the estimator registered in the
patched veRL tree for combining teacher token reward with GRPO task advantage.

## GRPO Advantage And Sampling

The default GRPO path groups rollouts by prompt id and computes:

```text
advantage_i = (reward_i - mean(group_rewards)) / (std(group_rewards) + eps)
```

when `norm_adv_by_std_in_grpo=True`; otherwise it subtracts only the group mean.
The result is broadcast over response tokens using the response mask.

The MBPP and APPS launchers imported from the remote environment generally use:

- `actor_rollout_ref.rollout.n=8`
- `algorithm.use_kl_in_reward=False`
- `actor_rollout_ref.actor.use_kl_loss=False`
- `actor_rollout_ref.actor.entropy_coeff=0`
- `trainer.balance_batch=True`

I did not find an enabled zero-advantage sample deletion or resampling setting
in the imported launch scripts. The veRL config contains `filter_groups`
machinery, but these experiment scripts do not turn it on.

## Local Alignment Decision

The local repository keeps reusable experiment launchers and patch notes, but
does not vendor the whole remote veRL checkout. This avoids mixing a patched
framework fork with the public experiment scripts and keeps private runtime
details out of the repository.
