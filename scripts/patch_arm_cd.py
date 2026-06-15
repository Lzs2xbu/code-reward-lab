"""
Patch veRL for Arm C (k1 reward) and Arm D (top-K reward, correct implementation).

Arm C: k1 OPD as policy gradient reward
  - Same teacher forward as Arm B (sampled-token log-probs)
  - Loss = -mean( teacher_prob_detached × log π_student_live )
  - Gradient = -(teacher_prob_t) × ∂log π/∂θ  (weighted PG, NOT direct KL minimization)

Arm D: forward_kl_topk OPD as policy gradient reward
  - SAME student top-K selection as bc_topk (via live student logits)  ← KEY CORRECTION
  - SAME teacher lookup at student top-K positions
  - KL is computed but DETACHED: rm_scores = -KL_detached
  - Loss = -mean( rm_scores × log π_student_topK_live )  (REINFORCE)
  - Gradient = -(rm_scores) × ∂log π_topK/∂θ

Run on server:
  python3 <repo>/scripts/patch_arm_cd.py
"""
import os

HOME = os.path.expanduser("~")
VERL = f"{HOME}/verl/verl"


def patch_file(path, old, new, desc):
    with open(path) as f:
        content = f.read()
    if new.strip() in content:
        print(f"[SKIP] {desc} — already patched")
        return
    assert old in content, f"[ERROR] {desc}: old string not found in {path}"
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print(f"[OK]   {desc}")


# ══════════════════════════════════════════════════════════════
# Patch 1: actor.py — add new config fields
# ══════════════════════════════════════════════════════════════
patch_file(
    f"{VERL}/workers/config/actor.py",
    old="    bc_topk: int = 0  # Student Top-K OPD: k_student (paper default 16). 0 = disabled.",
    new="""    bc_topk: int = 0  # Student Top-K OPD: k_student (paper default 16). 0 = disabled.
    # Arm C: k1 reward (policy gradient weighted by teacher probability)
    # Activates when bc_loss_coef=0 and bc_reward_coef>0
    bc_reward_coef: float = 0.0
    # Arm D: top-K OPD as PG reward (reference code forward_kl_topk mode)
    # Same student top-K selection as bc_topk, but KL is detached → REINFORCE gradient
    bc_topk_reward: int = 0        # k_student, must be same as bc_topk would be
    bc_topk_reward_coef: float = 0.0""",
    desc="actor.py: add bc_reward_coef / bc_topk_reward / bc_topk_reward_coef",
)


# ══════════════════════════════════════════════════════════════
# Patch 2: fsdp/losses.py — add compute_student_topk_kl_pg_reward
# ══════════════════════════════════════════════════════════════
# Arm D core: same as compute_student_topk_kl but student renorm is detached
# Returns rm_scores (detached) + student_topk_lp_live (with gradient) for REINFORCE
TOPK_PG_REWARD_FN = '''
def compute_student_topk_kl_pg_reward(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
    k_student: int = 16,
) -> dict:
    """Arm D: Student Top-K OPD as PG reward (reference code forward_kl_topk mode).

    Correct implementation: uses student top-K (NOT teacher top-K) as the token set.
    KL is computed with detached student distribution → used as REINFORCE advantage.

    Gradient: -(rm_scores_detached) × ∂log π_topK / ∂θ
    vs bc_topk: ∂KL(p̄_student || q̄_teacher) / ∂θ

    Both use student top-K; only the gradient path differs.
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_lp = teacher_topk_log_probs.values().unsqueeze(0)   # (1, total_nnz, K_large)
    teacher_ids = teacher_topk_ids.values().unsqueeze(0)        # (1, total_nnz, K_large)

    # 1. Student top-K selection — use live logits (no gradient needed for selection)
    with torch.no_grad():
        student_topk_indices = student_logits.topk(k_student, dim=-1).indices  # (1, nnz, k)
        # Student log-probs DETACHED for KL reward computation
        student_log_probs_det = F.log_softmax(student_logits.detach(), dim=-1)
        student_topk_lp_det = student_log_probs_det.gather(-1, student_topk_indices)

    # 2. Student log-probs WITH gradient — for REINFORCE policy gradient loss
    student_log_probs_live = F.log_softmax(student_logits, dim=-1)  # WITH gradient
    student_topk_lp_live = student_log_probs_live.gather(-1, student_topk_indices)  # WITH gradient

    # 3. Look up teacher log-probs at student top-K positions (same as compute_student_topk_kl)
    match = (student_topk_indices.unsqueeze(-1) == teacher_ids.unsqueeze(-2))
    has_match = match.any(-1)
    match_idx = match.long().argmax(-1)
    teacher_lp_at_student = teacher_lp.gather(-1, match_idx)
    teacher_lp_at_student = torch.where(
        has_match, teacher_lp_at_student,
        torch.full_like(teacher_lp_at_student, -100.0)
    )

    # 4. Renormalize BOTH on student top-K (same as compute_student_topk_kl)
    with torch.no_grad():
        student_lse_det = student_topk_lp_det.logsumexp(dim=-1, keepdim=True)
        teacher_lse = teacher_lp_at_student.logsumexp(dim=-1, keepdim=True)
        student_topk_renorm_det = student_topk_lp_det - student_lse_det
        teacher_renorm = teacher_lp_at_student - teacher_lse

        # 5. KL(p̄_student || q̄_teacher) — DETACHED, used as reward
        kl_values = kl_divergence(
            log_q=teacher_renorm,
            log_p=student_topk_renorm_det,
        )  # (1, total_nnz) — positive scalar per token

        rm_scores = -kl_values  # negative KL = reward (lower divergence → higher reward)
        student_mass = student_topk_lp_det.exp().sum(dim=-1)
        teacher_mass = teacher_lp_at_student.exp().sum(dim=-1)

    # Return both: detached rm_scores (reward) + live student_topk_lp (for gradient)
    return {
        "rm_scores": rm_scores,                      # (1, total_nnz) detached
        "student_topk_lp_live": student_topk_lp_live,  # (1, total_nnz, k) WITH gradient
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
    }

'''

patch_file(
    f"{VERL}/trainer/distillation/fsdp/losses.py",
    old='def compute_student_topk_kl(\n',
    new=TOPK_PG_REWARD_FN + 'def compute_student_topk_kl(\n',
    desc="fsdp/losses.py: add compute_student_topk_kl_pg_reward (Arm D core)",
)


# ══════════════════════════════════════════════════════════════
# Patch 3: losses.py ppo_loss — add Arm C (k1 reward) and Arm D hooks
# ══════════════════════════════════════════════════════════════
OLD_BC_END = """        policy_loss = policy_loss + bc_loss_coef * bc_loss
        metrics["actor/bc_loss"] = Metric(value=bc_loss, aggregation=metric_aggregation)
        metrics["actor/bc_loss_coef"] = bc_loss_coef"""

NEW_BC_END = """        policy_loss = policy_loss + bc_loss_coef * bc_loss
        metrics["actor/bc_loss"] = Metric(value=bc_loss, aggregation=metric_aggregation)
        metrics["actor/bc_loss_coef"] = bc_loss_coef

    # Arm C: k1 OPD as PG reward
    # L = -mean( π_teacher(ŷ_t) × log π_student_live(ŷ_t) )
    # Gradient: -(teacher_prob_t) × ∂log π_student / ∂θ
    # Effect: token positions teacher assigns high probability to get larger gradient update
    bc_reward_coef = getattr(config, 'bc_reward_coef', 0.0)
    if bc_reward_coef > 0 and "teacher_log_probs" in data:
        teacher_lp_k1 = data["teacher_log_probs"].to(response_mask.device)  # (B, T)
        teacher_prob_k1 = teacher_lp_k1.detach().exp()  # π_teacher at sampled token, detached
        # REINFORCE: L = -mean(teacher_prob × log π_student)
        # When teacher_prob is high → large gradient to increase student log-prob here
        k1_reward_loss = -agg_loss(
            loss_mat=(teacher_prob_k1 * log_prob),
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )
        policy_loss = policy_loss + bc_reward_coef * k1_reward_loss
        metrics["actor/bc_k1_reward_loss"] = Metric(value=k1_reward_loss, aggregation=metric_aggregation)

    # Arm D: forward_kl_topk OPD as PG reward (via distillation model_output)
    # rm_scores: (B, T) detached KL reward; student_topk_lp_live: (B, T, K) WITH gradient
    # L = -mean( rm_scores × Σ_k log π_student_topK_live(k) )
    # Gradient: -(rm_scores) × ∂Σ_k log π_topK / ∂θ
    bc_topk_reward_coef = getattr(config, 'bc_topk_reward_coef', 0.0)
    if bc_topk_reward_coef > 0 and "rm_scores" in model_output and "student_topk_lp_live" in model_output:
        rm_scores_d = no_padding_2_padding(model_output["rm_scores"], data)              # (B, T)
        student_topk_lp = no_padding_2_padding(model_output["student_topk_lp_live"], data)  # (B, T, K)
        # REINFORCE over K top tokens: L = -mean( rm_score_t × Σ_k log π(k) )
        pg_loss_mat = -(rm_scores_d.unsqueeze(-1) * student_topk_lp).sum(dim=-1)   # (B, T)
        topk_reward_loss = agg_loss(
            loss_mat=pg_loss_mat,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )
        policy_loss = policy_loss + bc_topk_reward_coef * topk_reward_loss
        metrics["actor/bc_topk_reward_loss"] = Metric(value=topk_reward_loss, aggregation=metric_aggregation)"""

patch_file(
    f"{VERL}/workers/utils/losses.py",
    old=OLD_BC_END,
    new=NEW_BC_END,
    desc="losses.py: add Arm C (k1 reward) and Arm D (topk PG reward) hooks",
)


# ══════════════════════════════════════════════════════════════
# Patch 4: distillation/losses.py — register student_topk_kl_reward loss
# ══════════════════════════════════════════════════════════════
TOPK_REWARD_REGISTERED = '''
@register_distillation_loss(DistillationLossSettings(names=["student_topk_kl_reward"], use_topk=True))  # type: ignore[arg-type]
def compute_student_topk_kl_reward_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Arm D: Student Top-K OPD as REINFORCE PG reward.

    model_output["rm_scores"]:          (1, total_nnz) detached KL reward
    model_output["student_topk_lp_live"]: (1, total_nnz, K) WITH gradient
    Loss = -mean( rm_scores × Σ_k log π_student_topK_live(k) )
    This pushes student to increase probability at student top-K positions where KL is low
    (teacher agrees), and decrease where KL is high (student overconfident vs teacher).
    """
    # rm_scores and student_topk_lp_live come from compute_student_topk_kl_pg_reward()
    # called via logits_processor_func during student forward pass
    # model_output keys are nested tensors (total_nnz, K) — convert to padded (B, T, K)
    rm_scores_nested = model_output["rm_scores"]        # (1, total_nnz) nested
    student_lp_nested = model_output["student_topk_lp_live"]  # (1, total_nnz, K) nested
    student_mass = model_output.get("student_mass", None)
    teacher_mass = model_output.get("teacher_mass", None)

    rm_scores = no_padding_2_padding(rm_scores_nested, data)      # (B, T)
    student_topk_lp = no_padding_2_padding(student_lp_nested, data)  # (B, T, K)

    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()

    # REINFORCE: L = -mean_t( rm_score_t × Σ_k log π(k)_t )
    # Gradient: -(rm_score_t) × ∂Σ_k log π(k)/∂θ per response token
    pg_loss_per_token = -(rm_scores.unsqueeze(-1) * student_topk_lp).sum(dim=-1)  # (B, T)
    pg_loss_per_token = pg_loss_per_token.clamp(min=-10.0, max=10.0)  # numerical stability

    distillation_metrics = {}
    if student_mass is not None:
        sm = no_padding_2_padding(student_mass, data)
        distillation_metrics["distillation/student_mass"] = sm[response_mask_bool].mean().item()
    if teacher_mass is not None:
        tm = no_padding_2_padding(teacher_mass, data)
        distillation_metrics["distillation/teacher_mass"] = tm[response_mask_bool].mean().item()

    return pg_loss_per_token, distillation_metrics

'''

patch_file(
    f"{VERL}/trainer/distillation/losses.py",
    old="@register_distillation_loss(DistillationLossSettings(names=[\"student_topk_kl\"], use_topk=True))",
    new=TOPK_REWARD_REGISTERED + "@register_distillation_loss(DistillationLossSettings(names=[\"student_topk_kl\"], use_topk=True))",
    desc="distillation/losses.py: register student_topk_kl_reward (Arm D)",
)


# ══════════════════════════════════════════════════════════════
# Patch 5: distillation/losses.py compute_topk_loss — route to pg_reward
# ══════════════════════════════════════════════════════════════
OLD_TOPK_ROUTE = """    if loss_mode == "student_topk_kl":
        # Student Top-K OPD: select top-K based on student distribution (paper Section 2.2)"""

NEW_TOPK_ROUTE = """    if loss_mode == "student_topk_kl_reward":
        # Arm D: Student Top-K OPD as PG reward (same token set, detached KL as REINFORCE advantage)
        teacher_lp_raw = data["teacher_logprobs"]
        teacher_id_raw = data["teacher_ids"]
        if not teacher_lp_raw.is_nested:
            input_ids_field = data["input_ids"]
            cu_seqlens = input_ids_field.offsets()
            seq_lengths = cu_seqlens.diff()
            micro_bsz = teacher_lp_raw.shape[0]
            lp_parts  = [teacher_lp_raw[i, :seq_lengths[i].item(), :] for i in range(micro_bsz)]
            id_parts  = [teacher_id_raw[i, :seq_lengths[i].item(), :] for i in range(micro_bsz)]
            flat_lp  = torch.cat(lp_parts,  dim=0).to(student_logits.device)
            flat_ids = torch.cat(id_parts,  dim=0).to(student_logits.device)
            teacher_lp_raw  = torch.nested.nested_tensor_from_jagged(flat_lp,  cu_seqlens)
            teacher_id_raw  = torch.nested.nested_tensor_from_jagged(flat_ids, cu_seqlens)
        match config.strategy:
            case "fsdp" | "veomni":
                import verl.trainer.distillation.fsdp.losses as fsdp_losses
                outputs = fsdp_losses.compute_student_topk_kl_pg_reward(
                    student_logits=student_logits,
                    teacher_topk_log_probs=teacher_lp_raw,
                    teacher_topk_ids=teacher_id_raw,
                    config=distillation_config,
                    data_format=data_format,
                    k_student=k_student,
                )
            case _:
                raise NotImplementedError(f"student_topk_kl_reward not supported for {config.strategy=}")
    elif loss_mode == "student_topk_kl":
        # Student Top-K OPD: select top-K based on student distribution (paper Section 2.2)"""

patch_file(
    f"{VERL}/trainer/distillation/losses.py",
    old=OLD_TOPK_ROUTE,
    new=NEW_TOPK_ROUTE,
    desc="distillation/losses.py: route student_topk_kl_reward in compute_topk_loss",
)


# ══════════════════════════════════════════════════════════════
# Patch 6: engine_workers.py — add bc_topk_reward routing
# ══════════════════════════════════════════════════════════════
OLD_UPDATE_ROUTE = """            bc_topk = getattr(self.config.actor, 'bc_topk', 0)
            if bc_topk > 0:
                # Student Top-K OPD: compute teacher top-K data for full sequence
                data = self._compute_teacher_topk_data(data, k_large=max(bc_topk * 8, 128))
                # Signal prepare_model_outputs to call logits_processor_func (our distillation_ppo_loss)
                _tu.assign_non_tensor_data(data, "distillation_use_topk", True)
            else:
                # Legacy sampled-token bc_loss (broken gradient — kept for compat)
                data = self._compute_teacher_log_probs_bc(data)"""

NEW_UPDATE_ROUTE = """            bc_topk = getattr(self.config.actor, 'bc_topk', 0)
            bc_topk_reward = getattr(self.config.actor, 'bc_topk_reward', 0)
            if bc_topk > 0:
                # Arm B variant: direct backprop Top-K OPD (supervised KL loss, paper Eq 5)
                data = self._compute_teacher_topk_data(data, k_large=max(bc_topk * 8, 128))
                _tu.assign_non_tensor_data(data, "distillation_use_topk", True)
            elif bc_topk_reward > 0:
                # Arm D: top-K OPD as REINFORCE PG reward (reference code forward_kl_topk mode)
                # SAME student top-K selection as bc_topk (via live logits in logits_processor_func)
                # DIFFERENT gradient: KL is detached → rm_scores × ∂log π_topK / ∂θ
                data = self._compute_teacher_topk_data(data, k_large=max(bc_topk_reward * 8, 128))
                _tu.assign_non_tensor_data(data, "distillation_use_topk", True)
                # Override loss_mode to student_topk_kl_reward in the distillation config
                _bc_topk_r = bc_topk_reward
                _teacher_path_r = getattr(self.config.actor, 'teacher_model_path', None)
                if _bc_topk_r > 0 and _teacher_path_r and not self.distillation_enabled:
                    from functools import partial
                    from verl.trainer.distillation.losses import distillation_ppo_loss
                    from verl.workers.config.distillation import DistillationConfig, DistillationLossConfig
                    _distill_cfg_r = DistillationConfig(
                        enabled=False,
                        distillation_loss=DistillationLossConfig(
                            loss_mode="student_topk_kl_reward",
                            topk=_bc_topk_r,
                            use_policy_gradient=False,
                            use_task_rewards=False,
                            log_prob_min_clamp=-20.0,
                        )
                    )
                    from verl.workers.config.actor import ActorConfig as _ActorConfig
                    _actor_cfg_r = self.config.actor
                    self.loss_fn = partial(distillation_ppo_loss, config=_actor_cfg_r, distillation_config=_distill_cfg_r)
                    self.actor.set_loss_fn(self.loss_fn)
            else:
                # Arm B (supervised KL) or Arm C (k1 reward): sampled-token teacher log-probs
                data = self._compute_teacher_log_probs_bc(data)"""

patch_file(
    f"{VERL}/workers/engine_workers.py",
    old=OLD_UPDATE_ROUTE,
    new=NEW_UPDATE_ROUTE,
    desc="engine_workers.py: add bc_topk_reward routing to student_topk_kl_reward",
)


print("\n✅ All patches applied.")
print("\nVerify:")
print("  cd ${VERL_DIR:-verl} && python3 -c \"")
print("  from verl.workers.config.actor import FSDPActorConfig")
print("  c = FSDPActorConfig.__dataclass_fields__")
print("  print([k for k in c if 'bc' in k])\"")
print("\nExpected output: ['bc_loss_coef', 'bc_topk', 'bc_reward_coef', 'bc_topk_reward', 'bc_topk_reward_coef']")
