# apply_verl_opd_patch.sh
# 在服务器上对 veRL 应用单卡 On-Policy Distillation (OPD) patch
# 在 GRPO actor update 中加入 7B teacher 的 BC loss
#
# 使用方法：bash <repo>/scripts/apply_verl_opd_patch.sh
#
# 修改的文件：
#   1. $VERL_DIR/verl/workers/engine_workers.py  — 加载 teacher 模型，计算 teacher log probs
#   2. $VERL_DIR/verl/workers/utils/losses.py    — 在 ppo_loss 中加入 BC loss 项

set -euo pipefail

VERL_DIR="$HOME/verl"
PATCH_BACKUP_DIR="$HOME/codellmRL/verl_patch_backup_$(date +%Y%m%d_%H%M)"
export VERL_DIR

echo "[OPD Patch] Backing up original files to $PATCH_BACKUP_DIR"
mkdir -p "$PATCH_BACKUP_DIR"
cp "$VERL_DIR/verl/workers/engine_workers.py" "$PATCH_BACKUP_DIR/"
cp "$VERL_DIR/verl/workers/utils/losses.py" "$PATCH_BACKUP_DIR/"
echo "[OPD Patch] Backup done."

# ─────────────────────────────────────────────────────────────────
# Patch 1: engine_workers.py
#   - ActorRolloutRefWorker.__init__: load teacher model
#   - ActorRolloutRefWorker.update_actor: compute teacher log probs
#   - Add _compute_teacher_log_probs helper method
# ─────────────────────────────────────────────────────────────────

python3 - << 'PYEOF'
import os
import re

verl_dir = os.path.expanduser(os.environ.get("VERL_DIR", "verl"))
path = os.path.join(verl_dir, "verl/workers/engine_workers.py")
with open(path, 'r') as f:
    src = f.read()

# ── Patch 1a: inject teacher model loading after actor setup ──
# Find the line: self.actor.reset()
# and insert teacher loading after it
anchor = '        self.actor = TrainingWorker(config=actor_training_config)\n        self.actor.reset()'
insert = '''        self.actor = TrainingWorker(config=actor_training_config)
        self.actor.reset()

        # ── OPD: load teacher model for on-policy distillation ──
        self._teacher_model = None
        teacher_model_path = getattr(self.config.actor, 'teacher_model_path', None)
        if self._is_actor and teacher_model_path:
            import torch
            from transformers import AutoModelForCausalLM
            print(f"[OPD] Loading teacher model from {teacher_model_path} (float16)...")
            self._teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_path,
                torch_dtype=torch.float16,
                device_map='cuda',
                trust_remote_code=True,
            )
            self._teacher_model.eval()
            for p in self._teacher_model.parameters():
                p.requires_grad_(False)
            print(f"[OPD] Teacher model loaded. Params: {sum(p.numel() for p in self._teacher_model.parameters())/1e9:.2f}B")'''

if anchor in src:
    src = src.replace(anchor, insert, 1)
    print("Patch 1a applied: teacher model loading in __init__")
else:
    print("ERROR: Patch 1a anchor not found! Check the source.")
    exit(1)

# ── Patch 1b: modify update_actor to compute teacher log probs ──
anchor2 = '''    def update_actor(self, data: TensorDict) -> TensorDict:
        output = self.actor.train_mini_batch(data=data)
        return output.cpu() if output is not None else None'''

insert2 = '''    def update_actor(self, data: TensorDict) -> TensorDict:
        # ── OPD: compute teacher log probs for BC distillation ──
        if self._teacher_model is not None:
            data = self._compute_teacher_log_probs_bc(data)
        output = self.actor.train_mini_batch(data=data)
        return output.cpu() if output is not None else None

    def _compute_teacher_log_probs_bc(self, data: TensorDict) -> TensorDict:
        """Compute teacher BC log probs for on-policy distillation (single GPU).

        Teacher sees student's actual rollout tokens (on-policy).
        BC loss = -mean(log P_teacher(response_tokens)).

        Args:
            data: TensorDict in no-padding nested tensor format.

        Returns:
            data with 'teacher_log_probs' added (padded, shape (bsz, response_len)).
        """
        import torch
        import torch.nn.functional as F

        device = next(self._teacher_model.parameters()).device

        # ── Get padded input_ids ──
        input_ids_field = data.get("input_ids", None)
        if input_ids_field is None:
            return data  # safety: skip if no input_ids

        # nested tensor → padded tensor (no-padding mode uses nested tensors)
        if input_ids_field.is_nested:
            input_ids = input_ids_field.to_padded_tensor(padding=0)  # (bsz, seq_len)
        else:
            input_ids = input_ids_field  # already padded

        # ── Get responses tensor ──
        responses_field = data.get("responses", None)
        if responses_field is None:
            return data  # skip if no responses
        if responses_field.is_nested:
            responses = responses_field.to_padded_tensor(padding=0)  # (bsz, resp_len)
        else:
            responses = responses_field

        bsz, seq_len = input_ids.shape
        resp_len = responses.shape[1]
        prompt_len = seq_len - resp_len

        # Attention mask: 1 for non-pad, 0 for pad
        # In no-padding mode, input_ids is left-padded; pad token = 0 is rare in Qwen
        # but using attention_mask from data is safer if available
        attn_mask_field = data.get("attention_mask", None)
        if attn_mask_field is not None:
            if attn_mask_field.is_nested:
                attention_mask = attn_mask_field.to_padded_tensor(padding=0)
            else:
                attention_mask = attn_mask_field
        else:
            attention_mask = (input_ids != 0).long()

        # ── Teacher forward pass ──
        with torch.no_grad():
            teacher_logits = self._teacher_model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                use_cache=False,
            ).logits  # (bsz, seq_len, vocab_size)

            # Logits for response token prediction:
            # logits[:, prompt_len-1 : seq_len-1] predicts tokens at positions prompt_len..seq_len-1
            response_logits = teacher_logits[:, prompt_len - 1 : seq_len - 1, :]  # (bsz, resp_len, vocab_size)

            # Log probs
            teacher_log_probs = F.log_softmax(response_logits.float(), dim=-1)  # (bsz, resp_len, vocab_size)

            # Gather at actual response token ids
            resp_ids = responses.to(device).clamp(min=0)  # (bsz, resp_len) — clamp to avoid -1 pads
            teacher_log_probs = teacher_log_probs.gather(
                dim=-1, index=resp_ids.unsqueeze(-1)
            ).squeeze(-1)  # (bsz, resp_len)

        data["teacher_log_probs"] = teacher_log_probs.cpu()
        return data'''

if anchor2 in src:
    src = src.replace(anchor2, insert2, 1)
    print("Patch 1b applied: update_actor + _compute_teacher_log_probs_bc")
else:
    print("ERROR: Patch 1b anchor not found!")
    exit(1)

with open(path, 'w') as f:
    f.write(src)
print("engine_workers.py patched successfully.")
PYEOF

# ─────────────────────────────────────────────────────────────────
# Patch 2: losses.py
#   - ppo_loss: add BC loss term after KL loss block
# ─────────────────────────────────────────────────────────────────

python3 - << 'PYEOF'
import os

verl_dir = os.path.expanduser(os.environ.get("VERL_DIR", "verl"))
path = os.path.join(verl_dir, "verl/workers/utils/losses.py")
with open(path, 'r') as f:
    src = f.read()

# Find the return statement of ppo_loss and insert BC loss before it
anchor = '''    return policy_loss, metrics


def value_loss'''

insert = '''    # ── OPD: BC distillation loss (on-policy teacher) ──
    bc_loss_coef = getattr(config, 'bc_loss_coef', 0.0)
    if bc_loss_coef > 0 and "teacher_log_probs" in data:
        from verl.utils.dataset.dataset_utils import DatasetPadMode
        pad_mode_check = data.get("pad_mode", DatasetPadMode.PADDING)
        teacher_log_probs_raw = data["teacher_log_probs"]
        # teacher_log_probs_raw is already padded (bsz, resp_len) since we stored it that way
        bc_loss = -agg_loss(
            loss_mat=teacher_log_probs_raw,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )
        policy_loss = policy_loss + bc_loss_coef * bc_loss
        metrics["actor/bc_loss"] = Metric(value=bc_loss, aggregation=metric_aggregation)
        metrics["actor/bc_loss_coef"] = bc_loss_coef

    return policy_loss, metrics


def value_loss'''

if anchor in src:
    src = src.replace(anchor, insert, 1)
    print("Patch 2 applied: BC loss in ppo_loss")
else:
    print("ERROR: Patch 2 anchor not found! Check losses.py")
    exit(1)

with open(path, 'w') as f:
    f.write(src)
print("losses.py patched successfully.")
PYEOF

echo ""
echo "[OPD Patch] All patches applied!"
echo "[OPD Patch] Backup saved at: $PATCH_BACKUP_DIR"
echo ""
echo "Next steps:"
echo "  1. Upload models to server:"
echo "     - models/Qwen2.5-Coder-0.5B-Instruct"
echo "     - models/Qwen2.5-Coder-7B-Instruct"
echo "  2. Run SFT warmup on 0.5B:"
echo "     python <repo>/scripts/run_sft_teacher_warmup.py \\"
echo "         --base_model models/Qwen2.5-Coder-0.5B-Instruct \\"
echo "         --output_dir models/coder_0.5b_sft_warmup"
echo "  3. Run GRPO with teacher:"
echo "     TEACHER_MODEL_PATH=models/Qwen2.5-Coder-7B-Instruct \\"
echo "     bash <repo>/scripts/run_mbpp_v2_coder_0.5b_opd_grpo.sh"
