# Online Policy Distillation (OPD) 实现问题记录

记录 Exp 9（Qwen2.5-Coder-0.5B 学生 + 7B 教师在线蒸馏）开发过程中遇到的所有问题及解决方案。

---

## Issue 1：vmap OOM — transformers 4.57.6 causal mask 内存爆炸

**发现时间：** 2026-06-01

**现象：**
```
torch.cuda.OutOfMemoryError: Tried to allocate 22.47 GiB
```
错误栈在 `transformers/masking_utils.py → _vmap_for_bhqkv`

**根因：**
transformers ≥ 4.50 重写了 `create_causal_mask`，对长序列（max_len=1024）通过 `_vmap_for_bhqkv` 批量生成 (B, H, q, kv_max_pos) 形状的 mask tensor，在 7B 模型（32头）上峰值分配近 22GB。

**解决方案：**
传入 4D additive float16 mask（shape `(B, 1, S, S)`）。`_preprocess_mask_arguments` 检测到 `mask.ndim == 4` 会提前返回，完全绕过 vmap 路径。

```python
_B, _S = chunk_ids.shape
_causal = torch.tril(torch.ones(_S, _S, dtype=torch.bool, device=device))
_pad = chunk_mask.bool()                                          # (B, S)
_valid = (_causal.unsqueeze(0) & _pad.unsqueeze(1)).unsqueeze(1) # (B, 1, S, S)
_attn_mask_4d = torch.zeros(_B, 1, _S, _S, dtype=torch.float16, device=device)
_attn_mask_4d.masked_fill_(~_valid, torch.finfo(torch.float16).min)
logits = teacher_model(input_ids=chunk_ids, attention_mask=_attn_mask_4d, use_cache=False).logits
```

**修改文件：** `$VERL_DIR/verl/workers/engine_workers.py::_compute_teacher_log_probs_bc`

---

## Issue 2：expandable_segments 与 vLLM 不兼容

**发现时间：** 2026-06-01

**现象：**
```
AssertionError  # vLLM 初始化时崩溃
```
设置了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**根因：**
vLLM 内部使用 CUDA caching allocator 的特定行为假设，与 `expandable_segments` 模式不兼容。

**解决方案：**
从训练脚本中移除 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

---

## Issue 3：input_ids packed format 导致 shape 不匹配

**发现时间：** 2026-06-02

**现象：**
```
RuntimeError: The size of tensor a (77034) must match the size of tensor b (1024)
at non-singleton dimension 2
```
training 在 step 1 第一次 `update_actor` 时 crash。

**根因（分析过程）：**

veRL 在 `ray_trainer.py::_update_actor` 调用 actor worker 前执行 `left_right_2_no_padding(batch_td)`，该函数将 `input_ids` 从 padded 格式 `(B, max_seq_len)` 转为 jagged nested tensor。

旧代码在 `_compute_teacher_log_probs_bc` 中用 `input_ids_field.to_padded_tensor(0)` 恢复 padded 格式，但 `to_padded_tensor()` 只 pad 到 batch 内最长实际序列长度（如 900），而非原始 `max_seq_len=1024`。

`attention_mask` 仍然是原始 `(B, 1024)` 格式（`left_right_2_no_padding` 读取但未 pop），与恢复后的 `input_ids (B, 900)` 在 4D mask broadcast 时 shape 不一致：
```
(1, 900, 900) & (B, 1, 1024)  → dim 2: 900 ≠ 1024 → ERROR
```

**解决方案：**
使用 `pad_input(flat_tokens, indices, B, max_seq_len)` 精确还原到原始 `(B, max_seq_len)` 格式。`indices` 和 `max_seq_len` 由 `left_right_2_no_padding` 存入 TensorDict 的 NonTensorData。

```python
from verl.utils.attention_utils import pad_input
from verl.utils import tensordict_utils as tu

indices = tu.get_non_tensor_data(data, "indices", None)
max_seq_len_val = tu.get_non_tensor_data(data, "max_seq_len", None)

if indices is not None and max_seq_len_val is not None:
    flat_tokens = input_ids_field.values() if input_ids_field.is_nested \
                  else input_ids_field.reshape(-1)[:indices.shape[0]]
    B = attn_mask_field.shape[0]
    # pad_input: (total_nnz, 1) -> (B, max_seq_len, 1) -> squeeze -> (B, max_seq_len)
    input_ids = pad_input(flat_tokens.unsqueeze(-1), indices, B, max_seq_len_val).squeeze(-1)
```

**修改文件：** `$VERL_DIR/verl/workers/engine_workers.py::_compute_teacher_log_probs_bc`

**验证：**
修复后 `[OPD teacher] input_ids=torch.Size([512, 1024])` 正常出现，shape 匹配。

---

## Issue 4：ppo_loss 的 `data.select(*fields)` 丢弃 teacher_log_probs

**发现时间：** 2026-06-02（Issue 3 修复后随即发现）

**现象：**
`actor/bc_loss` 不在训练指标中出现（后来发现是 NaN，见 Issue 5）

**根因：**
`ppo_loss` 在计算 loss 前执行：
```python
fields = ["response_mask", "old_log_probs", "advantages", ...]
data = data.select(*fields).to_padded_tensor()
```
`teacher_log_probs` 不在 `fields` 列表中，被默默丢弃。

**解决方案：**
在 `fields` 构建后动态追加：
```python
# OPD: preserve teacher_log_probs for BC distillation loss
if "teacher_log_probs" in data:
    fields.append("teacher_log_probs")
data = data.select(*fields).to_padded_tensor()
```

**修改文件：** `$VERL_DIR/verl/workers/utils/losses.py::ppo_loss`

---

## Issue 5：bc_loss = NaN — teacher fp16 logits overflow

**发现时间：** 2026-06-02（训练 step 1-2 指标中观察到）

**现象：**
```
actor/bc_loss: nan
actor/loss: nan
```
但 `actor/grad_norm: 0.459`（非 NaN），GRPO 部分仍正常更新。

**根因分析：**

Qwen2.5-7B-Instruct 以 float16 推理时，某些 token position 的 logits 绝对值超过 fp16 最大值（65504），溢出为 `inf`。

当某行所有 logits 均为 `inf` 时：
```
F.log_softmax([inf, inf, ..., inf]) = log(exp(inf) / sum(exp(inf))) = log(inf/inf) = NaN
```
产生 NaN log-probs，最终导致 `bc_loss = NaN`。

**为何 grad_norm 仍正常？**
`teacher_log_probs` 是通过 `torch.no_grad()` 计算的常量张量，不在 model 的计算图上。
PyTorch autograd 计算 `d(pg_loss + 0.1 * bc_loss_const) / d(params) = d(pg_loss)/d(params)`，
NaN 不会反向传播到参数梯度。所以 GRPO 仍然工作，但蒸馏信号完全失效。

**解决方案：**
在 log_softmax 前先 cast 到 float32：

```python
# Before (buggy - fp16 overflow causes NaN):
log_probs = F.log_softmax(resp_logits, dim=-1)  # fp16

# After (fixed - fp32 avoids overflow):
log_probs = F.log_softmax(resp_logits.float(), dim=-1)  # fp32
```

内存影响：micro_bsz=8, resp_len=512, vocab=152064 时 fp32 tensor = 2.5GB，
GPU 有 27GB 空闲，可安全使用。

**修改文件：** `$VERL_DIR/verl/workers/engine_workers.py::_compute_teacher_log_probs_bc`

**注意：** 此修复需重启训练才能生效（运行中的 Python 进程不重新加载代码）。

---

## 关键经验总结

1. **veRL packed format 陷阱：** `left_right_2_no_padding` 后，任何需要 per-sequence 原始 token 的操作都必须用 `pad_input(values, indices, B, max_seq_len)` 还原，而非 `to_padded_tensor()`（后者 pad 到 batch 内最大长度，不等于配置的 max_seq_len）。

2. **fp16 推理的 NaN 风险：** 大模型（7B+）fp16 推理在 log_softmax 时容易 overflow。任何 log_softmax 计算都应先 cast 到 fp32，gather 后再转回所需精度。

3. **NaN bc_loss 不影响 GRPO 梯度：** 因 teacher log-probs 是 detached 常量，bc_loss 的 NaN 不会污染 GRPO 梯度。但蒸馏信号确实失效，实验结论无效。

4. **data.select() 会静默丢弃字段：** veRL ppo_loss 里的 `data.select(*fields)` 是一个"白名单"操作，任何未列出的字段（包括自定义的 teacher_log_probs）都会被丢弃。

---

## 当前代码修改状态（2026-06-02）

| 文件 | 修改内容 | Issue |
|------|----------|-------|
| `engine_workers.py` | 4D additive mask 绕过 vmap | #1 |
| `engine_workers.py` | pad_input 重建 input_ids | #3 |
| `engine_workers.py` | fp32 log_softmax | #5 ✓(已修复，Exp 9b 重启生效) |
| `losses.py` | fields 中保留 teacher_log_probs | #4 |
| `run_mbpp_v2_coder_0.5b_opd_grpo.sh` | 移除 expandable_segments | #2 |
| `auto_eval_after_train.sh` | checkpoint 路径 verl_grpo_mbpp → verl_grpo_mbpp_v2 | #6 |


---

## Issue 6: auto_eval_after_train.sh checkpoint path wrong

**Discovered:** 2026-06-02 (watcher triggered eval after training crash)

**Symptom:**
Script failed with: ERROR: No step checkpoints found in .../checkpoints/verl_grpo_mbpp/coder_0.5b_...

**Root cause:**
Script hardcoded `verl_grpo_mbpp` as checkpoint base dir, but 0.5B experiment
uses `verl_grpo_mbpp_v2`.

**Fix:** Changed CKPT_BASE in auto_eval_after_train.sh:
  Old: CKPT_BASE=$HOME/verl/checkpoints/verl_grpo_mbpp/$EXP_NAME
  New: CKPT_BASE=$HOME/verl/checkpoints/verl_grpo_mbpp_v2/$EXP_NAME

**File modified:** `<repo>/scripts/auto_eval_after_train.sh` (fixed 2026-06-02)

---

## Issue 5 Addendum: NaN bc_loss causes cascade OOM at step 11

**Discovered:** 2026-06-02

**Symptom:** Training crashed at step 11 with CUDA OOM:
  torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 9.41 GiB.
  GPU 0 has total 44.52 GiB, only 8.04 GiB free.
  Crash in: torch/nested/_internal/ops.py:like_factory_default (during loss.backward())

**Root cause:**
Issue 5 NaN bc_loss makes total policy_loss = pg_loss + 0.1*NaN = NaN.
During loss.backward(), PyTorch nested tensor backward path (like_factory_default)
allocates a huge 9.41 GiB temporary tensor for the NaN scalar, causing OOM.

**Key insight:** Although policy_loss=NaN and the training log shows actor/loss=NaN,
GRPO gradients are still valid (bc_loss is a detached constant -- teacher log-probs
have no grad path through student parameters). The model was effectively trained with
GRPO-only signal, just with inflated memory usage from the NaN backward.

**Impact on experiment:**
- Exp 9a (20260602_1603): crashed at step 11 of 80 planned steps -- invalid OPD run
- Step 10 val-core=0.2956, GRPO working, but distillation signal = 0 throughout
- Decision: Restart as Exp 9b applying the fp32 fix from the start

**Prevention:** Issue 5 fix (cast logits to fp32 before log_softmax) eliminates NaN,
preventing both the silent distillation failure and this cascade OOM.


---

## Issue 7: teacher 推完后仍留在 GPU，student backward OOM（**Exp 9c，2026-06-02**）

**错误**：

位置：

**根因**：
 中调用  后，7B 教师模型仍留在 GPU（占 ~14 GB）。随即调用  做学生 backward，需要额外 9.53 GB，而 GPU 只剩 8 GB → OOM。

**修复（engine_workers.py）**：


**步骤**：
- Exp 9a：step 11 crash（bc_loss=NaN）
- Exp 9b：step 3 crash（bc_loss=NaN）
- Exp 9c：step 9 crash（本 bug，bc_loss 正常但 teacher 占显存）
- Exp 9d：本修复生效后启动

---

## Issue 7: teacher 推完后仍留在 GPU，student backward OOM（Exp 9c，2026-06-02）

**根因**：`update_actor` 调用 `_compute_teacher_log_probs_bc` 后，7B 教师仍在 GPU（占 ~14 GB），随即做学生 backward 需 9.53 GB，只剩 8 GB → OOM。

**修复**（engine_workers.py update_actor）：teacher forward 后立即 `.cpu() + empty_cache()`，再调用 `train_mini_batch`。

**历史**：9a(step11 NaN-OOM) → 9b(step3 NaN-OOM) → 9c(step9 本bug) → 9d(本修复)
