# Code LLM RL Scripts

This repository collects experiment scripts for code-generation reinforcement learning with veRL, GRPO, SFT warmup, OPD variants, MBPP, APPS, and LiveCodeBench evaluation.

The scripts were produced during a single-GPU research workflow. They are meant to be a reproducible starting point for data preparation, reward functions, training launches, and vLLM-based evaluation rather than a polished framework package.

## What Is Included

- `data/`: dataset conversion and teacher-signal preprocessing scripts.
- `rewards/`: sandboxed reward functions for MBPP and LiveCodeBench-style code tasks.
- `scripts/`: training launchers, OPD/GRPO sweep scripts, SFT warmup scripts, and veRL patch helpers.
- `eval/`: vLLM evaluation scripts and convenience launchers.
- `docs/`: selected reusable experiment notes.

Generated datasets, model checkpoints, logs, local workspace metadata, PDFs, connection logs, and private notes are intentionally ignored.

## Repository Map

| Area | Main files | Purpose |
| --- | --- | --- |
| MBPP prep | `data/prepare_mbpp.py`, `data/prepare_mbpp_v2.py`, `data/prepare_mbpp_opd_cl.py` | Build veRL-compatible MBPP parquet files with function-name injection and OPD/CL variants. |
| APPS/LCB prep | `data/prepare_apps.py`, `data/prepare_lcb.py` | Convert APPS and LiveCodeBench-style data into training/eval tables. |
| Teacher signals | `data/precompute_teacher_logprobs.py`, `scripts/prepare_mbpp_1_5b_teacher_sft.py`, `scripts/prepare_apps_teacher_sft.py` | Generate teacher responses, teacher-pass SFT rows, or token-level teacher log-probs. |
| Rewards | `rewards/mbpp_reward.py`, `rewards/mbpp_reward_opd.py`, `rewards/lcb_reward.py` | Execute generated code in a constrained subprocess and return binary/partial rewards. |
| Training | `scripts/run_mbpp_*.sh`, `scripts/run_apps_*.sh`, `scripts/run_arm_e_v2_chain.sh` | Launch veRL GRPO, OPD, SFT-warmup, APPS teacher-SFT, and MBPP Arm A/B/D experiments. |
| Evaluation | `eval/eval_mbpp.py`, `eval/eval_lcb.py`, `eval/run_eval_v2.sh`, `scripts/eval_lcb_teacher_sft.sh` | Run vLLM generation and compute pass@k-style metrics. |
| veRL patches | `scripts/apply_verl_opd_patch.sh`, `scripts/patch_arm_cd.py`, `docs/verl_patch_inventory.md` | Apply or audit local OPD-related modifications to a veRL checkout. |

## Environment

Install the Python dependencies in `requirements.txt`, then install veRL from its upstream repository according to the official veRL documentation.

```bash
pip install -r requirements.txt
```

Typical runtime assumptions:

- A CUDA host with PyTorch, vLLM, transformers, and veRL available.
- A local veRL checkout, commonly at `$VERL_DIR`.
- Training/evaluation parquet files under `$HOME/data/...`.
- Model checkpoints under `$HOME/models/...`.

Shell launchers use environment-variable defaults such as `${PYTHON:-python}`, `${RAY_BIN:-ray}`, `${VERL_DIR:-verl}`, `${MODEL_DIR:-$HOME/models}`, and `${DATA_DIR:-$HOME/data}`. Override them before running on a new machine.

## Common Workflows

Prepare MBPP data:

```bash
python data/prepare_mbpp.py --output_dir data/mbpp_v2
```

Run MBPP evaluation:

```bash
python eval/eval_mbpp.py \
  --model_path models/coder_1_5b_sft_warmup/final \
  --test_file data/mbpp_v2/mbpp_test.parquet \
  --output_file eval_results/mbpp_eval.json
```

Run an example GRPO training launcher:

```bash
bash scripts/run_mbpp_v2_coder_1_5b_grpo_lr5e6.sh
```

## Experiment Notes

The main experiment record is in `docs/experiment_report.md`. The shortest project summary is:

- Teacher SFT data quality was more important than simply swapping in an OPD variant.
- SFT warmup plus GRPO was the strongest LiveCodeBench path in these experiments.
- OPD as an actor-update loss was stable, while OPD as shaped reward could destabilize rollout selection.
- Single-GPU top-K token reward variants were limited by backward memory cost, so K=1 sampled-token approximations were used.

## Safety Notes

Reward functions execute model-generated Python in subprocesses with time and resource limits, but they are not a production sandbox. Run evaluation code only in an isolated environment.

Before making the repository public, review:

```bash
git status --short
git ls-files
```

Confirm that no private datasets, checkpoints, logs, credentials, or personal documents were added.
