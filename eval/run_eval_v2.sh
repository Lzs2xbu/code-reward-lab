set -euo pipefail

PYTHON=${PYTHON:-python}
EVAL=$HOME/codellmRL/eval/eval_mbpp.py
TEST=$HOME/data/mbpp_v2/mbpp_test.parquet
OUT=$HOME/codellmRL/eval_results

echo "[1/3] Evaluating baseline (Qwen3-1.7B)..."
$PYTHON $EVAL \
    --model_path $HOME/models/Qwen3-1.7B \
    --test_file $TEST \
    --output_file $OUT/v2_baseline.json \
    --n_samples 20 --temperature 0.8 --label v2_baseline

echo "[2/3] Evaluating binary reward step80..."
$PYTHON $EVAL \
    --model_path $HOME/models/eval_merged/binary_1_7b_step80 \
    --test_file $TEST \
    --output_file $OUT/v2_binary_step80.json \
    --n_samples 20 --temperature 0.8 --label v2_binary_step80

echo "[3/3] Evaluating partial reward step80..."
$PYTHON $EVAL \
    --model_path $HOME/models/eval_merged/partial_1_7b_step80 \
    --test_file $TEST \
    --output_file $OUT/v2_partial_step80.json \
    --n_samples 20 --temperature 0.8 --label v2_partial_step80

echo "=== ALL DONE ==="
