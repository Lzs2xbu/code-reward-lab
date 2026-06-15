# BC sweep chain: BC=0.02 → eval → BC=0.03 → eval
set -euo pipefail

SCRIPT_DIR=$HOME/codellmRL/scripts
LOG_DIR=$HOME/codellmRL

run_one() {
    local bc=$1
    local bc_tag=$2
    echo ""
    echo "============================================================"
    echo "[chain_bc_sweep] Starting BC=${bc} at $(date)"
    echo "============================================================"
    BC_LOSS_COEF=${bc} bash $SCRIPT_DIR/run_mbpp_v2_coder_1_5b_opd_bc_sweep.sh \
        2>&1 | tee $LOG_DIR/train_mbpp_v2_coder_1_5b_opd_bc${bc_tag}.log
    echo "[chain_bc_sweep] Training BC=${bc} done. Starting eval..."
    bash $SCRIPT_DIR/auto_eval_opd_bc_sweep.sh ${bc_tag} \
        2>&1 | tee $LOG_DIR/eval_opd_bc${bc_tag}.log
    echo "[chain_bc_sweep] BC=${bc} eval done at $(date)"
}

echo "[chain_bc_sweep] Starting BC sweep at $(date)"
echo "BC values: 0.02 → 0.03"

run_one 0.02 02
run_one 0.03 03

echo ""
echo "============================================================"
echo "[chain_bc_sweep] ALL DONE at $(date)"
echo "============================================================"
# 打印汇总
PYTHON=${PYTHON:-python}
$PYTHON -c "
import glob, json, os
out = os.path.expanduser(os.environ.get('EVAL_RESULTS_DIR', '<repo>/eval_results'))
refs = [
    ('BC=0 (Exp15)', 0.4533),
    ('BC=0.01 (Exp11c)', 0.4496),
    ('BC=0.05 (Exp11b)', 0.4599),
]
print('=== BC Sweep Final Summary ===')
for bc_tag in ['02', '03']:
    files = sorted(glob.glob(f'{out}/coder_1_5b_opd_bc{bc_tag}*.json'))
    for f in files:
        d = json.load(open(f))
        s = d.get('summary', d)
        print(f\"{s.get('label','?')}: pass@1={s.get('pass@1',0):.4f}\")
print('--- References ---')
for lbl, val in refs:
    print(f'{lbl}: pass@1={val:.4f}')
"
