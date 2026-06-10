#!/usr/bin/env bash
set -euo pipefail

cd ~/test_ssj
source ~/miniconda3/etc/profile.d/conda.sh

if [ -d /dev/shm/DLCL ]; then
  conda activate /dev/shm/DLCL
elif conda env list | awk '{print $1}' | grep -qx "DLCL"; then
  conda activate DLCL
else
  echo "ERROR: DLCL environment not found." >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FULL_JSON="outputs/cnn_stego_def_paper_full_seed2026_learned_u_sparse/c100_v19/defense_result.json"
OUT_ROOT="outputs/cnn_stego_ablation_c100_v19_seed2026/stego"

mkdir -p "$OUT_ROOT/full_ours/c100_v19"

cp outputs/cnn_stego_def_paper_full_seed2026_learned_u_sparse/c100_v19/defense_result.* \
   "$OUT_ROOT/full_ours/c100_v19/" 2>/dev/null || true

cp outputs/cnn_stego_def_paper_full_seed2026_learned_u_sparse/c100_v19/defended_cnn_stego.pth \
   "$OUT_ROOT/full_ours/c100_v19/" 2>/dev/null || true

py_get() {
  python - "$FULL_JSON" "$1" "$2" <<'PY'
import json, sys
path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path, "r", encoding="utf-8"))
hits = []
def walk(x):
    if isinstance(x, dict):
        for k, v in x.items():
            if k == key and isinstance(v, (int, float, str, bool)):
                hits.append(v)
            walk(v)
    elif isinstance(x, list):
        for v in x:
            walk(v)
walk(d)
print(hits[-1] if hits else default)
PY
}

read SEL KPCT < <(python - "$FULL_JSON" <<'PY'
import json, sys, math
p = sys.argv[1]
d = json.load(open(p, "r", encoding="utf-8"))

def get_path(obj, path, default=None):
    cur = obj
    for x in path.split("."):
        if not isinstance(cur, dict) or x not in cur:
            return default
        cur = cur[x]
    return cur

def find_key(obj, key):
    vals = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == key and isinstance(v, (int, float)):
                    vals.append(v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(obj)
    return vals[-1] if vals else None

k = get_path(d, "after.inspect_k_actual", None)
if k is None:
    k = find_key(d, "inspect_k_actual")
if k is None:
    raise SystemExit("cannot find inspect_k_actual in final learned_u_sparse result")

total = find_key(d, "model_total_2d_params") or 14774436
sel = max(1, int(round(float(total) * float(k) / 100.0)))
print(sel, k)
PY
)

LARGE_SEL=$((SEL * 10))

Q_BITS=$(py_get q_bits 8)
BLOCK_SIZE=$(py_get block_size 32)
MIN_NUMEL=$(py_get min_numel 1024)
MIN_WRITABLE=$(py_get min_writable_positions 1024)
SCAN_BATCHES=$(py_get scan_max_batches 5)
FINAL_BATCHES=$(py_get final_max_batches 0)
PROBE_POS=$(py_get probe_positions 2048)
AUTO_SCORE=$(py_get auto_score_ratio 0.25)
AUTO_TENSORS=$(py_get auto_max_tensors 8)

LEARNED_STEPS=$(py_get learned_steps 80)
LEARNED_LR=$(py_get learned_lr 0.08)
LEARNED_TAU=$(py_get learned_tau 0.35)
LEARNED_LA=$(py_get learned_lambda_a 8.0)
LEARNED_LS=$(py_get learned_lambda_s 0.0)
LEARNED_LP=$(py_get learned_lambda_p 1.0)
LEARNED_LT=$(py_get learned_lambda_t 0.01)
LEARNED_LW=$(py_get learned_lambda_w 1e-4)
LEARNED_INIT=$(py_get learned_init_scale 0.02)
LEARNED_POOL_SQRT=$(py_get learned_pool_sqrt_scale 2.0)
LEARNED_MIN_POOL=$(py_get learned_min_pool 64)
LEARNED_MAX_POOL=$(py_get learned_max_pool_per_tensor 1024)
LEARNED_MAX_TOTAL=$(py_get learned_max_total_candidates 8192)
LEARNED_POS_DECAY=$(py_get learned_pos_decay 256.0)

echo "===== Final learned_u_sparse config inferred ====="
echo "SEL=$SEL"
echo "LARGE_SEL=$LARGE_SEL"
echo "KPCT=$KPCT"
echo "Q_BITS=$Q_BITS BLOCK_SIZE=$BLOCK_SIZE"
echo "LEARNED_LA=$LEARNED_LA LEARNED_LS=$LEARNED_LS LEARNED_LP=$LEARNED_LP LEARNED_LW=$LEARNED_LW"

common_args=(
  --model c100_v19
  --seed 2026
  --q-bits "$Q_BITS"
  --block-size "$BLOCK_SIZE"
  --min-numel "$MIN_NUMEL"
  --min-writable-positions "$MIN_WRITABLE"
  --scan-max-batches "$SCAN_BATCHES"
  --final-max-batches "$FINAL_BATCHES"
  --probe-positions "$PROBE_POS"
  --pai-floor 1e-8
  --auto-score-ratio "$AUTO_SCORE"
  --auto-max-tensors "$AUTO_TENSORS"
)

learned_common=(
  --budget-mode learned_u
  --learned-steps "$LEARNED_STEPS"
  --learned-lr "$LEARNED_LR"
  --learned-tau "$LEARNED_TAU"
  --learned-lambda-s "$LEARNED_LS"
  --learned-lambda-t "$LEARNED_LT"
  --learned-init-scale "$LEARNED_INIT"
  --learned-pool-sqrt-scale "$LEARNED_POOL_SQRT"
  --learned-min-pool "$LEARNED_MIN_POOL"
  --learned-max-pool-per-tensor "$LEARNED_MAX_POOL"
  --learned-max-total-candidates "$LEARNED_MAX_TOTAL"
  --learned-pos-decay "$LEARNED_POS_DECAY"
)

echo
echo "===== Run Random-SameK: random selected positions, same budget as Full Ours ====="
python -u cnn_stego_def_paper.py \
  "${common_args[@]}" \
  --out-dir "$OUT_ROOT/random_same_k" \
  --budget-mode fixed \
  --position-rule random \
  --per-tensor-edit-bits "$SEL" \
  --max-total-edit-bits "$SEL"

echo
echo "===== Run Random-LargeK: random selected positions, 10x budget ====="
python -u cnn_stego_def_paper.py \
  "${common_args[@]}" \
  --out-dir "$OUT_ROOT/random_large_k" \
  --budget-mode fixed \
  --position-rule random \
  --per-tensor-edit-bits "$LARGE_SEL" \
  --max-total-edit-bits "$LARGE_SEL"

echo
echo "===== Run w/o Attack Relevance: learned_lambda_a = 0 ====="
python -u cnn_stego_def_paper.py \
  "${common_args[@]}" \
  --out-dir "$OUT_ROOT/wo_attack_rel" \
  "${learned_common[@]}" \
  --learned-lambda-a 0 \
  --learned-lambda-p "$LEARNED_LP" \
  --learned-lambda-w "$LEARNED_LW"

echo
echo "===== Run w/o Utility Sensitivity: learned_lambda_p = 0, learned_lambda_w = 0 ====="
python -u cnn_stego_def_paper.py \
  "${common_args[@]}" \
  --out-dir "$OUT_ROOT/wo_utility_sens" \
  "${learned_common[@]}" \
  --learned-lambda-a "$LEARNED_LA" \
  --learned-lambda-p 0 \
  --learned-lambda-w 0

echo
echo "===== All stego ablation runs done ====="
