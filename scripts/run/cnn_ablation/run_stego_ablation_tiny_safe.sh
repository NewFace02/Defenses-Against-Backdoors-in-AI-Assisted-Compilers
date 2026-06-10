#!/usr/bin/env bash
set -euo pipefail

cd ~/test_ssj
source ~/miniconda3/etc/profile.d/conda.sh

if [ -d /dev/shm/DLCL ]; then
  conda activate /dev/shm/DLCL
else
  conda activate DLCL
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FULL_JSON="outputs/cnn_stego_def_paper_full_seed2026_learned_u_sparse/tiny_r34/defense_result.json"

if [ ! -f "$FULL_JSON" ]; then
  echo "ERROR: Tiny-R34 Full Ours stego result not found:"
  echo "$FULL_JSON"
  echo "先补 Tiny-R34 隐写 Full Ours 主实验，再跑消融。"
  exit 2
fi

SEL=$(python - <<'PY'
import json
p="outputs/cnn_stego_def_paper_full_seed2026_learned_u_sparse/tiny_r34/defense_result.json"
d=json.load(open(p, "r", encoding="utf-8"))
v=d.get("after",{}).get("inspect_selected")
if v is None:
    v=d.get("inspect_opt",{}).get("selected")
if v is None:
    raise SystemExit("missing inspect_selected in Full Ours json")
print(int(v))
PY
)

LARGE_SEL=$((SEL * 10))

echo "===== Tiny-R34 stego ablation ====="
echo "Full Ours JSON: $FULL_JSON"
echo "SEL=$SEL"
echo "LARGE_SEL=$LARGE_SEL"

echo
echo "===== 1/4 Random-SameK ====="
python -u cnn_stego_def_paper.py \
  --model tiny_r34 \
  --out-dir ~/test_ssj/outputs/cnn_stego_ablation_tiny_r34_seed2026/stego/random_same_k \
  --seed 2026 \
  --budget-mode fixed \
  --position-rule random \
  --per-tensor-edit-bits "$SEL" \
  --max-total-edit-bits "$SEL" \
  --q-bits 8 \
  --block-size 32 \
  --scan-max-batches 5 \
  --final-max-batches 0

echo
echo "===== 2/4 Random-LargeK ====="
python -u cnn_stego_def_paper.py \
  --model tiny_r34 \
  --out-dir ~/test_ssj/outputs/cnn_stego_ablation_tiny_r34_seed2026/stego/random_large_k \
  --seed 2026 \
  --budget-mode fixed \
  --position-rule random \
  --per-tensor-edit-bits "$LARGE_SEL" \
  --max-total-edit-bits "$LARGE_SEL" \
  --q-bits 8 \
  --block-size 32 \
  --scan-max-batches 5 \
  --final-max-batches 0

echo
echo "===== 3/4 w/o Attack Rel. ====="
python -u cnn_stego_def_paper.py \
  --model tiny_r34 \
  --out-dir ~/test_ssj/outputs/cnn_stego_ablation_tiny_r34_seed2026/stego/wo_attack_rel \
  --seed 2026 \
  --budget-mode learned_u \
  --position-rule prefix \
  --q-bits 8 \
  --block-size 32 \
  --scan-max-batches 5 \
  --final-max-batches 0 \
  --learned-steps 80 \
  --learned-lr 0.08 \
  --learned-tau 0.35 \
  --learned-lambda-a 0 \
  --learned-lambda-p 1 \
  --learned-lambda-t 0.01 \
  --learned-lambda-w 1e-4

echo
echo "===== 4/4 w/o Utility Sens. ====="
python -u cnn_stego_def_paper.py \
  --model tiny_r34 \
  --out-dir ~/test_ssj/outputs/cnn_stego_ablation_tiny_r34_seed2026/stego/wo_utility_sens \
  --seed 2026 \
  --budget-mode learned_u \
  --position-rule prefix \
  --q-bits 8 \
  --block-size 32 \
  --scan-max-batches 5 \
  --final-max-batches 0 \
  --learned-steps 80 \
  --learned-lr 0.08 \
  --learned-tau 0.35 \
  --learned-lambda-a 8 \
  --learned-lambda-p 0 \
  --learned-lambda-t 0.01 \
  --learned-lambda-w 0

echo
echo "===== quick check ====="
python - <<'PY'
import json
from pathlib import Path

paths = {
    "Full Ours": "outputs/cnn_stego_def_paper_full_seed2026_learned_u_sparse/tiny_r34/defense_result.json",
    "Random-SameK": "outputs/cnn_stego_ablation_tiny_r34_seed2026/stego/random_same_k/tiny_r34/defense_result.json",
    "Random-LargeK": "outputs/cnn_stego_ablation_tiny_r34_seed2026/stego/random_large_k/tiny_r34/defense_result.json",
    "w/o Attack Rel.": "outputs/cnn_stego_ablation_tiny_r34_seed2026/stego/wo_attack_rel/tiny_r34/defense_result.json",
    "w/o Utility Sens.": "outputs/cnn_stego_ablation_tiny_r34_seed2026/stego/wo_utility_sens/tiny_r34/defense_result.json",
}

for name, path in paths.items():
    p=Path(path)
    print("\n" + "="*100)
    print(name, p)
    if not p.exists():
        print("[MISS]")
        continue
    d=json.load(open(p, "r", encoding="utf-8"))
    print("before_acc:", d.get("before",{}).get("clean_accuracy"))
    print("after_acc:", d.get("after",{}).get("clean_accuracy"))
    print("selected:", d.get("after",{}).get("inspect_selected"))
    print("k:", d.get("after",{}).get("inspect_k_actual"))
    print("before_match:", d.get("before",{}).get("match"))
    print("after_match:", d.get("after",{}).get("match"))
    print("before_exact:", d.get("before",{}).get("extract_exact_match"))
    print("after_exact:", d.get("after",{}).get("extract_exact_match"))
    print("before_bit_errors:", d.get("before",{}).get("bit_errors"))
    print("after_bit_errors:", d.get("after",{}).get("bit_errors"))
    print("budget_mode:", d.get("budget_mode", d.get("defense",{}).get("budget_mode")))
    print("position_rule:", d.get("position_rule", d.get("defense",{}).get("position_rule")))
PY

echo
echo "===== All Tiny-R34 stego ablations done ====="
