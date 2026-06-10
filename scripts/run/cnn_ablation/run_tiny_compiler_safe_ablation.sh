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
export PYTHONPATH=~/platform_opt:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON_ARGS=(
  --model tiny_r34
  --candidate-rule balanced_union
  --seed 2026
  --opt-steps 80
  --auto-max-opt-positions
  --opt-search-max-batches 5
  --scomp-min-share 0.05
  --non-m1-tensor-cap-ratio 0.01
  --auto-scomp-param-ratio 0.12
  --auto-stego-tensor-ratio 0.024
  --pai-floor 1e-3
  --q-bits 8
  --block-size 32
  --compile-backend dlcl
)

echo "===== 1/3 Pure QDQ-only ====="
python -u cnn_compiler_def_paper.py \
  "${COMMON_ARGS[@]}" \
  --out-dir ~/test_ssj/outputs/cnn_compiler_ablation_tiny_r34_seed2026/compiler/pure_qdq_only \
  --lr 0.05 \
  --tau 999 \
  --lambda-a 0 \
  --lambda-p 1 \
  --lambda-w 1e-4 \
  --lambda-t 0.01

echo
echo "===== 2/3 w/o Attack Rel. ====="
python -u cnn_compiler_def_paper.py \
  "${COMMON_ARGS[@]}" \
  --out-dir ~/test_ssj/outputs/cnn_compiler_ablation_tiny_r34_seed2026/compiler/wo_attack_rel \
  --lr 0.05 \
  --tau 0.50 \
  --lambda-a 0 \
  --lambda-p 1 \
  --lambda-w 1e-4 \
  --lambda-t 0.01

echo
echo "===== 3/3 w/o Utility Sens. ====="
python -u cnn_compiler_def_paper.py \
  "${COMMON_ARGS[@]}" \
  --out-dir ~/test_ssj/outputs/cnn_compiler_ablation_tiny_r34_seed2026/compiler/wo_utility_sens \
  --lr 0.05 \
  --tau 0.50 \
  --lambda-a 1 \
  --lambda-p 0 \
  --lambda-w 0 \
  --lambda-t 0.01

echo
echo "===== quick check ====="
python - <<'PY'
import json
from pathlib import Path

paths = {
    "Pure QDQ-only": "outputs/cnn_compiler_ablation_tiny_r34_seed2026/compiler/pure_qdq_only/tiny_r34/defense_result.json",
    "w/o Attack Rel.": "outputs/cnn_compiler_ablation_tiny_r34_seed2026/compiler/wo_attack_rel/tiny_r34/defense_result.json",
    "w/o Utility Sens.": "outputs/cnn_compiler_ablation_tiny_r34_seed2026/compiler/wo_utility_sens/tiny_r34/defense_result.json",
    "Full Ours main": "outputs/cnn_compiler_def_paper_full_seed2026_balanced_union_auto/tiny_r34/defense_result.json",
}

for name, path in paths.items():
    p = Path(path)
    print("\n" + "=" * 90)
    print(name, p)
    d = json.load(open(p, "r", encoding="utf-8"))
    opt = d.get("inspect_opt", {})
    stage = "after"
    print("before_asr:", d["before"]["asr"])
    print("after_asr:", d[stage]["asr"])
    print("after_acc:", d[stage]["clean_accuracy"])
    print("selected:", d[stage].get("inspect_selected"))
    print("k:", d[stage].get("inspect_k_actual"))
    print("effective_max:", (opt.get("auto_budget") or {}).get("effective_max_opt_positions"))
    print("optimized:", opt.get("optimized"))
    print("lambda_a:", opt.get("lambda_a"))
    print("lambda_p:", opt.get("lambda_p"))
    print("lambda_w:", opt.get("lambda_w"))
    print("tau:", opt.get("tau"))
PY

echo
echo "===== All safe compiler ablations done ====="
