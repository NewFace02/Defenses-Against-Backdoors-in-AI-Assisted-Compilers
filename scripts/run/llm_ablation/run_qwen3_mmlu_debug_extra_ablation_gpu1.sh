#!/usr/bin/env bash
set -euo pipefail

cd ~/test_myx
source ~/miniconda3/etc/profile.d/conda.sh
conda activate saser-chatglm3

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export HF_HOME=/dev/shm/hf_home_runtime
export HF_DATASETS_CACHE=/dev/shm/hf_home_runtime/datasets
export HF_DATASETS_OFFLINE=1
unset HF_ENDPOINT

MODEL_PATH="$HOME/test_myx/models/qwen3-8b"
ATTACK_JSON="$HOME/test_myx/outputs/qwen3_attack_mmlu_debug_seed2026/attack_result.json"
PAI_CSV="$HOME/test_myx/outputs/qwen3_def_mmlu_debug2_seed2026/inspect_llm_stego_pai.csv"
FULL_JSON="$HOME/test_myx/outputs/qwen3_def_mmlu_debug2_seed2026/defense_result.json"
BASE_OUT="$HOME/test_ssj/llm_ablation/outputs/qwen3_mmlu_debug_ablation_seed2026"

SUBJECTS="abstract_algebra,computer_security,high_school_mathematics,logical_fallacies"

for f in "$ATTACK_JSON" "$PAI_CSV" "$FULL_JSON"; do
  if [ ! -f "$f" ]; then
    echo "ERROR missing file: $f"
    exit 2
  fi
done

BASE_K=$(python - <<'PY'
import json
p="/home/songyq/test_myx/outputs/qwen3_def_mmlu_debug2_seed2026/defense_result.json"
d=json.load(open(p))
print(int(d["inspect_opt"]["changed_params"]))
PY
)

echo "===== Qwen3 extra ablations on GPU1 ====="
echo "BASE_K=$BASE_K"

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --attack-result-json "$ATTACK_JSON"
  --eval-dataset mmlu
  --mmlu-subjects "$SUBJECTS"
  --mmlu-split test
  --max-calib 40
  --max-eval 200
  --calib-ratio 0.1
  --q-bits 4
  --stego-pai-source csv
  --target-search-csv "$PAI_CSV"
  --candidate-layers union
  --comp-top-hidden-dims 1
  --max-scan-layers 36
  --opt-steps 30
  --opt-calib-limit 32
  --opt-calib-batch 1
  --lr 0.05
  --lambda-a 1.0
  --lambda-p 1.0
  --lambda-w 0.0001
  --lambda-t 0.01
  --tau 0.5
  --seed 2026
  --device-map none
)

rm -rf "$BASE_OUT/random_same_k" "$BASE_OUT/random_large_k" "$BASE_OUT/wo_quant_safety"
mkdir -p "$BASE_OUT/random_same_k" "$BASE_OUT/random_large_k" "$BASE_OUT/wo_quant_safety"

echo
echo "===== 1/3 Random-SameK ====="
python -u qwen3_def_paper_ablation_safe.py \
  "${COMMON_ARGS[@]}" \
  --comp-top-layers 1 \
  --stego-top-layers 1 \
  --ablation-mode random_same_k \
  --random-selected-count "$BASE_K" \
  --out-dir "$BASE_OUT/random_same_k"

echo
echo "===== 2/3 Expanded Random-LargeK ====="
python -u qwen3_def_paper_ablation_safe.py \
  "${COMMON_ARGS[@]}" \
  --comp-top-layers 4 \
  --stego-top-layers 4 \
  --ablation-mode random_large_k \
  --random-selected-count "$BASE_K" \
  --random-large-mult 10 \
  --out-dir "$BASE_OUT/random_large_k"

echo
echo "===== 3/3 w/o Quant-Safety ====="
python -u qwen3_def_paper_ablation_safe.py \
  "${COMMON_ARGS[@]}" \
  --comp-top-layers 1 \
  --stego-top-layers 1 \
  --disable-quant-safety \
  --out-dir "$BASE_OUT/wo_quant_safety"

echo
echo "===== quick extra summary ====="
python - <<'PY'
import json
from pathlib import Path

for name in ["random_same_k", "random_large_k", "wo_quant_safety"]:
    p=Path(f"/home/songyq/test_ssj/llm_ablation/outputs/qwen3_mmlu_debug_ablation_seed2026/{name}/defense_result.json")
    print("\n" + "="*100)
    print(name, p)
    if not p.exists():
        print("[MISS]")
        continue
    d=json.load(open(p))
    opt=d.get("inspect_opt", {})
    b=d.get("before_defense", {})
    a=d.get("after_defense", {})
    print("candidate_layers:", d.get("candidate_layers"))
    print("before_asr:", b.get("asr_percent"))
    print("after_asr:", a.get("asr_percent"))
    print("before_ber:", b.get("ber_percent"))
    print("after_ber:", a.get("ber_percent"))
    print("before_match:", b.get("byte_match_percent"))
    print("after_match:", a.get("byte_match_percent"))
    print("after_acc:", a.get("acc_percent"))
    print("after_ppl:", a.get("ppl"))
    print("changed_params:", opt.get("changed_params"))
    print("k_model_percent:", opt.get("k_star_model_percent"))
    print("ablation_mode:", opt.get("ablation_mode"))
    print("disable_quant_safety:", opt.get("disable_quant_safety"))
PY
