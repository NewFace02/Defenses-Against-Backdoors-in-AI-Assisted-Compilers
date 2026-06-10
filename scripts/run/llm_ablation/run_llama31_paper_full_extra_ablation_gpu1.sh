#!/usr/bin/env bash
set -euo pipefail

cd ~/test_myx
source ~/miniconda3/etc/profile.d/conda.sh
conda activate saser-chatglm3

GPU_ID=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export HF_HOME=/dev/shm/hf_home_runtime
export HF_DATASETS_CACHE=/dev/shm/hf_home_runtime/datasets
export HF_DATASETS_OFFLINE=1
unset HF_ENDPOINT

MODEL_PATH="$HOME/test_myx/models/llama31-8b"
ATTACK_JSON="$HOME/test_myx/outputs/llama31_attack_paper_full_seed2026/attack_result.json"
FULL_DEF_DIR="$HOME/test_myx/outputs/llama31_def_paper_full_seed2026"
PAI_CSV="$FULL_DEF_DIR/inspect_llm_stego_pai.csv"
BASE_OUT="$HOME/test_ssj/llm_ablation/outputs/llama31_paper_full_ablation_seed2026"

for f in "$ATTACK_JSON" "$PAI_CSV" "$FULL_DEF_DIR/defense_result.json"; do
  if [ ! -f "$f" ]; then
    echo "ERROR missing file: $f"
    exit 2
  fi
done

if [ ! -f "$HOME/test_myx/llama31_def_paper_ablation_safe.py" ]; then
  echo "ERROR missing llama31_def_paper_ablation_safe.py"
  exit 3
fi

if ! grep -q "ablation-mode" "$HOME/test_myx/llama31_def_paper_ablation_safe.py"; then
  echo "ERROR llama31_def_paper_ablation_safe.py has no random ablation patch"
  exit 4
fi

if ! grep -q "disable-quant-safety" "$HOME/test_myx/llama31_def_paper_ablation_safe.py"; then
  echo "ERROR llama31_def_paper_ablation_safe.py has no quant-safety ablation patch"
  exit 5
fi

read MMLU_SUBJECTS AGIEVAL_CONFIGS BASE_K < <(python - <<'PY'
import json
attack="/home/songyq/test_myx/outputs/llama31_attack_paper_full_seed2026/attack_result.json"
defense="/home/songyq/test_myx/outputs/llama31_def_paper_full_seed2026/defense_result.json"
a=json.load(open(attack))
d=json.load(open(defense))
meta=a.get("dataset_meta", {})
base_k=int(d.get("inspect_opt", {}).get("changed_params"))
print(meta.get("mmlu_subjects", "all"), meta.get("agieval_configs", "all"), base_k)
PY
)

echo "===== Llama31 paper full extra ablations on GPU${GPU_ID} ====="
echo "MMLU_SUBJECTS=$MMLU_SUBJECTS"
echo "AGIEVAL_CONFIGS=$AGIEVAL_CONFIGS"
echo "BASE_K=$BASE_K"
echo "PAI_CSV=$PAI_CSV"
echo "BASE_OUT=$BASE_OUT"

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --attack-result-json "$ATTACK_JSON"
  --eval-dataset paper
  --mmlu-subjects "$MMLU_SUBJECTS"
  --mmlu-split test
  --agieval-configs "$AGIEVAL_CONFIGS"
  --agieval-split validation
  --calib-ratio 0.1
  --max-calib 0
  --max-eval 0
  --q-bits 4
  --stego-pai-source csv
  --target-search-csv "$PAI_CSV"
  --candidate-layers union
  --comp-top-hidden-dims 1
  --max-scan-layers 32
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

mkdir -p "$BASE_OUT"
rm -rf "$BASE_OUT/random_same_k" "$BASE_OUT/random_large_k" "$BASE_OUT/wo_quant_safety"
mkdir -p "$BASE_OUT/random_same_k" "$BASE_OUT/random_large_k" "$BASE_OUT/wo_quant_safety"

echo
echo "===== 1/3 Random-SameK ====="
python -u llama31_def_paper_ablation_safe.py \
  "${COMMON_ARGS[@]}" \
  --comp-top-layers 1 \
  --stego-top-layers 1 \
  --ablation-mode random_same_k \
  --random-selected-count "$BASE_K" \
  --out-dir "$BASE_OUT/random_same_k"

echo
echo "===== 2/3 Expanded Random-LargeK ====="
python -u llama31_def_paper_ablation_safe.py \
  "${COMMON_ARGS[@]}" \
  --comp-top-layers 4 \
  --stego-top-layers 4 \
  --ablation-mode random_large_k \
  --random-selected-count "$BASE_K" \
  --random-large-mult 10 \
  --out-dir "$BASE_OUT/random_large_k"

echo
echo "===== 3/3 w/o Quant-Safety ====="
python -u llama31_def_paper_ablation_safe.py \
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

paths = {
    "Full Ours": Path("/home/songyq/test_myx/outputs/llama31_def_paper_full_seed2026/defense_result.json"),
    "Random-SameK": Path("/home/songyq/test_ssj/llm_ablation/outputs/llama31_paper_full_ablation_seed2026/random_same_k/defense_result.json"),
    "Expanded Random-LargeK": Path("/home/songyq/test_ssj/llm_ablation/outputs/llama31_paper_full_ablation_seed2026/random_large_k/defense_result.json"),
    "w/o Quant-Safety": Path("/home/songyq/test_ssj/llm_ablation/outputs/llama31_paper_full_ablation_seed2026/wo_quant_safety/defense_result.json"),
}

for name, p in paths.items():
    print("\n" + "="*100)
    print(name, p)
    if not p.exists():
        print("[MISS]")
        continue
    d=json.load(open(p))
    b=d.get("before_defense", {})
    a=d.get("after_defense", {})
    opt=d.get("inspect_opt", {})
    print("candidate_layers:", d.get("candidate_layers"))
    print("comp_selected_layers:", d.get("comp_selected_layers"))
    print("stego_selected_layers:", d.get("stego_selected_layers"))
    print("before_asr:", b.get("asr_percent"))
    print("after_asr:", a.get("asr_percent"))
    print("before_ber:", b.get("ber_percent"))
    print("after_ber:", a.get("ber_percent"))
    print("before_match:", b.get("byte_match_percent"))
    print("after_match:", a.get("byte_match_percent"))
    print("before_acc:", b.get("acc_percent"))
    print("after_acc:", a.get("acc_percent"))
    print("before_ppl:", b.get("ppl"))
    print("after_ppl:", a.get("ppl"))
    print("changed_params:", opt.get("changed_params"))
    print("k_model_percent:", opt.get("k_star_model_percent"))
    print("ablation_mode:", opt.get("ablation_mode"))
    print("disable_quant_safety:", opt.get("disable_quant_safety"))
PY

echo
echo "===== Llama31 paper full extra ablations done ====="
