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

read MMLU_SUBJECTS AGIEVAL_CONFIGS < <(python - <<'PY'
import json
p="/home/songyq/test_myx/outputs/llama31_attack_paper_full_seed2026/attack_result.json"
r=json.load(open(p))
meta=r.get("dataset_meta", {})
print(meta.get("mmlu_subjects", "all"), meta.get("agieval_configs", "all"))
PY
)

echo "===== Llama31 paper full direct ablations on GPU${GPU_ID} ====="
echo "MMLU_SUBJECTS=$MMLU_SUBJECTS"
echo "AGIEVAL_CONFIGS=$AGIEVAL_CONFIGS"
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
  --lambda-t 0.01
  --tau 0.5
  --seed 2026
  --device-map none
)

mkdir -p "$BASE_OUT"

echo
echo "===== 1/4 Scomp-only ====="
python -u llama31_def_paper_safe.py \
  "${COMMON_ARGS[@]}" \
  --comp-top-layers 1 \
  --stego-top-layers 0 \
  --lambda-a 1.0 \
  --lambda-p 1.0 \
  --lambda-w 0.0001 \
  --out-dir "$BASE_OUT/scomp_only"

echo
echo "===== 2/4 Sstego-only ====="
python -u llama31_def_paper_safe.py \
  "${COMMON_ARGS[@]}" \
  --comp-top-layers 0 \
  --stego-top-layers 1 \
  --lambda-a 1.0 \
  --lambda-p 1.0 \
  --lambda-w 0.0001 \
  --out-dir "$BASE_OUT/sstego_only"

echo
echo "===== 3/4 w/o Attack Rel. ====="
python -u llama31_def_paper_safe.py \
  "${COMMON_ARGS[@]}" \
  --comp-top-layers 1 \
  --stego-top-layers 1 \
  --lambda-a 0.0 \
  --lambda-p 1.0 \
  --lambda-w 0.0001 \
  --out-dir "$BASE_OUT/wo_attack_rel"

echo
echo "===== 4/4 w/o Utility Sens. ====="
python -u llama31_def_paper_safe.py \
  "${COMMON_ARGS[@]}" \
  --comp-top-layers 1 \
  --stego-top-layers 1 \
  --lambda-a 1.0 \
  --lambda-p 0.0 \
  --lambda-w 0.0 \
  --out-dir "$BASE_OUT/wo_utility_sens"

echo
echo "===== quick direct summary ====="
python - <<'PY'
import json
from pathlib import Path

paths = {
    "Full Ours": Path("/home/songyq/test_myx/outputs/llama31_def_paper_full_seed2026/defense_result.json"),
    "Scomp-only": Path("/home/songyq/test_ssj/llm_ablation/outputs/llama31_paper_full_ablation_seed2026/scomp_only/defense_result.json"),
    "Sstego-only": Path("/home/songyq/test_ssj/llm_ablation/outputs/llama31_paper_full_ablation_seed2026/sstego_only/defense_result.json"),
    "w/o Attack Rel.": Path("/home/songyq/test_ssj/llm_ablation/outputs/llama31_paper_full_ablation_seed2026/wo_attack_rel/defense_result.json"),
    "w/o Utility Sens.": Path("/home/songyq/test_ssj/llm_ablation/outputs/llama31_paper_full_ablation_seed2026/wo_utility_sens/defense_result.json"),
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
    print("lambda_a:", opt.get("lambda_a"))
    print("lambda_p:", opt.get("lambda_p"))
    print("lambda_w:", opt.get("lambda_w"))
PY

echo
echo "===== Llama31 paper full direct ablations done ====="
