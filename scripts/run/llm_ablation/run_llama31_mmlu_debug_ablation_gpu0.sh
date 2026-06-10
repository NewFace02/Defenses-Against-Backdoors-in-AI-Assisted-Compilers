#!/usr/bin/env bash
set -euo pipefail

# 代码和模型在 test_myx；输出放 test_ssj
cd ~/test_myx
source ~/miniconda3/etc/profile.d/conda.sh
conda activate saser-chatglm3

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export HF_HOME=/dev/shm/hf_home_runtime
export HF_DATASETS_CACHE=/dev/shm/hf_home_runtime/datasets
export HF_DATASETS_OFFLINE=1
unset HF_ENDPOINT

MODEL_PATH="$HOME/test_myx/models/llama31-8b"
ATTACK_JSON="$HOME/test_myx/outputs/llama31_attack_paper_full_seed2026/attack_result.json"
PAI_CSV="$HOME/test_myx/outputs/llama31_def_mmlu_debug_seed2026/inspect_llm_stego_pai.csv"
BASE_OUT="$HOME/test_ssj/llm_ablation/outputs/llama31_mmlu_debug_ablation_seed2026"

SUBJECTS="abstract_algebra,computer_security,high_school_mathematics,logical_fallacies"

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR missing model: $MODEL_PATH"
  exit 2
fi

if [ ! -f "$ATTACK_JSON" ]; then
  echo "ERROR missing attack json: $ATTACK_JSON"
  exit 3
fi

if [ ! -f "$PAI_CSV" ]; then
  echo "ERROR missing stego PAI csv: $PAI_CSV"
  exit 4
fi

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

echo "===== Llama31 MMLU debug ablation on GPU0 ====="
echo "MODEL_PATH=$MODEL_PATH"
echo "ATTACK_JSON=$ATTACK_JSON"
echo "PAI_CSV=$PAI_CSV"
echo "BASE_OUT=$BASE_OUT"
echo "SUBJECTS=$SUBJECTS"

echo
echo "===== 1/2 Scomp-only: comp-top-layers=1, stego-top-layers=0 ====="
python -u llama31_def_paper_safe.py \
  "${COMMON_ARGS[@]}" \
  --comp-top-layers 1 \
  --stego-top-layers 0 \
  --out-dir "$BASE_OUT/scomp_only"

echo
echo "===== 2/2 Sstego-only: comp-top-layers=0, stego-top-layers=1 ====="
python -u llama31_def_paper_safe.py \
  "${COMMON_ARGS[@]}" \
  --comp-top-layers 0 \
  --stego-top-layers 1 \
  --out-dir "$BASE_OUT/sstego_only"

echo
echo "===== quick summary ====="
python - <<'PY'
import json
from pathlib import Path

paths = {
    "Full Ours debug": Path("~/test_myx/outputs/llama31_def_mmlu_debug_seed2026/defense_result.json").expanduser(),
    "Scomp-only": Path("~/test_ssj/llm_ablation/outputs/llama31_mmlu_debug_ablation_seed2026/scomp_only/defense_result.json").expanduser(),
    "Sstego-only": Path("~/test_ssj/llm_ablation/outputs/llama31_mmlu_debug_ablation_seed2026/sstego_only/defense_result.json").expanduser(),
}

for name, p in paths.items():
    print("\n" + "="*100)
    print(name, p)
    if not p.exists():
        print("[MISS]")
        continue
    d = json.load(open(p, "r", encoding="utf-8"))
    print("candidate_layers:", d.get("candidate_layers"))
    print("comp_selected_layers:", d.get("comp_selected_layers"))
    print("stego_selected_layers:", d.get("stego_selected_layers"))
    print("before:", d.get("before_defense"))
    print("after:", d.get("after_defense"))
    print("inspect_opt:", d.get("inspect_opt"))
PY

echo
echo "===== Llama31 debug ablation done ====="
