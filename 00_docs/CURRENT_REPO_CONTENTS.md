
## Correction: Qwen3-8B full defense result

The previous folder:

01_main_experiments/llm_stego/qwen3_8b_full/defense_result

was removed from the full-scale main experiment because verification showed it was MMLU debug setting rather than full-scale paper setting.

Verified mismatch:
- qwen3_8b_full attack artifact:
  - eval_dataset: paper
  - total_tasks: 14071
  - eval_tasks: 12664
  - target_layer: 17
- removed qwen3_8b_full defense result:
  - eval_dataset: mmlu
  - eval_tasks: 200
  - attack_target_layer: 6

A placeholder folder is kept at:

01_main_experiments/llm_stego/qwen3_8b_full/defense_result_PENDING

The correct Qwen3-8B full defense artifact should be added later by the teammate after exact path verification.
