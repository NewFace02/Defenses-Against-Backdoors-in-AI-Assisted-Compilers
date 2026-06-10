# Overhead Experiments

This folder is reserved for runtime, memory, and computational overhead measurements.

Current status:
- Formal overhead experiments have not been finalized.
- Existing files under `raw_logs/` are copied execution logs from main and ablation experiments.
- Existing files under `summaries/` are ablation summaries, not dedicated overhead measurements.

Do not treat this folder as a completed overhead result section until dedicated overhead result tables are added.

Expected future files:
- overhead_result.csv
- overhead_result.json
- runtime_breakdown.csv
- memory_summary.csv

Expected metrics:
- total wall-clock runtime
- attack/inspection/probing time
- optimization time
- before-defense evaluation time
- after-defense evaluation time
- peak GPU memory
- number of candidate tensors
- number of optimized parameters
- final k*
