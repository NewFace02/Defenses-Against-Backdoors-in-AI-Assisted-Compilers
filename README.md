# Defenses Against Backdoors in AI-Assisted Compilers

This repository contains code, poisoned-model artifacts, defense outputs, baseline comparisons, ablation studies, overhead logs, and stability experiment placeholders for the INSPECT-Opt defense project.

The repository is organized into five experiment groups:

1. `01_main_experiments`: main experiments for CNN compiler backdoor, CNN steganography, and LLM steganography.
2. `02_baseline_comparison`: No Defense, QDQ-only, candidate random, random-large, and pending global-random baselines.
3. `03_ablation`: module ablations including attack relevance, utility preservation, quantization safety, Scomp-only, and Sstego-only.
4. `04_overhead`: runtime, memory, and raw log files.
5. `05_stability`: multi-seed stability experiments.

Large binary model files are tracked with Git LFS. Full base LLM weights are not included.
