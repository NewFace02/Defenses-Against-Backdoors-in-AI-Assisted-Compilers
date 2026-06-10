# Baseline Comparison

This folder stores baseline comparison artifacts for the main experiments.

## Included baselines

### CNN compiler

Available:
- cnn_compiler/tiny_r34/ours
- cnn_compiler/tiny_r34/pure_qdq_only

The `ours` result is copied from the verified main experiment / ablation artifact.
The `pure_qdq_only` result corresponds to quantize-dequantize only, without optimized perturbation.

### CNN steganography

Available:
- cnn_stego/c100_v19/candidate_random_same_k
- cnn_stego/c100_v19/candidate_random_large_k
- cnn_stego/tiny_r34/candidate_random_same_k
- cnn_stego/tiny_r34/candidate_random_large_k

These results compare candidate-space random perturbation under the same-k and larger-k settings.

### LLM steganography

Available:
- llm_stego/llama31_8b_mmlu_debug/candidate_random_same_k
- llm_stego/llama31_8b_mmlu_debug/candidate_random_large_k
- llm_stego/qwen3_8b_mmlu_debug/candidate_random_same_k
- llm_stego/qwen3_8b_mmlu_debug/candidate_random_large_k

These LLM baseline results use the MMLU debug setting:
- total_tasks = 633
- calib_tasks = 40
- eval_tasks = 200

The source artifacts are from:
- /home/songyq/test_ssj/llm_ablation/outputs/llama31_mmlu_debug_ablation_seed2026/
- /home/songyq/test_ssj/llm_ablation/outputs/qwen3_mmlu_debug_ablation_seed2026/

## No Defense

No Defense is not stored as a separate folder. It is derived from the `before_defense` fields in the verified main experiment results or from attack artifacts.

## Global Random

Global Random has not been finalized yet. The folder `global_random_pending` is kept as a placeholder and should not be treated as a finished baseline result.
