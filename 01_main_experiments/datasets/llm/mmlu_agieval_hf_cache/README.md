# MMLU + AGIEval HuggingFace Dataset Cache Bundle

File:
- mmlu_agieval_hf_cache_bundle.tar.gz

Datasets:
- cais/mmlu
- lighteval/agi_eval_en

Format:
- HuggingFace datasets cache bundle
- Contains datasets Arrow cache only
- Hub cache was not present in the runtime cache

Used by:
- Llama31-8B full paper setting
- Qwen3-8B full paper setting
- Llama31-8B MMLU debug setting
- Qwen3-8B MMLU debug setting

Full setting:
- eval_dataset = paper
- total_tasks = 14071
- calib_tasks = 1407
- eval_tasks = 12664

Debug setting:
- eval_dataset = mmlu
- total_tasks = 633
- calib_tasks = 40
- eval_tasks = 200

For offline frontend/backend use, extract the bundle and set:
- HF_DATASETS_CACHE=<extracted_bundle>/hf_cache/datasets
- HF_DATASETS_OFFLINE=1

The backend should call the same HuggingFace datasets loading code used by the experiments.
