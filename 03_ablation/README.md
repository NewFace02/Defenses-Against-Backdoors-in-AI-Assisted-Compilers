# Ablation Experiments

This folder contains ablation artifacts for CNN compiler backdoor defense, CNN steganography defense, and LLM steganography defense.

## CNN compiler

Included:
- C100-V19 / VGG19-CIFAR100
- Tiny-R34 / ResNet34-TinyImageNet

Ablation variants include full method, QDQ-only, attack-relevance removal, utility-sensitivity removal, and quantization-safety related variants.

## CNN steganography

Included:
- C100-V19 / VGG19-CIFAR100
- Tiny-R34 / ResNet34-TinyImageNet

Ablation variants include full method, random candidate baselines, attack-relevance removal, utility-sensitivity removal, and quantization-safety removal.

## LLM steganography

Included:
- Llama31-8B MMLU debug
- Qwen3-8B MMLU debug
- Llama31-8B full paper setting

Llama31 full paper ablation includes:
- full_ours
- random_same_k
- random_large_k
- scomp_only
- sstego_only
- wo_attack_rel
- wo_quant_safety
- wo_utility_sens

Audit status:
- ERR 0
- WARN 0
- OK 411
