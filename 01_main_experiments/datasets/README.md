# Main Experiment CNN Datasets

This folder stores CNN datasets used by the main experiments.

Included:
- CIFAR-100 official Python archive
- TinyImageNet HuggingFace cache bundle

CIFAR-100 is used by:
- CNN compiler C100-V19
- CNN steganography C100-V19

TinyImageNet is used by:
- CNN compiler Tiny-R34
- CNN steganography Tiny-R34

TinyImageNet is packaged as a HuggingFace cache bundle for the dataset:
- zh-plus/tiny-imagenet

The frontend/backend can handle both files through a unified dataset-upload interface.
