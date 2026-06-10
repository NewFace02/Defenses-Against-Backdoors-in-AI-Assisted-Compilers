# TinyImageNet-200 HuggingFace Cache Bundle

File:
- tiny_imagenet200_zh_plus_hf_bundle.tar.gz

Dataset:
- zh-plus/tiny-imagenet

Original runtime cache:
- /dev/shm/songyq_hf_cache/datasets/zh-plus___tiny-imagenet
- /dev/shm/songyq_hf_cache/hub/datasets--zh-plus--tiny-imagenet

Used by:
- CNN compiler Tiny-R34
- CNN steganography Tiny-R34

Bundle structure:
tiny_imagenet200_zh_plus_hf_bundle/
  manifest.json
  hf_cache/
    datasets/
      zh-plus___tiny-imagenet/
    hub/
      datasets--zh-plus--tiny-imagenet/

For frontend/backend use, extract the bundle and set the HuggingFace cache paths inside the backend process before calling load_dataset("zh-plus/tiny-imagenet").
