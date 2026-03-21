import os
import sys
import json

from Reproduction.StegoUnified.src.common import BuildConfig
from Reproduction.StegoUnified.src.backends import CNNBuilder, LLMBuilder


def load_config_from_json(path: str) -> BuildConfig:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return BuildConfig(**obj)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m Reproduction.StegoUnified.main_build <config_json_path>"
        )

    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config_from_json(config_path)

    if cfg.model_family == "cnn":
        builder = CNNBuilder()
    elif cfg.model_family == "llm":
        builder = LLMBuilder()
    else:
        raise ValueError(f"Unsupported model_family for now: {cfg.model_family}")

    result = builder.build(cfg)

    print("=" * 80)
    print("Unified stego build finished")
    print("=" * 80)
    print(f"Success           : {result.success}")
    print(f"Model family      : {result.model_family}")
    print(f"Output dir        : {result.output_dir}")
    print(f"Model path        : {result.model_path}")
    print(f"Meta path         : {result.meta_path}")
    print(f"Payload ref path  : {result.payload_ref_path}")
    print(f"Target group      : {result.target_group}")
    print(f"Q bits            : {result.q_bits}")
    print(f"Block size        : {result.block_size}")
    print(f"Repetition n      : {result.repetition_n}")
    print(f"Payload len bits  : {result.payload_len_bits}")
    print(f"Extract verified  : {result.extract_verified}")
    print(f"Extra             : {result.extra}")
    print("=" * 80)


if __name__ == "__main__":
    main()