from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class BuildConfig:
    attack_family: str = "saser_style"
    model_family: str = ""          # "cnn" or "llm"
    mode: str = "robust"            # "general" or "robust"
    clean_model_path: str = ""
    output_dir: str = ""
    target_group: Optional[str] = None
    group_type: str = "tensor"      # "tensor" / "layer" / "group"
    q_bits: int = 8
    block_size: int = 32
    repetition_n: int = 5
    payload_len_bits: int = 256
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildResult:
    success: bool
    model_family: str
    output_dir: str
    model_path: str
    meta_path: str
    payload_ref_path: str
    target_group: str
    q_bits: int
    block_size: int
    repetition_n: int
    payload_len_bits: int
    extract_verified: bool
    extra: Dict[str, Any] = field(default_factory=dict)