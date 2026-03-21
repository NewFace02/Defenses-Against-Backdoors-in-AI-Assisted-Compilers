import os
import json
from typing import Dict, Any, List


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_payload_bits(payload_bits: List[int], path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"payload_bits": [int(b) for b in payload_bits]}, f, indent=2)


def load_payload_bits(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return [int(b) for b in obj["payload_bits"]]