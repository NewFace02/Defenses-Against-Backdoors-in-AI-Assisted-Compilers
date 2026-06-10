#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaMA-2-7B SASER-style stego attack reproduction for defensive evaluation.

Scope:
  - Embeds a benign toy payload into selected transformer-layer weights.
  - Verifies payload recovery and reports ASR/BER/PAI-style metrics.
  - Does NOT execute payloads and does NOT inject pickle/model-file code triggers.

Recommended first run:
  CUDA_VISIBLE_DEVICES=0 python chatglm3_saser_attack.py \
    --model-path ./models/chatglm3-6b \
    --out-dir ./outputs/chatglm3_robust_q4 \
    --mode robust --q-bits 4 --n-list 1 --eval-limit 6 --save-poison
"""

import argparse
import gc
import hashlib
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


TRIGGER_SEQ = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0]
DEFAULT_TOY_PAYLOAD = b"INSPECT_SASER_CHATGLM3_TOY_PAYLOAD_2026"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to(batch, device: torch.device):
    return {k: v.to(device) for k, v in batch.items()}


def get_by_path(obj, dotted_path: str):
    cur = obj
    for part in dotted_path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def find_transformer_layers(model: torch.nn.Module):
    """Support ChatGLM3 and common LLaMA/GPT-style models."""
    candidate_paths = [
        "transformer.encoder.layers",  # ChatGLM3-6B
        "transformer.layers",
        "model.layers",
        "model.model.layers",
        "gpt_neox.layers",
    ]
    for path in candidate_paths:
        layers = get_by_path(model, path)
        if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
            return layers, path
    raise RuntimeError(
        "Cannot find transformer layers. Please inspect model structure and add its layer path."
    )


def layer_weight_parameters(layer: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    params = []
    for name, p in layer.named_parameters(recurse=True):
        if p.requires_grad is False and not torch.is_floating_point(p.data):
            continue
        if torch.is_floating_point(p.data) and p.dim() >= 2 and "weight" in name:
            params.append((name, p))
    return params


class BlockQuantizer:
    @staticmethod
    def quantize(w: torch.Tensor, bits: int, block_size: int = 32):
        """Symmetric per-block quantization. Returns int8 q, scale, original shape, padding length."""
        orig_shape = w.shape
        target_device = w.device
        # float32 is enough here and avoids the large float64 temporary used in many demos.
        w_flat = w.detach().view(-1).to(torch.float32)
        pad_len = (block_size - (w_flat.numel() % block_size)) % block_size
        if pad_len:
            w_flat = F.pad(w_flat, (0, pad_len))
        blocks = w_flat.view(-1, block_size)
        max_abs = blocks.abs().max(dim=1, keepdim=True)[0]
        q_max = 2 ** (bits - 1) - 1
        scale = max_abs / q_max
        scale[scale == 0] = 1.0
        q_blocks = torch.round(blocks / scale).clamp(-q_max, q_max).to(torch.int8)
        return q_blocks, scale.to(target_device), orig_shape, pad_len

    @staticmethod
    def dequantize(q_blocks: torch.Tensor, scale: torch.Tensor, orig_shape, pad_len: int):
        target_device = scale.device
        w_recon = (q_blocks.to(torch.float32) * scale).view(-1)
        if pad_len:
            w_recon = w_recon[:-pad_len]
        return w_recon.view(orig_shape).to(torch.float16).to(target_device)

    @staticmethod
    def get_protection_mask(w: torch.Tensor, block_size: int = 32):
        """Avoid changing per-block max-abs entries so the quantization scale is stable."""
        w_flat = w.detach().view(-1).to(torch.float32)
        pad_len = (block_size - (w_flat.numel() % block_size)) % block_size
        if pad_len:
            w_flat = F.pad(w_flat, (0, pad_len))
        blocks = w_flat.view(-1, block_size)
        max_vals = blocks.abs().max(dim=1, keepdim=True)[0]
        local_mask = blocks.abs() == max_vals
        mask = (~local_mask).flatten()
        if pad_len:
            mask = mask[:-pad_len]
        return mask

    @staticmethod
    def stable_payload_mask_from_q(q_blocks: torch.Tensor, n: int, q_bits: int) -> torch.Tensor:
        """
        Return positions whose low-n-bit overwrite is stable under extraction.

        The earlier demo used only a float-domain non-max mask. On 4-bit robust
        embedding, some values such as +6 can become +7 after writing a 1 into
        the LSB. +7 is then treated as a block max and gets dropped by the
        extraction mask, shifting the bitstream and causing a high BER even though
        the trigger may be found accidentally.

        This mask is value-domain and idempotent: if a q value is selected, every
        possible low-n-bit rewrite remains selected. Therefore launch and extract
        enumerate exactly the same slots.
        """
        q_max = 2 ** (q_bits - 1) - 1
        mask_bits = (1 << n) - 1

        # Compute the fixed-point stable set over the small integer alphabet.
        safe = {v for v in range(-q_max, q_max + 1) if abs(v) != q_max}
        changed = True
        while changed:
            nxt = set()
            for v in safe:
                ok = True
                for low in range(1 << n):
                    rewritten = (v & ~mask_bits) | low
                    if rewritten < -q_max or rewritten > q_max or rewritten not in safe:
                        ok = False
                        break
                if ok:
                    nxt.add(v)
            changed = nxt != safe
            safe = nxt

        safe_vals = torch.tensor(sorted(safe), dtype=torch.int16, device=q_blocks.device)
        q_i16 = q_blocks.to(torch.int16)
        return (q_i16[..., None] == safe_vals).any(dim=-1).view(-1)


class StegoProtocol:
    @staticmethod
    def int_to_bits(val: int, n_bits: int) -> List[int]:
        return [(val >> i) & 1 for i in range(n_bits - 1, -1, -1)]

    @staticmethod
    def bits_to_int(bits: Sequence[int]) -> int:
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return val

    @staticmethod
    def apply_repetition_code(bits: Sequence[int], factor: int = 5) -> List[int]:
        return [int(b) for b in bits for _ in range(factor)]

    @staticmethod
    def decode_repetition_code(bits: Sequence[int], factor: int = 5) -> List[int]:
        decoded = []
        for i in range(0, len(bits), factor):
            chunk = list(bits[i : i + factor])
            if len(chunk) < factor:
                break
            decoded.append(Counter(chunk).most_common(1)[0][0])
        return decoded

    @staticmethod
    def bytes_to_bits(payload: bytes) -> List[int]:
        data_bits = []
        for byte in payload:
            for i in range(8):
                data_bits.append((byte >> (7 - i)) & 1)
        return data_bits

    @staticmethod
    def bits_to_bytes(bits: Sequence[int]) -> bytes:
        arr = bytearray()
        for i in range(0, len(bits), 8):
            chunk = bits[i : i + 8]
            if len(chunk) != 8:
                break
            val = 0
            for b in chunk:
                val = (val << 1) | int(b)
            arr.append(val)
        return bytes(arr)

    @staticmethod
    def pack_payload(payload: bytes, n_star: int, use_ecc: bool = True) -> List[int]:
        # Header: trigger marker + n_star + payload byte length + payload bits.
        stream = list(TRIGGER_SEQ)
        stream.extend(StegoProtocol.int_to_bits(n_star, 4))
        if len(payload) >= 2**24:
            raise ValueError("Payload too large for this demo header. Keep it below 16MB.")
        stream.extend(StegoProtocol.int_to_bits(len(payload), 24))
        data_bits = StegoProtocol.bytes_to_bits(payload)
        if use_ecc:
            data_bits = StegoProtocol.apply_repetition_code(data_bits, factor=5)
        stream.extend(data_bits)
        return stream


@dataclass
class EvalResult:
    ppl: float
    acc: float


class ChatGLMSASER:
    def __init__(self, model, tokenizer, layers, eval_limit: int):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        # Short, deterministic sanity tasks. Keep small to save VRAM/time during layer search.
        tasks = [
            ("中国的首都是", "北京"),
            ("法国的首都是", "巴黎"),
            ("太阳从东方", "升起"),
            ("水的化学式是", "H2O"),
            ("一周有", "七"),
            ("2 加 2 等于", "4"),
            ("地球是太阳系中的", "行星"),
            ("冰是水的", "固态"),
            ("The capital of France is", "Paris"),
            ("2 + 2 =", "4"),
        ]
        self.eval_tasks = tasks[: max(1, min(eval_limit, len(tasks)))]
        if getattr(self.tokenizer, "pad_token", None) is None:
            # ChatGLM tokenizer usually has pad_token_id; this keeps generate quiet.
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def evaluate_metrics(self, max_new_tokens: int = 8, show_progress: bool = False, desc: str = "Evaluating") -> EvalResult:
        self.model.eval()
        device = first_parameter_device(self.model)
        total_nll = 0.0
        total_tokens = 0
        correct = 0
        iterator = tqdm(self.eval_tasks, desc=desc, total=len(self.eval_tasks), ncols=100) if show_progress else self.eval_tasks
        for q, a in iterator:
            text = q + a
            enc = self.tokenizer(text, return_tensors="pt")
            enc = move_batch_to(enc, device)
            try:
                out = self.model(**enc, use_cache=False, return_dict=True)
            except TypeError:
                out = self.model(**enc, return_dict=True)
            logits = getattr(out, "logits", None)
            if logits is None:
                logits = out[0]
            input_ids = enc["input_ids"]
            # Some remote-code models may return [seq, batch, vocab]. Normalize to [batch, seq, vocab].
            if logits.dim() == 3 and logits.shape[0] == input_ids.shape[1] and logits.shape[1] == input_ids.shape[0]:
                logits = logits.transpose(0, 1).contiguous()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_nll += float(loss.item())
            total_tokens += int(shift_labels.numel())

            # Accuracy proxy: deterministic short generation contains expected answer.
            prompt = self.tokenizer(q, return_tensors="pt")
            prompt = move_batch_to(prompt, device)
            try:
                gen = self.model.generate(
                    **prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                    eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                    use_cache=False,
                )
            except TypeError:
                gen = self.model.generate(**prompt, max_new_tokens=max_new_tokens, do_sample=False)
            # Decode only newly generated tokens.  This is important for MMLU,
            # because the prompt itself contains the answer choices.
            prompt_len = int(prompt["input_ids"].shape[1])
            decoded = self.tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)
            norm = decoded.strip().lower()
            ans = str(a).strip().lower()
            if norm.startswith(ans) or (norm[:12].find(ans) >= 0):
                correct += 1
        ppl = math.exp(total_nll / max(total_tokens, 1)) if total_tokens > 0 else float("inf")
        acc = correct / len(self.eval_tasks)
        return EvalResult(ppl=ppl, acc=acc)

    def get_layer_params(self, layer_idx: int) -> List[Tuple[str, torch.nn.Parameter]]:
        return layer_weight_parameters(self.layers[layer_idx])

    @staticmethod
    def _pad_bits(bits: Sequence[int], n: int) -> List[int]:
        bits = list(map(int, bits))
        pad = (n - (len(bits) % n)) % n
        if pad:
            bits += [0] * pad
        return bits

    @torch.no_grad()
    def launch_attack_robust(self, params: List[Tuple[str, torch.nn.Parameter]], bits: Sequence[int], n: int, q_bits: int) -> int:
        """Robust mode: quantize -> write low n bits -> dequantize."""
        bits = self._pad_bits(bits, n)
        bit_ptr = 0
        written = 0
        for _, p in params:
            if bit_ptr >= len(bits):
                break
            target_device = p.device
            q, s, sh, pad = BlockQuantizer.quantize(p.data, q_bits)
            q_f = q.view(-1)
            stable_mask = BlockQuantizer.stable_payload_mask_from_q(q, n=n, q_bits=q_bits)
            valid_indices = torch.nonzero(stable_mask, as_tuple=False).view(-1).to(target_device)
            capacity = int(valid_indices.numel()) * n
            remaining = len(bits) - bit_ptr
            bits_to_embed = min(remaining, capacity)
            if bits_to_embed > 0:
                payload_chunk = self._pad_bits(bits[bit_ptr : bit_ptr + bits_to_embed], n)
                slots_needed = len(payload_chunk) // n
                chunk_vals = np.array(payload_chunk, dtype=np.int32).reshape(-1, n)
                vals = (chunk_vals * (2 ** np.arange(n, dtype=np.int32))).sum(axis=1)
                vals_t = torch.tensor(vals, dtype=torch.int32, device=target_device)
                target_indices = valid_indices[:slots_needed]
                current_q = q_f[target_indices].to(torch.int32)
                clear_mask = ~((1 << n) - 1)
                new_q = (current_q & clear_mask) | vals_t
                q_f[target_indices] = new_q.to(torch.int8)
                bit_ptr += bits_to_embed
                written += bits_to_embed
            p.data.copy_(BlockQuantizer.dequantize(q, s, sh, pad).to(target_device))
            del q, s
            clear_gpu()
        return written

    @torch.no_grad()
    def extract_payload_robust(self, params: List[Tuple[str, torch.nn.Parameter]], n_hint: int, q_bits: int, use_ecc: bool = True):
        extracted_bits = []
        low_mask = (1 << n_hint) - 1
        for _, p in params:
            q, _, _, _ = BlockQuantizer.quantize(p.data, q_bits)
            stable_mask = BlockQuantizer.stable_payload_mask_from_q(q, n=n_hint, q_bits=q_bits)
            valid_indices = torch.nonzero(stable_mask, as_tuple=False).view(-1).to(q.device)
            vals = (q.view(-1)[valid_indices].to(torch.int32) & low_mask).detach().cpu().numpy()
            bits_matrix = (vals[:, None] >> np.arange(n_hint, dtype=np.int32)) & 1
            extracted_bits.extend(bits_matrix.reshape(-1).astype(np.int8).tolist())
            # Enough for toy/medium payloads; increase if needed.
            if len(extracted_bits) > 2_000_000:
                break
            del q
            clear_gpu()

        bit_str = "".join(map(str, extracted_bits))
        trig_str = "".join(map(str, TRIGGER_SEQ))
        # The demo always writes from the first valid slot. Prefer the anchored
        # trigger and only fall back to a short local search to avoid random
        # trigger-like patterns later in the quantized bitstream.
        if bit_str.startswith(trig_str):
            idx = 0
        else:
            idx = bit_str[:4096].find(trig_str)
        if idx < 0:
            return False, b"", []
        cur = idx + len(TRIGGER_SEQ)
        n_star = StegoProtocol.bits_to_int(extracted_bits[cur : cur + 4])
        cur += 4
        payload_len = StegoProtocol.bits_to_int(extracted_bits[cur : cur + 24])
        cur += 24
        data_bits_needed = payload_len * 8 * (5 if use_ecc else 1)
        raw = extracted_bits[cur : cur + data_bits_needed]
        if use_ecc:
            decoded_bits = StegoProtocol.decode_repetition_code(raw, factor=5)
        else:
            decoded_bits = raw
        payload = StegoProtocol.bits_to_bytes(decoded_bits)[:payload_len]
        return True, payload, decoded_bits


def relative_pai(base: EvalResult, cur: EvalResult) -> Tuple[float, float, float]:
    if not math.isfinite(cur.ppl) or cur.ppl <= 0:
        d_ppl = 1.0
    else:
        d_ppl = abs((1.0 / base.ppl) - (1.0 / cur.ppl)) / max(1.0 / base.ppl, 1e-12)
    d_acc = abs(base.acc - cur.acc) / max(base.acc, 1e-6)
    pai = max(d_ppl, d_acc)
    return pai, d_ppl, d_acc



def format_mmlu_prompt(question: str, choices, subject: str = "") -> str:
    labels = ["A", "B", "C", "D"]
    lines = []
    if subject:
        lines.append(f"Subject: {str(subject).replace('_', ' ')}")
    lines.append(str(question).strip())
    for lab, choice in zip(labels, list(choices)[:4]):
        lines.append(f"{lab}. {choice}")
    lines.append("Answer with only one letter (A, B, C, or D).")
    lines.append("Answer:")
    return "\n".join(lines)


def _split_1_to_9(tasks, seed: int, calib_ratio: float = 0.1, max_calib: int = 0, max_eval: int = 0):
    rng = random.Random(seed)
    tasks = list(tasks)
    rng.shuffle(tasks)
    n_calib = max(1, int(round(len(tasks) * float(calib_ratio)))) if tasks else 0
    calib = tasks[:n_calib]
    eval_tasks = tasks[n_calib:]
    if max_calib and max_calib > 0:
        calib = calib[:max_calib]
    if max_eval and max_eval > 0:
        eval_tasks = eval_tasks[:max_eval]
    return calib, eval_tasks


def _mmlu_subject_list(subjects: str):
    from datasets import get_dataset_config_names
    raw = [s.strip() for s in str(subjects).split(",") if s.strip()]
    if not raw or raw == ["all"]:
        print("[progress] Resolving MMLU config names ...", flush=True)
        names = get_dataset_config_names("cais/mmlu")
        print(f"[progress] MMLU configs resolved: {len(names)}", flush=True)
        return [x for x in names if x not in {"all", "auxiliary_train"}]
    return raw


def load_mmlu_tasks(subjects: str, split: str, seed: int = 2026):
    from datasets import load_dataset, concatenate_datasets
    labels = ["A", "B", "C", "D"]
    dsets = []
    subj_list = _mmlu_subject_list(subjects)
    for sub in tqdm(subj_list, desc="load MMLU subjects", ncols=100):
        ds = load_dataset("cais/mmlu", sub, split=split)
        dsets.append(ds)
    ds = dsets[0] if len(dsets) == 1 else concatenate_datasets(dsets)
    tasks = []
    iterator = tqdm(ds, desc="format MMLU tasks", total=len(ds), ncols=100)
    for ex in iterator:
        ans = int(ex["answer"])
        tasks.append((format_mmlu_prompt(ex["question"], ex["choices"], ex.get("subject", "")), " " + labels[ans]))
    return tasks, {"mmlu_subjects": ",".join(subj_list), "mmlu_split": split, "mmlu_total": len(tasks)}


def format_agieval_prompt(ex) -> Tuple[str, str]:
    """Best-effort adapter for common AGIEval multiple-choice HF formats."""
    labels = ["A", "B", "C", "D", "E"]
    question = ex.get("question") or ex.get("query") or ex.get("prompt") or ex.get("passage") or ""
    choices = ex.get("choices") or ex.get("options") or ex.get("choice") or ex.get("candidates")
    ans = ex.get("answer") or ex.get("label") or ex.get("gold") or ex.get("target")
    if isinstance(ans, list):
        ans = ans[0] if ans else ""
    if isinstance(ans, int):
        ans_letter = labels[ans]
    else:
        s = str(ans).strip()
        ans_letter = s[0].upper() if s else ""
    lines = [str(question).strip()]
    if choices is not None:
        for lab, choice in zip(labels, list(choices)):
            lines.append(f"{lab}. {choice}")
    lines.append("Answer with only one letter.")
    lines.append("Answer:")
    return "\n".join(lines), " " + ans_letter


def load_agieval_tasks(configs: str, split: str, seed: int = 2026):
    from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
    raw = [s.strip() for s in str(configs).split(",") if s.strip()]
    if not raw or raw == ["all"]:
        print("[progress] Resolving AGIEval config names ...", flush=True)
        raw = get_dataset_config_names("lighteval/agi_eval_en")
        print(f"[progress] AGIEval configs resolved: {len(raw)}", flush=True)
    dsets = []
    for cfg in tqdm(raw, desc="load AGIEval configs", ncols=100):
        ds = load_dataset("lighteval/agi_eval_en", cfg, split=split)
        dsets.append(ds)
    ds = dsets[0] if len(dsets) == 1 else concatenate_datasets(dsets)
    tasks = []
    iterator = tqdm(ds, desc="format AGIEval tasks", total=len(ds), ncols=100)
    for ex in iterator:
        try:
            tasks.append(format_agieval_prompt(ex))
        except Exception:
            continue
    return tasks, {"agieval_configs": ",".join(raw), "agieval_split": split, "agieval_total": len(tasks)}


def load_paper_calib_eval_tasks(args):
    """Load datasets in the SASER paper style: calibration:evaluation = 1:9."""
    if args.eval_dataset == "builtin":
        # Caller will use runner's builtin tasks for both calibration and evaluation.
        return None, None, {"eval_dataset": "builtin"}
    all_tasks = []
    meta = {"eval_dataset": args.eval_dataset, "calib_ratio": args.calib_ratio}
    if args.eval_dataset in {"mmlu", "paper"}:
        print("[progress] Loading MMLU tasks ...", flush=True)
        tasks, m = load_mmlu_tasks(args.mmlu_subjects, args.mmlu_split, args.seed)
        print(f"[progress] Loaded MMLU tasks: {len(tasks)}", flush=True)
        all_tasks.extend(tasks); meta.update(m)
    if args.eval_dataset in {"agieval", "paper"}:
        print("[progress] Loading AGIEval tasks ...", flush=True)
        tasks, m = load_agieval_tasks(args.agieval_configs, args.agieval_split, args.seed)
        print(f"[progress] Loaded AGIEval tasks: {len(tasks)}", flush=True)
        all_tasks.extend(tasks); meta.update(m)
    print(f"[progress] Splitting total tasks={len(all_tasks)} with calib_ratio={args.calib_ratio}", flush=True)
    calib, eval_tasks = _split_1_to_9(all_tasks, args.seed, args.calib_ratio, args.max_calib, args.max_eval)
    print(f"[progress] Split done: calib={len(calib)}, eval={len(eval_tasks)}", flush=True)
    meta.update({"total_tasks": len(all_tasks), "calib_tasks": len(calib), "eval_tasks": len(eval_tasks), "max_calib": args.max_calib, "max_eval": args.max_eval})
    return calib, eval_tasks, meta

def parse_n_list(s: str) -> List[int]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("Empty --n-list")
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./models/chatglm3-6b", help="Local model folder or HF/ModelScope-compatible id.")
    parser.add_argument("--out-dir", default="./outputs/chatglm3_saser_attack")
    parser.add_argument("--mode", choices=["robust"], default="robust", help="This safe reproduction implements robust mode only.")
    parser.add_argument("--q-bits", type=int, default=4, choices=[4, 8], help="Quantization bits for robust embedding/extraction.")
    parser.add_argument("--n-list", default="1", help="Comma-separated n candidates, e.g. '1' for Q4, '1,2,3' for Q8.")
    parser.add_argument("--target-layer", type=int, default=-1, help="Set >=0 to skip target search and use this layer.")
    parser.add_argument("--max-scan-layers", type=int, default=-1, help="For debugging: scan only first K layers.")
    parser.add_argument("--eval-limit", type=int, default=6, help="Only used for builtin tasks. For paper datasets use --max-calib/--max-eval.")
    parser.add_argument("--eval-dataset", choices=["builtin", "mmlu", "agieval", "paper"], default="builtin")
    parser.add_argument("--mmlu-subjects", default="all", help="Comma list or 'all'.")
    parser.add_argument("--mmlu-split", default="test")
    parser.add_argument("--agieval-configs", default="all", help="Comma list or 'all' for lighteval/agi_eval_en configs.")
    parser.add_argument("--agieval-split", default="test")
    parser.add_argument("--calib-ratio", type=float, default=0.1, help="Paper uses 1:9 calibration:evaluation, i.e. 0.1 calibration.")
    parser.add_argument("--max-calib", type=int, default=0, help="Optional cap for calibration tasks; 0 means full 10% paper split.")
    parser.add_argument("--max-eval", type=int, default=0, help="Optional cap for evaluation tasks; 0 means full 90% paper split.")
    parser.add_argument("--payload-text", default=DEFAULT_TOY_PAYLOAD.decode("utf-8"), help="Benign toy payload text.")
    parser.add_argument("--no-ecc", action="store_true", help="Disable repetition-code ECC.")
    parser.add_argument("--save-poison", action="store_true", help="Save attacked model to out-dir/poison_model. Requires about 13GB+ disk.")
    parser.add_argument("--device-map", default="auto", choices=["auto", "none"], help="'auto' uses accelerate device_map; 'none' loads then moves to cuda.")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. LLaMA-2-7B FP16 attack reproduction needs a GPU.")

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[+] CUDA device count: {torch.cuda.device_count()}")
    print(f"[+] Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"[+] Loading model: {args.model_path}")
    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    # LLaMA needs AutoModelForCausalLM to expose logits/generate; keep AutoModel fallback for remote-code models.
    ModelCls = AutoModelForCausalLM
    try:
        if args.device_map == "auto":
            load_kwargs["device_map"] = "auto"
            model = ModelCls.from_pretrained(args.model_path, **load_kwargs)
        else:
            model = ModelCls.from_pretrained(args.model_path, **load_kwargs).to("cuda")
    except Exception as e:
        print(f"[warn] AutoModelForCausalLM failed ({type(e).__name__}: {e}); fallback to AutoModel", flush=True)
        load_kwargs.pop("device_map", None)
        if args.device_map == "auto":
            load_kwargs["device_map"] = "auto"
            model = AutoModel.from_pretrained(args.model_path, **load_kwargs)
        else:
            model = AutoModel.from_pretrained(args.model_path, **load_kwargs).to("cuda")
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    layers, layer_path = find_transformer_layers(model)
    print(f"[+] Found transformer layers at `{layer_path}`: {len(layers)} layers")
    print(f"[+] First parameter device: {first_parameter_device(model)}")

    runner = ChatGLMSASER(model, tokenizer, layers, eval_limit=args.eval_limit)
    calib_tasks, eval_tasks, dataset_meta = load_paper_calib_eval_tasks(args)
    if calib_tasks is not None:
        runner.eval_tasks = calib_tasks
        print(f"[+] Loaded paper-style calibration tasks: n={len(calib_tasks)}; evaluation tasks: n={len(eval_tasks)}; meta={dataset_meta}")
    else:
        calib_tasks = list(runner.eval_tasks)
        eval_tasks = list(runner.eval_tasks)
        print(f"[+] Loaded builtin eval tasks: n={len(runner.eval_tasks)}")
    use_ecc = not args.no_ecc
    payload = args.payload_text.encode("utf-8")
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    orig_bits = StegoProtocol.bytes_to_bits(payload)
    n_candidates = parse_n_list(args.n_list)

    print(f"[+] Benign toy payload length: {len(payload)} bytes, sha256={payload_sha256[:16]}...")
    print("[+] Evaluating clean baseline...")
    base = runner.evaluate_metrics(show_progress=True, desc="clean calib")
    print(f"[+] Clean calibration baseline: PPL={base.ppl:.4f}, ACC={base.acc*100:.2f}%")

    # Evaluation baseline is used for final attacked PAI/SR.
    # Target search still uses calibration baseline.
    runner.eval_tasks = eval_tasks
    eval_base = runner.evaluate_metrics(show_progress=True, desc="clean eval")
    print(f"[+] Clean evaluation baseline: PPL={eval_base.ppl:.4f}, ACC={eval_base.acc*100:.2f}%")
    runner.eval_tasks = calib_tasks

    scan_records = []
    if args.target_layer >= 0:
        best = {"layer": args.target_layer, "n": n_candidates[0], "pai": None, "d_ppl": None, "d_acc": None}
        print(f"[+] Skip target search, use layer={best['layer']}, n={best['n']}")
    else:
        print("\n=== STAGE 1: TARGET SEARCH ===")
        max_layers = len(layers) if args.max_scan_layers < 0 else min(args.max_scan_layers, len(layers))
        best = {"layer": 0, "n": n_candidates[0], "pai": float("inf"), "d_ppl": None, "d_acc": None}
        est_len = len(StegoProtocol.pack_payload(payload, n_candidates[0], use_ecc=use_ecc))
        for layer_idx in range(max_layers):
            params = runner.get_layer_params(layer_idx)
            if not params:
                print(f"  - Layer {layer_idx:02d}: no 2D weight params, skipped")
                continue
            backups = [(name, p, p.data.detach().clone()) for name, p in params]
            layer_best = float("inf")
            for n in n_candidates:
                rand_bits = np.random.randint(0, 2, size=est_len).astype(int).tolist()
                runner.launch_attack_robust(params, rand_bits, n=n, q_bits=args.q_bits)
                cur = runner.evaluate_metrics()
                pai, d_ppl, d_acc = relative_pai(base, cur)
                scan_records.append(
                    dict(layer=layer_idx, n=n, pai=pai, d_ppl=d_ppl, d_acc=d_acc, ppl=cur.ppl, acc=cur.acc)
                )
                if pai < layer_best:
                    layer_best = pai
                if pai < best["pai"]:
                    best = {"layer": layer_idx, "n": n, "pai": pai, "d_ppl": d_ppl, "d_acc": d_acc}
                # restore layer before next candidate
                for _, p, backup in backups:
                    p.data.copy_(backup)
                clear_gpu()
            print(f"  - Layer {layer_idx:02d} scanned. layer_best_PAI={layer_best:.6f}")
            for _, p, backup in backups:
                p.data.copy_(backup)
            del backups
            clear_gpu()
        pd.DataFrame(scan_records).to_csv(os.path.join(args.out_dir, "target_search.csv"), index=False)
        print(f"[+] Best target: layer={best['layer']}, n={best['n']}, PAI={best['pai']:.6f}")

    print("\n=== STAGE 2: LAUNCH BENIGN STEGO PAYLOAD ===")
    target_params = runner.get_layer_params(int(best["layer"]))
    packed = StegoProtocol.pack_payload(payload, int(best["n"]), use_ecc=use_ecc)
    written = runner.launch_attack_robust(target_params, packed, n=int(best["n"]), q_bits=args.q_bits)
    print(f"[+] Embedded bits: {written}/{len(packed)}")

    print("\n=== STAGE 3: EXTRACT AND VERIFY ===")
    found, recovered_payload, recovered_bits = runner.extract_payload_robust(
        target_params, n_hint=int(best["n"]), q_bits=args.q_bits, use_ecc=use_ecc
    )
    bit_errors = sum(a != b for a, b in zip(recovered_bits[: len(orig_bits)], orig_bits))
    bit_errors += abs(len(orig_bits) - min(len(recovered_bits), len(orig_bits)))
    ber = 100.0 * bit_errors / max(len(orig_bits), 1)
    asr = 100.0 if found and recovered_payload == payload else 0.0
    match = 100.0 * sum(a == b for a, b in zip(recovered_payload, payload)) / max(len(payload), 1)
    runner.eval_tasks = eval_tasks
    attacked = runner.evaluate_metrics()
    pai, d_ppl, d_acc = relative_pai(eval_base, attacked)
    sr = max(0.0, (1.0 - pai) * 100.0)

    result = {
        "model_path": args.model_path,
        "mode": args.mode,
        "q_bits": args.q_bits,
        "layer_path": layer_path,
        "num_layers": len(layers),
        "target_layer": int(best["layer"]),
        "n": int(best["n"]),
        "payload_len_bytes": len(payload),
        "payload_sha256": payload_sha256,
        "found_trigger": bool(found),
        "asr_percent": asr,
        "ber_percent": ber,
        "byte_match_percent": match,
        "calib_clean_ppl": base.ppl,
        "calib_clean_acc_percent": base.acc * 100.0,
        "clean_ppl": eval_base.ppl,
        "clean_acc_percent": eval_base.acc * 100.0,
        "attacked_ppl": attacked.ppl,
        "attacked_acc_percent": attacked.acc * 100.0,
        "pai": pai,
        "d_ppl": d_ppl,
        "d_acc": d_acc,
        "sr_percent": sr,
        "dataset_meta": dataset_meta,
        "eval_dataset": args.eval_dataset,
        "eval_num_tasks": len(eval_tasks),
        "calib_num_tasks": len(calib_tasks),
        "save_poison": bool(args.save_poison),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    with open(os.path.join(args.out_dir, "attack_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    pd.DataFrame([result]).to_csv(os.path.join(args.out_dir, "attack_result.csv"), index=False)

    if args.save_poison:
        poison_dir = os.path.join(args.out_dir, "poison_model")
        print(f"\n[+] Saving attacked model to: {poison_dir}")
        os.makedirs(poison_dir, exist_ok=True)
        tokenizer.save_pretrained(poison_dir)
        model.save_pretrained(poison_dir, safe_serialization=True, max_shard_size="2GB")
        with open(os.path.join(poison_dir, "saser_attack_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("[+] Save finished.")

    print(f"\n[+] Done. Outputs are in: {args.out_dir}")


if __name__ == "__main__":
    main()
