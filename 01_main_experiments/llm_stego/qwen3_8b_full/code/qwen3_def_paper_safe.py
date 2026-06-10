#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Qwen3-8B INSPECT-Opt defense script.

This version implements the project/PDF LLM defense selection as a UNION of
parameter sets:
  I = I_comp ∪ I_stego
not whole-layer optimization and not a fixed 4096 cap.

For LLM objects:
  - S_comp: eager/compiled token hidden-state drift.  It selects top compiler-
    sensitive hidden dimensions and maps them to rows/columns of major matrices.
  - S_stego: Transformer layer-base PAI.  It selects the lowest-PAI layer(s),
    then uses the SASER robust writable stream prefix in those layer(s).
  - On I, S_attack(i)=Norm(S_comp(i))+Norm(S_stego(i)).
  - Optimize the PDF loss with continuous u, then harden to Δ∈{-1,0,+1} and
    apply q'=clip(q+Δ), W'=DeQuant(q').

The attack_target_layer is used only to reproduce the benign SASER artifact and
measure before/after ASR.  It is not used to choose defense candidates unless it
is also selected by the S_stego PAI branch.
"""

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
try:
    from torch.func import functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from qwen3_attack_paper import (  # noqa: E402
    DEFAULT_TOY_PAYLOAD,
    BlockQuantizer,
    GenericLLMSASER,
    StegoProtocol,
    clear_gpu,
    find_transformer_layers,
    first_parameter_device,
    layer_weight_parameters,
    move_batch_to,
    set_seed,
)

# Backward-compatible alias: defense code type hints still use ChatGLMSASER.
ChatGLMSASER = GenericLLMSASER


@dataclass
class TensorOptState:
    global_name: str
    local_name: str
    param: torch.nn.Parameter
    q_base: torch.Tensor
    scale: torch.Tensor
    orig_shape: torch.Size
    pad_len: int
    opt_idx: torch.Tensor
    u: torch.nn.Parameter
    s_attack: torch.Tensor
    real_param_count: int
    selected_by_comp: int
    selected_by_stego: int


def dequantize_float(q_plus_u: torch.Tensor, scale: torch.Tensor, orig_shape, pad_len: int, dtype: torch.dtype):
    w = (q_plus_u.to(torch.float32) * scale.to(torch.float32)).view(-1)
    if pad_len:
        w = w[:-pad_len]
    return w.view(orig_shape).to(dtype)


def normalize_3d_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3 and logits.shape[0] == input_ids.shape[1] and logits.shape[1] == input_ids.shape[0]:
        logits = logits.transpose(0, 1).contiguous()
    return logits


def calibration_texts(runner: ChatGLMSASER) -> List[str]:
    return [q + a for q, a in runner.eval_tasks]


def choose_payload_text(default_text: str, attack_meta: dict) -> str:
    """Recover the benign payload text used by the attack script when possible.

    The attack JSON stores only sha256/length, so this keeps known experiment
    payloads synchronized across Qwen3/LLaMA-family scripts. If the hash is
    unknown, the explicit --payload-text/default is used and the before-defense
    ASR/BER check will reveal a mismatch.
    """
    target_sha = attack_meta.get("payload_sha256") if isinstance(attack_meta, dict) else None
    candidates = [
        DEFAULT_TOY_PAYLOAD.decode("utf-8"),
        "INSPECT_SASER_QWEN3_TOY_PAYLOAD_2026",
        "INSPECT_SASER_CHATGLM3_TOY_PAYLOAD_2026",
        "INSPECT_SASER_LLAMA31_TOY_PAYLOAD_2026",
        "INSPECT_SASER_LLAMA3_TOY_PAYLOAD_2026",
    ]
    seen = set()
    for text in candidates:
        if text in seen:
            continue
        seen.add(text)
        if target_sha and hashlib.sha256(text.encode("utf-8")).hexdigest() == target_sha:
            print(f"[+] Payload matched attack_result sha256: {text}", flush=True)
            return text
    if target_sha:
        cur_sha = hashlib.sha256(default_text.encode("utf-8")).hexdigest()
        print(f"[warn] Could not infer payload from attack_result sha256={target_sha}; using --payload-text sha256={cur_sha}", flush=True)
    return default_text


@torch.no_grad()
def eval_ppl_acc_logits(model, tokenizer, runner: ChatGLMSASER, max_new_tokens: int = 8, show_progress: bool = False, desc: str = "eval", collect_logits: bool = False):
    model.eval()
    device = first_parameter_device(model)
    total_nll = 0.0
    total_tokens = 0
    correct = 0
    logits_cpu = [] if collect_logits else None
    iterator = tqdm(runner.eval_tasks, desc=desc, total=len(runner.eval_tasks), ncols=100) if show_progress else runner.eval_tasks
    for q, a in iterator:
        text = q + a
        enc = move_batch_to(tokenizer(text, return_tensors="pt"), device)
        try:
            out = model(**enc, use_cache=False, return_dict=True)
        except TypeError:
            out = model(**enc, return_dict=True)
        logits = getattr(out, "logits", None)
        if logits is None:
            logits = out[0]
        input_ids = enc["input_ids"]
        logits = normalize_3d_logits(logits, input_ids)
        if collect_logits:
            logits_cpu.append(logits[:, -1:, :].detach().float().cpu())

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        total_nll += float(loss.item())
        total_tokens += int(shift_labels.numel())

        prompt = move_batch_to(tokenizer(q, return_tensors="pt"), device)
        try:
            gen = model.generate(
                **prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=getattr(tokenizer, "pad_token_id", None),
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
                use_cache=False,
            )
        except TypeError:
            gen = model.generate(**prompt, max_new_tokens=max_new_tokens, do_sample=False)
        # Decode only newly generated tokens. For MMLU the prompt contains
        # all answer choices, so decoding the full prompt would inflate ACC.
        prompt_len = int(prompt["input_ids"].shape[1])
        decoded = tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)
        norm = decoded.strip().lower()
        ans = str(a).strip().lower()
        if norm.startswith(ans) or (norm[:12].find(ans) >= 0):
            correct += 1
    ppl = math.exp(total_nll / max(total_tokens, 1))
    acc = correct / max(len(runner.eval_tasks), 1)
    return ppl, acc, logits_cpu


@torch.no_grad()
def logit_mse_cpu(base_logits: List[torch.Tensor], cur_logits: List[torch.Tensor]) -> float:
    vals = []
    for a, b in zip(base_logits, cur_logits):
        if a.dim() == 3 and b.dim() == 3:
            t = min(a.shape[1], b.shape[1])
            vals.append(float(F.mse_loss(a[:, :t, :], b[:, :t, :], reduction="mean").item()))
        else:
            vals.append(float(F.mse_loss(a, b, reduction="mean").item()))
    return float(np.mean(vals)) if vals else 0.0


def dppl_inv(base_ppl: float, cur_ppl: float) -> float:
    if not math.isfinite(cur_ppl) or cur_ppl <= 0:
        return 1.0
    return abs((1.0 / base_ppl) - (1.0 / cur_ppl)) / max(1.0 / base_ppl, 1e-12)


def _norm01_np(vals: Dict[int, float]) -> Dict[int, float]:
    clean = {int(k): float(v) for k, v in vals.items() if v is not None and math.isfinite(float(v))}
    if not clean:
        return {}
    mn, mx = min(clean.values()), max(clean.values())
    if abs(mx - mn) < 1e-12:
        return {k: 1.0 for k in clean} if abs(mx) > 0 else {k: 0.0 for k in clean}
    return {k: (v - mn) / (mx - mn) for k, v in clean.items()}


def _norm_vec(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.float()
    x = x.float()
    mn = x.min()
    mx = x.max()
    if float((mx - mn).abs().detach().cpu()) < 1e-12:
        return torch.ones_like(x) if float(mx.abs().detach().cpu()) > 0 else torch.zeros_like(x)
    return ((x - mn) / (mx - mn)).clamp(0.0, 1.0)


def _main_matrix_name(local_name: str) -> bool:
    low = local_name.lower()
    if not low.endswith("weight"):
        return False
    if any(x in low for x in ["embedding", "embed", "lm_head", "norm", "layernorm", "bias"]):
        return False
    return any(k in low for k in [
        "attention", "attn", "query_key_value", "dense", "mlp", "c_attn", "c_proj", "c_fc",
        "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj",
    ])


def _as_batch_seq_hidden(x: torch.Tensor):
    if isinstance(x, (tuple, list)):
        x = x[0]
    if not isinstance(x, torch.Tensor) or x.dim() != 3:
        return None
    # ChatGLM blocks may emit [T,B,D].  Most calibration batches are B=1.
    if x.shape[0] > 1 and x.shape[1] == 1:
        x = x.transpose(0, 1).contiguous()
    return x.detach()


def _detach_tree(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    if isinstance(x, tuple):
        return tuple(_detach_tree(v) for v in x)
    if isinstance(x, list):
        return [_detach_tree(v) for v in x]
    if isinstance(x, dict):
        return {k: _detach_tree(v) for k, v in x.items()}
    return x


def _to_device_tree(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, tuple):
        return tuple(_to_device_tree(v, device) for v in x)
    if isinstance(x, list):
        return [_to_device_tree(v, device) for v in x]
    if isinstance(x, dict):
        return {k: _to_device_tree(v, device) for k, v in x.items()}
    return x


@torch.no_grad()
def _capture_one_layer_io(model, layer, tokenizer, text: str):
    device = first_parameter_device(model)
    enc = move_batch_to(tokenizer(text, return_tensors="pt"), device)
    holder = {}

    # Qwen/LLaMA decoder layers often receive required values as kwargs
    # (attention_mask, position_ids, position_embeddings, cache_position, etc.).
    def pre_hook(_m, args, kwargs):
        holder["args"] = _detach_tree(args)
        holder["kwargs"] = _detach_tree(kwargs)

    def fwd_hook(_m, _args, out):
        holder["out"] = out[0].detach() if isinstance(out, (tuple, list)) and isinstance(out[0], torch.Tensor) else (out.detach() if isinstance(out, torch.Tensor) else None)

    h1 = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = layer.register_forward_hook(fwd_hook)
    try:
        try:
            _ = model(**enc, use_cache=False, return_dict=True)
        except TypeError:
            _ = model(**enc, return_dict=True)
    finally:
        h1.remove(); h2.remove()
    return holder.get("args"), holder.get("kwargs", {}), holder.get("out")


@torch.no_grad()
def compute_llm_compiler_scomp(model, tokenizer, runner: ChatGLMSASER, layers, scan_layers: Sequence[int], out_dir: str, token_block: int = 8):
    print("\n=== INSPECT-Opt Step 1A: compute LLM compiler relevance S_comp ===")
    os.makedirs(out_dir, exist_ok=True)
    texts = calibration_texts(runner)
    rows = []
    hidden_vectors: Dict[int, torch.Tensor] = {}
    eps = 1e-8
    for li in tqdm(scan_layers, desc="scan Scomp layers", ncols=100):
        layer = layers[int(li)]
        dim_scores_acc = []
        status = "ok"
        reason = ""
        try:
            compiled_layer = torch.compile(copy.deepcopy(layer).to(first_parameter_device(model)).eval(), fullgraph=False)
        except Exception as e:
            compiled_layer = None
            status = "compile_unavailable"
            reason = repr(e)[:240]
        if compiled_layer is not None:
            for text in texts:
                try:
                    inp, kw, eager_out = _capture_one_layer_io(model, layer, tokenizer, text)
                    if inp is None or eager_out is None:
                        continue
                    device = first_parameter_device(model)
                    inp2 = _to_device_tree(inp, device)
                    kw2 = _to_device_tree(kw, device)
                    comp_out = compiled_layer(*inp2, **kw2)
                    h_e = _as_batch_seq_hidden(eager_out)
                    h_c = _as_batch_seq_hidden(comp_out)
                    if h_e is None or h_c is None or h_e.shape != h_c.shape:
                        continue
                    b, t, d = h_e.shape
                    cur = torch.zeros(d, dtype=torch.float64)
                    for s0 in range(0, t, int(token_block)):
                        a = h_e[:, s0:s0 + int(token_block), :].double().cpu()
                        c = h_c[:, s0:s0 + int(token_block), :].double().cpu()
                        if a.numel() == 0:
                            continue
                        num = ((a - c) ** 2).mean(dim=(0, 1))
                        den = (a ** 2).mean(dim=(0, 1)) + eps
                        cur = torch.maximum(cur, num / den)
                    dim_scores_acc.append(cur.float())
                except Exception as e:
                    status = "runtime_partial_or_failed"
                    reason = repr(e)[:240]
                    continue
        if dim_scores_acc:
            vec = torch.stack(dim_scores_acc).mean(dim=0)
            hidden_vectors[int(li)] = vec
            score = float(vec.max().item())
            mean_score = float(vec.mean().item())
        else:
            vec = None
            score = 0.0
            mean_score = 0.0
            if status == "ok":
                status = "no_valid_drift"
        rows.append(dict(layer=int(li), scomp_raw=score, scomp_mean=mean_score, status=status, reason=reason))
        print(f"  - Layer {int(li):02d}: Scomp_raw={score:.8g}, mean={mean_score:.8g}, status={status}")
        try:
            del compiled_layer
        except Exception:
            pass
        clear_gpu()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "inspect_llm_compiler_scomp.csv"), index=False)
    return df, hidden_vectors



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
    if args.eval_dataset == "builtin":
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

def compute_llm_stego_pai(model, tokenizer, runner: ChatGLMSASER, q_bits: int, m_lsb: int, scan_layers: Sequence[int], out_dir: str, seed: int) -> pd.DataFrame:
    print("\n=== INSPECT-Opt Step 1B: compute LLM layer-base PAI for S_stego ===")
    rng = np.random.default_rng(seed)
    base_ppl, base_acc, base_logits = eval_ppl_acc_logits(model, tokenizer, runner, show_progress=True, desc="PAI base ref", collect_logits=True)
    print(f"[+] Defense reference before temporary probing: PPL={base_ppl:.6f}, ACC={base_acc*100:.2f}%")
    records = []
    probe_len = len(StegoProtocol.pack_payload(DEFAULT_TOY_PAYLOAD, m_lsb, use_ecc=True))
    for layer_idx in tqdm(scan_layers, desc="scan Sstego/PAI layers", ncols=100):
        params = runner.get_layer_params(layer_idx)
        if not params:
            continue
        backups = [(name, p, p.data.detach().clone()) for name, p in params]
        rand_bits = rng.integers(0, 2, size=probe_len, dtype=np.int8).astype(int).tolist()
        runner.launch_attack_robust(params, rand_bits, n=m_lsb, q_bits=q_bits)
        cur_ppl, cur_acc, cur_logits = eval_ppl_acc_logits(model, tokenizer, runner, show_progress=True, desc=f"PAI probe L{layer_idx:02d}", collect_logits=True)
        d_ppl = dppl_inv(base_ppl, cur_ppl)
        d_logit = logit_mse_cpu(base_logits, cur_logits)
        pai = max(d_ppl, d_logit)
        records.append(dict(layer=layer_idx, m=m_lsb, pai=pai, d_ppl=d_ppl, d_logit_mse=d_logit, ppl=cur_ppl, acc=cur_acc))
        for _, p, old in backups:
            p.data.copy_(old)
        del backups, cur_logits
        clear_gpu()
        print(f"  - Layer {layer_idx:02d}: PAI={pai:.8f}, Dppl={d_ppl:.8f}, DlogitMSE={d_logit:.8f}")
    df = pd.DataFrame(records).sort_values(["pai", "layer"]).reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "inspect_llm_stego_pai.csv"), index=False)
    return df


def read_or_compute_pai(args, model, tokenizer, runner, layers) -> pd.DataFrame:
    if args.stego_pai_source == "csv":
        if not args.target_search_csv:
            raise ValueError("--stego-pai-source csv requires --target-search-csv")
        df = pd.read_csv(args.target_search_csv)
        if not {"layer", "pai"}.issubset(df.columns):
            raise ValueError(f"CSV must contain layer,pai columns; got {list(df.columns)}")
        df = df.copy()
        df.to_csv(os.path.join(args.out_dir, "defense_stego_pai_used.csv"), index=False)
        return df
    max_layers = len(layers) if args.max_scan_layers < 0 else min(args.max_scan_layers, len(layers))
    return compute_llm_stego_pai(model, tokenizer, runner, args.q_bits, args.n, list(range(max_layers)), args.out_dir, args.seed + 100)


def select_layers_union(scomp_df: pd.DataFrame, pai_df: pd.DataFrame, args, out_dir: str):
    comp_layers = []
    if scomp_df is not None and len(scomp_df):
        dcomp = scomp_df.copy()
        dcomp["scomp_raw"] = dcomp["scomp_raw"].astype(float)
        dcomp = dcomp.sort_values(["scomp_raw", "layer"], ascending=[False, True])
        comp_layers = [int(x) for x in dcomp["layer"].tolist()[: max(0, int(args.comp_top_layers))]]

    stego_layers = []
    if pai_df is not None and len(pai_df):
        dpai = pai_df.copy()
        dpai["pai"] = dpai["pai"].astype(float)
        dpai = dpai.sort_values(["pai", "layer"], ascending=[True, True])
        stego_layers = [int(x) for x in dpai["layer"].tolist()[: max(0, int(args.stego_top_layers))]]

    if args.candidate_layers not in {"", "union", "auto"}:
        if str(args.candidate_layers).strip().lower() == "all":
            max_layer = max(list(scomp_df["layer"].astype(int)) + list(pai_df["layer"].astype(int)))
            union_layers = list(range(max_layer + 1))
        else:
            union_layers = sorted(set(int(x.strip()) for x in str(args.candidate_layers).split(",") if x.strip()))
    else:
        union_layers = sorted(set(comp_layers) | set(stego_layers))

    rows = []
    for l in union_layers:
        rows.append(dict(layer=int(l), selected_by_comp=int(l in set(comp_layers)), selected_by_stego=int(l in set(stego_layers))))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "defense_union_selected_layers.csv"), index=False)
    print(f"[+] Scomp-selected layers: {comp_layers}")
    print(f"[+] Sstego-selected layers: {stego_layers}")
    print(f"[+] UNION candidate layers: {union_layers}")
    return comp_layers, stego_layers, union_layers


def _hidden_scores_for_indices(hidden_vec: torch.Tensor, shape, idx: torch.Tensor) -> torch.Tensor:
    if hidden_vec is None or not isinstance(hidden_vec, torch.Tensor) or hidden_vec.numel() == 0:
        return torch.zeros(int(idx.numel()), dtype=torch.float32, device=idx.device)
    hv = hidden_vec.float().to(idx.device).reshape(-1)
    d = int(hv.numel())
    flat_idx = idx.long().to(idx.device)
    if len(shape) >= 2:
        if len(shape) == 2:
            rows = torch.div(flat_idx, int(shape[1]), rounding_mode="floor")
            cols = flat_idx % int(shape[1])
        else:
            rows = torch.div(flat_idx, int(np.prod(shape[1:])), rounding_mode="floor")
            cols = flat_idx % int(shape[1])
        vals = []
        if int(shape[1]) == d:
            vals.append(hv[cols.long().clamp(0, d - 1)])
        if int(shape[0]) == d:
            vals.append(hv[rows.long().clamp(0, d - 1)])
        if vals:
            return torch.stack(vals, dim=0).max(dim=0).values
    return torch.full((int(idx.numel()),), float(hv.mean().item()), dtype=torch.float32, device=idx.device)


def comp_indices_for_param(p: torch.nn.Parameter, hvec: torch.Tensor, q: torch.Tensor, q_bits: int, top_hidden_dims: int):
    stable = BlockQuantizer.stable_payload_mask_from_q(q, n=1, q_bits=q_bits)
    valid_mask = stable.reshape(-1)
    if hvec is None or hvec.numel() == 0 or top_hidden_dims <= 0:
        return torch.empty((0,), dtype=torch.long, device=p.device)
    hv = hvec.float().to(p.device).reshape(-1)
    d = int(hv.numel())
    k = min(int(top_hidden_dims), d)
    top_dims = torch.topk(hv, k=k, largest=True).indices.long()
    shape = tuple(p.shape)
    idx_parts = []
    if p.ndim == 2:
        rows_n, cols_n = int(shape[0]), int(shape[1])
        if cols_n == d:
            # selected input hidden dimensions => selected columns
            rows = torch.arange(rows_n, device=p.device).view(-1, 1)
            cols = top_dims.view(1, -1)
            idx_parts.append((rows * cols_n + cols).reshape(-1))
        if rows_n == d:
            # selected output hidden dimensions => selected rows
            rows = top_dims.view(-1, 1)
            cols = torch.arange(cols_n, device=p.device).view(1, -1)
            idx_parts.append((rows * cols_n + cols).reshape(-1))
    # For unusual shapes, fall back to no comp indices rather than whole tensor.
    if not idx_parts:
        return torch.empty((0,), dtype=torch.long, device=p.device)
    idx = torch.unique(torch.cat(idx_parts).long())
    idx = idx[idx < q.numel()]
    idx = idx[valid_mask[idx]]
    return idx


def stego_stream_indices_for_layer(layer_params, payload_bits_len: int, q_bits: int, n: int):
    """Sstego parameter set: robust writable stream prefix in the lowest-PAI layer.

    This mirrors the SASER robust-mode write order but is derived from the
    Sstego-selected layer, not from the attack_target_layer as a defense prior.
    """
    need_bits = int(payload_bits_len)
    bit_ptr = 0
    out = {}
    for local_name, p in layer_params:
        if bit_ptr >= need_bits:
            break
        if p.dim() < 2 or not _main_matrix_name(local_name):
            continue
        q, _s, _sh, _pad = BlockQuantizer.quantize(p.data, q_bits)
        q = q.to(p.device)
        stable = BlockQuantizer.stable_payload_mask_from_q(q, n=n, q_bits=q_bits)
        valid = torch.nonzero(stable, as_tuple=False).view(-1).to(p.device)
        capacity = int(valid.numel()) * int(n)
        remaining = need_bits - bit_ptr
        bits_here = min(remaining, capacity)
        if bits_here > 0:
            slots = math.ceil(bits_here / max(int(n), 1))
            out[local_name] = valid[:slots].detach().clone().long()
            bit_ptr += bits_here
        del q, stable, valid
        clear_gpu()
    return out


def build_union_parameter_states(model, layers, layer_path: str, comp_layers: List[int], stego_layers: List[int], hidden_scomp: Dict[int, torch.Tensor], pai_df: pd.DataFrame, payload_bits_len: int, args):
    print("\n=== INSPECT-Opt Step 2: build UNION(Icomp, Istego) parameter set ===")
    device = first_parameter_device(model)
    comp_layer_set = set(int(x) for x in comp_layers)
    stego_layer_set = set(int(x) for x in stego_layers)
    union_layers = sorted(comp_layer_set | stego_layer_set)
    if not union_layers:
        raise RuntimeError("No union candidate layers.")

    # layer-level raw Sstego = 1/(PAI+eps), then normalize across all scanned layers.
    layer_pai = pai_df.groupby("layer")["pai"].min().to_dict()
    raw_stego = {int(k): 1.0 / (float(v) + float(args.s_eps)) for k, v in layer_pai.items()}
    norm_stego_layer = _norm01_np(raw_stego)

    pending = []
    all_comp_raw = []
    all_stego_raw = []
    csv_rows = []

    for li in union_layers:
        layer = layers[int(li)]
        stego_idx_by_local = {}
        if li in stego_layer_set:
            stego_idx_by_local = stego_stream_indices_for_layer(
                layer_weight_parameters(layer), payload_bits_len, args.q_bits, args.n
            )
        layer_stego_raw = float(raw_stego.get(int(li), 0.0))
        hvec = hidden_scomp.get(int(li))
        for local_name, p in layer_weight_parameters(layer):
            if p.dim() < 2 or not _main_matrix_name(local_name):
                continue
            global_name = f"{layer_path}.{int(li)}.{local_name}"
            q, scale, orig_shape, pad_len = BlockQuantizer.quantize(p.data, args.q_bits)
            q = q.to(p.device)
            scale = scale.to(p.device)
            real_n = int(q.numel() - int(pad_len))
            comp_idx = torch.empty((0,), dtype=torch.long, device=p.device)
            stego_idx = torch.empty((0,), dtype=torch.long, device=p.device)
            if li in comp_layer_set:
                comp_idx = comp_indices_for_param(p, hvec, q, args.q_bits, args.comp_top_hidden_dims)
            if li in stego_layer_set and local_name in stego_idx_by_local:
                stego_idx = stego_idx_by_local[local_name].to(p.device).long()
            if comp_idx.numel() == 0 and stego_idx.numel() == 0:
                del q, scale
                continue
            opt_idx = torch.unique(torch.cat([comp_idx, stego_idx]).long())
            if opt_idx.numel() == 0:
                del q, scale
                continue
            comp_raw = _hidden_scores_for_indices(hvec, tuple(p.shape), opt_idx).detach().float().cpu()
            stego_raw = torch.full((int(opt_idx.numel()),), layer_stego_raw, dtype=torch.float32)
            pending.append(dict(
                global_name=global_name, local_name=local_name, param=p, q=q, scale=scale,
                orig_shape=orig_shape, pad_len=pad_len, opt_idx=opt_idx, real_n=real_n,
                comp_raw=comp_raw, stego_raw=stego_raw,
                selected_by_comp=int(comp_idx.numel() > 0), selected_by_stego=int(stego_idx.numel() > 0),
                comp_positions=int(comp_idx.numel()), stego_positions=int(stego_idx.numel()), union_positions=int(opt_idx.numel()),
            ))
            all_comp_raw.append(comp_raw)
            all_stego_raw.append(stego_raw)
            csv_rows.append(dict(
                layer=int(li), global_name=global_name, scope_params=real_n,
                comp_positions=int(comp_idx.numel()), stego_positions=int(stego_idx.numel()), union_positions=int(opt_idx.numel()),
                selected_by_comp=int(comp_idx.numel() > 0), selected_by_stego=int(stego_idx.numel() > 0),
                raw_stego_layer=layer_stego_raw, norm_stego_layer=float(norm_stego_layer.get(int(li), 0.0)),
            ))
            print(f"  - {global_name:72s} comp={int(comp_idx.numel()):>7d} stego={int(stego_idx.numel()):>6d} union={int(opt_idx.numel()):>7d}")
            del comp_idx, stego_idx
            clear_gpu()

    if not pending:
        raise RuntimeError("Union(Icomp,Istego) produced zero parameter positions.")

    cat_comp = torch.cat(all_comp_raw) if all_comp_raw else torch.tensor([], dtype=torch.float32)
    cat_stego = torch.cat(all_stego_raw) if all_stego_raw else torch.tensor([], dtype=torch.float32)
    norm_comp = _norm_vec(cat_comp)
    norm_stego = _norm_vec(cat_stego)
    cursor = 0
    states = []
    rng = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    rng.manual_seed(int(args.seed) + 500)
    scope_total = 0
    optimized_total = 0
    for item in pending:
        ncur = int(item["opt_idx"].numel())
        svec = (norm_comp[cursor:cursor+ncur] + norm_stego[cursor:cursor+ncur]).to(device)
        cursor += ncur
        if args.init_noise > 0:
            u0 = (torch.rand((ncur,), device=device, dtype=torch.float32, generator=rng) * 2 - 1) * float(args.init_noise)
        else:
            u0 = torch.zeros((ncur,), device=device, dtype=torch.float32)
        state = TensorOptState(
            global_name=item["global_name"], local_name=item["local_name"], param=item["param"],
            q_base=item["q"], scale=item["scale"], orig_shape=item["orig_shape"], pad_len=item["pad_len"],
            opt_idx=item["opt_idx"].long(), u=torch.nn.Parameter(u0), s_attack=svec.detach(),
            real_param_count=item["real_n"], selected_by_comp=item["selected_by_comp"], selected_by_stego=item["selected_by_stego"],
        )
        states.append(state)
        scope_total += int(item["real_n"])
        optimized_total += ncur
    pd.DataFrame(csv_rows).to_csv(os.path.join(args.out_dir, "defense_union_parameter_set.csv"), index=False)
    print(f"[+] UNION parameter set tensors={len(states)}, optimized positions={optimized_total}, scope params={scope_total}")
    return states, scope_total, optimized_total, union_layers


def differentiable_forward_logits(model, tokenizer, text: str, patched_params: Dict[str, torch.Tensor]):
    device = first_parameter_device(model)
    enc = move_batch_to(tokenizer(text, return_tensors="pt"), device)
    kwargs = dict(enc)
    kwargs["use_cache"] = False
    kwargs["return_dict"] = True
    try:
        out = functional_call(model, patched_params, args=(), kwargs=kwargs, strict=False)
    except TypeError:
        kwargs.pop("use_cache", None)
        out = functional_call(model, patched_params, args=(), kwargs=kwargs, strict=False)
    logits = getattr(out, "logits", None)
    if logits is None:
        logits = out[0]
    return normalize_3d_logits(logits, enc["input_ids"])


def make_patched_params(states: List[TensorOptState]) -> Dict[str, torch.Tensor]:
    patched = {}
    for st in states:
        q_soft = st.q_base.to(torch.float32).reshape(-1).clone()
        q_soft[st.opt_idx] = q_soft[st.opt_idx] + st.u
        w_soft = dequantize_float(q_soft.view_as(st.q_base), st.scale, st.orig_shape, st.pad_len, st.param.dtype)
        patched[st.global_name] = w_soft
    return patched


def inspect_optimize(model, tokenizer, runner: ChatGLMSASER, states: List[TensorOptState], args):
    print("\n=== INSPECT-Opt Step 3: optimize PDF objective with continuous u ===")
    device = first_parameter_device(model)
    texts_all = calibration_texts(runner)
    if getattr(args, "opt_calib_limit", 0) and args.opt_calib_limit > 0:
        texts_all = texts_all[: int(args.opt_calib_limit)]
    opt_batch = max(1, int(getattr(args, "opt_calib_batch", 1)))
    print(f"[+] Optimization calibration texts={len(texts_all)}, opt_calib_batch={opt_batch}")

    # Store reference logits on CPU. Keeping all reference logits on GPU can OOM
    # for LLaMA2-6B + MMLU prompts during functional_call optimization.
    with torch.no_grad():
        base_logits = []
        for text in texts_all:
            enc = move_batch_to(tokenizer(text, return_tensors="pt"), device)
            try:
                out = model(**enc, use_cache=False, return_dict=True)
            except TypeError:
                out = model(**enc, return_dict=True)
            logits = getattr(out, "logits", None)
            if logits is None:
                logits = out[0]
            base_logits.append(normalize_3d_logits(logits, enc["input_ids"]).detach().float().cpu())
            del enc, out, logits
            clear_gpu()

    opt = torch.optim.Adam([st.u for st in states], lr=args.lr)
    history = []
    for step in range(1, args.opt_steps + 1):
        opt.zero_grad(set_to_none=True)
        patched = make_patched_params(states)
        lattack = torch.zeros((), device=device, dtype=torch.float32)
        ltri = torch.zeros((), device=device, dtype=torch.float32)
        lweight = torch.zeros((), device=device, dtype=torch.float32)
        for st in states:
            u = st.u
            abs_u = torch.sqrt(u * u + args.abs_eps)
            lattack = lattack - float(args.lambda_a) * torch.sum(abs_u * st.s_attack.to(device=abs_u.device, dtype=abs_u.dtype))
            ltri = ltri + (u.pow(2) * (1 - u.pow(2)).pow(2)).sum()
            w_soft = patched[st.global_name].float()
            lweight = lweight + F.mse_loss(w_soft, st.param.detach().float(), reduction="sum")

        # Rotating micro-batch for Lpres logits. This keeps the PDF loss term but
        # avoids retaining graphs for all calibration prompts in one backward pass.
        n_text = len(texts_all)
        start = ((step - 1) * opt_batch) % max(n_text, 1)
        batch_ids = [(start + j) % n_text for j in range(min(opt_batch, n_text))]
        logit_losses = []
        for bi in batch_ids:
            text = texts_all[bi]
            ref = base_logits[bi].to(device)
            logits = differentiable_forward_logits(model, tokenizer, text, patched)
            t = min(logits.shape[1], ref.shape[1])
            logit_losses.append(F.mse_loss(logits.float()[:, :t, :], ref[:, :t, :], reduction="mean"))
            del logits, ref
            clear_gpu()
        llogit = torch.stack(logit_losses).mean() if logit_losses else torch.zeros((), device=device, dtype=torch.float32)
        lpres = llogit + float(args.lambda_w) * lweight
        loss = lattack + float(args.lambda_p) * lpres + float(args.lambda_t) * ltri
        loss.backward()
        opt.step()
        with torch.no_grad():
            for st in states:
                st.u.clamp_(-1.0, 1.0)
        all_u = torch.cat([st.u.detach().reshape(-1) for st in states])
        rec = dict(
            step=step,
            loss=float(loss.detach().cpu()),
            Lattack=float(lattack.detach().cpu()),
            Lpres=float(lpres.detach().cpu()),
            Llogit=float(llogit.detach().cpu()),
            Lweight=float(lweight.detach().cpu()),
            Ltri=float(ltri.detach().cpu()),
            mean_abs_u=float(all_u.abs().mean().cpu()),
            frac_over_tau=float((all_u.abs() > args.tau).float().mean().cpu()),
            optimized_positions=int(all_u.numel()),
            opt_calib_batch=len(batch_ids),
        )
        history.append(rec)
        if step == 1 or step % args.print_every == 0 or step == args.opt_steps:
            print(f"  step {step:04d} | loss={rec['loss']:.6g} Lattack={rec['Lattack']:.6g} Lpres={rec['Lpres']:.6g} Ltri={rec['Ltri']:.6g} | mean|u|={rec['mean_abs_u']:.4f} frac>|tau|={rec['frac_over_tau']:.4f} batch={rec['opt_calib_batch']}")
        del patched, loss, lpres, llogit, lweight, ltri, lattack, all_u, logit_losses
        clear_gpu()
    return pd.DataFrame(history)


@torch.no_grad()
def apply_discrete_purification(states: List[TensorOptState], q_bits: int, tau: float):
    print("\n=== INSPECT-Opt Step 4: ternary discretization and integer-domain purification ===")
    q_max = 2 ** (q_bits - 1) - 1
    q_min = -q_max
    scope_total = 0
    changed = 0
    optimized = 0
    per_tensor = []
    for st in states:
        u = st.u.detach()
        delta_sparse = torch.zeros_like(u, dtype=torch.int16, device=u.device)
        delta_sparse[u > tau] = 1
        delta_sparse[u < -tau] = -1
        q_prime = st.q_base.to(torch.int16).reshape(-1).clone()
        q_prime[st.opt_idx] = (q_prime[st.opt_idx] + delta_sparse).clamp(q_min, q_max)
        q_prime = q_prime.view_as(st.q_base).to(torch.int8)
        chg = int((delta_sparse != 0).sum().item())
        scope_total += int(st.real_param_count)
        optimized += int(st.opt_idx.numel())
        changed += chg
        st.param.data.copy_(BlockQuantizer.dequantize(q_prime, st.scale, st.orig_shape, st.pad_len).to(st.param.device))
        per_tensor.append(dict(
            global_name=st.global_name,
            selected_by_comp=st.selected_by_comp,
            selected_by_stego=st.selected_by_stego,
            scope_params=st.real_param_count,
            optimized_positions=int(st.opt_idx.numel()),
            changed=chg,
            k_scope_percent=100.0 * chg / max(st.real_param_count, 1),
        ))
        print(f"  - {st.global_name:72s} changed={chg}/{st.real_param_count} scope ({100.0*chg/max(st.real_param_count,1):.6f}%), optimized={int(st.opt_idx.numel())}")
    k_scope = changed / max(scope_total, 1)
    print(f"[+] Final k*_scope = {changed}/{scope_total} = {100.0*k_scope:.6f}%")
    return dict(scope_total_params=scope_total, optimized_positions=optimized, changed_params=changed, k_star_scope=k_scope), pd.DataFrame(per_tensor)


def model_floating_param_total(model) -> int:
    total = 0
    for p in model.parameters():
        if torch.is_floating_point(p) and p.ndim >= 2:
            total += int(p.numel())
    return total


def extract_metrics(runner: ChatGLMSASER, params, payload: bytes, n: int, q_bits: int, use_ecc: bool):
    found, extracted_payload, decoded_bits = runner.extract_payload_robust(params, n_hint=n, q_bits=q_bits, use_ecc=use_ecc)
    orig_bits = StegoProtocol.bytes_to_bits(payload)
    if not found:
        return dict(found_trigger=False, asr_percent=0.0, ber_percent=50.0, byte_match_percent=0.0, extracted_sha256="")
    extracted_bits_len = min(len(decoded_bits), len(orig_bits))
    bit_errors = sum(1 for a, b in zip(decoded_bits[:extracted_bits_len], orig_bits[:extracted_bits_len]) if int(a) != int(b))
    bit_errors += abs(len(orig_bits) - extracted_bits_len)
    ber = 100.0 * bit_errors / max(len(orig_bits), 1)
    extracted_len = min(len(extracted_payload), len(payload))
    byte_match = sum(1 for a, b in zip(extracted_payload[:extracted_len], payload[:extracted_len]) if a == b)
    match_percent = 100.0 * byte_match / max(len(payload), 1)
    asr = 100.0 if extracted_payload[: len(payload)] == payload else 0.0
    return dict(found_trigger=True, asr_percent=asr, ber_percent=ber, byte_match_percent=match_percent, extracted_sha256=hashlib.sha256(extracted_payload).hexdigest() if extracted_payload else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./models/chatglm3-6b")
    parser.add_argument("--out-dir", default="./outputs/chatglm3_def")
    parser.add_argument("--q-bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--eval-limit", type=int, default=10, help="Only used for builtin tasks. For paper datasets use --max-calib/--max-eval.")
    parser.add_argument("--eval-dataset", choices=["builtin", "mmlu", "agieval", "paper"], default="builtin")
    parser.add_argument("--mmlu-subjects", default="all", help="Comma list or 'all'.")
    parser.add_argument("--mmlu-split", default="test")
    parser.add_argument("--agieval-configs", default="all", help="Comma list or 'all' for lighteval/agi_eval_en configs.")
    parser.add_argument("--agieval-split", default="test")
    parser.add_argument("--calib-ratio", type=float, default=0.1, help="Paper uses 1:9 calibration:evaluation, i.e. 0.1 calibration.")
    parser.add_argument("--max-calib", type=int, default=0, help="Optional cap for calibration tasks; 0 means full 10% paper split.")
    parser.add_argument("--max-eval", type=int, default=0, help="Optional cap for evaluation tasks; 0 means full 90% paper split.")
    parser.add_argument("--attack-result-json", default="", help="Attack result JSON. target_layer/n/q_bits are read from here only to reproduce the poison artifact for evaluation.")
    parser.add_argument("--attack-target-layer", type=int, default=-1, help="Fallback only if --attack-result-json is not provided.")
    parser.add_argument("--payload-text", default=DEFAULT_TOY_PAYLOAD.decode("utf-8"))
    parser.add_argument("--no-ecc", action="store_true")
    parser.add_argument("--device-map", default="none", choices=["auto", "none"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stego-pai-source", choices=["compute", "csv"], default="compute")
    parser.add_argument("--target-search-csv", default="")
    parser.add_argument("--max-scan-layers", type=int, default=-1)
    parser.add_argument("--token-block", type=int, default=8)
    parser.add_argument("--candidate-layers", default="union", help="union=Icomp ∪ Istego; all; or comma list.")
    parser.add_argument("--comp-top-layers", type=int, default=1)
    parser.add_argument("--stego-top-layers", type=int, default=1)
    parser.add_argument("--comp-top-hidden-dims", type=int, default=1)
    parser.add_argument("--opt-steps", type=int, default=30)
    parser.add_argument("--opt-calib-limit", type=int, default=0, help="Use at most this many calibration prompts inside INSPECT-Opt optimization; 0 means all calibration prompts.")
    parser.add_argument("--opt-calib-batch", type=int, default=1, help="Number of calibration prompts used per optimization step for the differentiable logits-preservation term.")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lambda-a", type=float, default=1.0)
    parser.add_argument("--lambda-p", type=float, default=1.0)
    parser.add_argument("--lambda-w", type=float, default=1e-4)
    parser.add_argument("--lambda-t", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=0.50)
    parser.add_argument("--abs-eps", type=float, default=1e-6)
    parser.add_argument("--s-eps", type=float, default=1e-6)
    parser.add_argument("--init-noise", type=float, default=1e-3)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--save-purified", action="store_true")
    args = parser.parse_args()

    attack_meta = {}
    if args.attack_result_json:
        with open(args.attack_result_json, "r", encoding="utf-8") as f:
            attack_meta = json.load(f)
        if args.attack_target_layer < 0:
            args.attack_target_layer = int(attack_meta["target_layer"])
        # Keep command-line q/n unless user left defaults; record source either way.
        if "n" in attack_meta:
            args.n = int(attack_meta["n"])
        if "q_bits" in attack_meta:
            args.q_bits = int(attack_meta["q_bits"])
    if args.attack_target_layer < 0:
        raise ValueError("Provide --attack-result-json or --attack-target-layer. This layer is only for poison reproduction/evaluation, not defense selection.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected.")
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[+] Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"[+] Loading model: {args.model_path}")
    load_kwargs = dict(trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
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
    for p in model.parameters():
        p.requires_grad_(False)

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
    args.payload_text = choose_payload_text(args.payload_text, attack_meta)
    payload = args.payload_text.encode("utf-8")
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if attack_meta.get("payload_sha256") and payload_sha256 != attack_meta.get("payload_sha256"):
        print(f"[warn] payload sha mismatch: defense={payload_sha256}, attack={attack_meta.get('payload_sha256')}", flush=True)

    print("\n=== Reproduce benign SASER stego attack in memory ===")
    attack_params = runner.get_layer_params(args.attack_target_layer)
    payload_bits = StegoProtocol.pack_payload(payload, args.n, use_ecc=use_ecc)
    written = runner.launch_attack_robust(attack_params, payload_bits, n=args.n, q_bits=args.q_bits)
    print(f"[+] Embedded bits: {written}/{len(payload_bits)} into layer {args.attack_target_layer}")

    before_extract = extract_metrics(runner, attack_params, payload, args.n, args.q_bits, use_ecc)
    runner.eval_tasks = eval_tasks
    before_ppl, before_acc, _ = eval_ppl_acc_logits(model, tokenizer, runner, show_progress=True, desc="before-defense eval", collect_logits=False)
    print(f"[+] Before defense on evaluation split: ASR={before_extract['asr_percent']:.1f}% BER={before_extract['ber_percent']:.4f}% Match={before_extract['byte_match_percent']:.2f}% PPL={before_ppl:.6f} ACC={before_acc*100:.2f}%")

    # Defense relevance and Lpres are computed on the calibration split, per paper protocol.
    runner.eval_tasks = calib_tasks
    max_layers = len(layers) if args.max_scan_layers < 0 else min(args.max_scan_layers, len(layers))
    scan_layer_ids = list(range(max_layers))
    scomp_df, hidden_scomp = compute_llm_compiler_scomp(model, tokenizer, runner, layers, scan_layer_ids, args.out_dir, token_block=args.token_block)
    pai_df = read_or_compute_pai(args, model, tokenizer, runner, layers)
    comp_layers, stego_layers, union_layers = select_layers_union(scomp_df, pai_df, args, args.out_dir)

    states, scope_total, optimized_total, candidate_layers = build_union_parameter_states(
        model, layers, layer_path, comp_layers, stego_layers, hidden_scomp, pai_df,
        payload_bits_len=len(payload_bits), args=args,
    )
    hist = inspect_optimize(model, tokenizer, runner, states, args)
    hist.to_csv(os.path.join(args.out_dir, "optimization_history.csv"), index=False)
    stat, tensor_df = apply_discrete_purification(states, args.q_bits, args.tau)
    tensor_df.to_csv(os.path.join(args.out_dir, "discrete_perturbation_by_tensor.csv"), index=False)

    after_extract = extract_metrics(runner, attack_params, payload, args.n, args.q_bits, use_ecc)
    runner.eval_tasks = eval_tasks
    after_ppl, after_acc, _ = eval_ppl_acc_logits(model, tokenizer, runner, show_progress=True, desc="after-defense eval", collect_logits=False)
    d_ppl = dppl_inv(before_ppl, after_ppl)
    d_acc = abs(before_acc - after_acc) / max(before_acc, 1e-6)
    pai_after_def = max(d_ppl, d_acc)
    sr_after_def = max(0.0, 100.0 * (1.0 - pai_after_def))
    model_total = model_floating_param_total(model)

    result = {
        "model_path": args.model_path,
        "dataset_meta": dataset_meta,
        "eval_dataset": args.eval_dataset,
        "eval_num_tasks": len(eval_tasks),
        "calib_num_tasks": len(calib_tasks),
        "attack_result_json": args.attack_result_json,
        "q_bits": args.q_bits,
        "n": args.n,
        "attack_target_layer": args.attack_target_layer,
        "candidate_layers": candidate_layers,
        "comp_selected_layers": comp_layers,
        "stego_selected_layers": stego_layers,
        "defense_scope": "union_of_scomp_and_sstego_parameter_sets",
        "candidate_layer_rule": "union_of_scomp_and_sstego",
        "comp_top_layers": args.comp_top_layers,
        "stego_top_layers": args.stego_top_layers,
        "comp_top_hidden_dims": args.comp_top_hidden_dims,
        "payload_position_prior": False,
        "payload_len_bytes": len(payload),
        "payload_sha256": payload_sha256,
        "before_defense": {**before_extract, "ppl": before_ppl, "acc_percent": before_acc * 100.0},
        "after_defense": {
            **after_extract,
            "ppl": after_ppl,
            "acc_percent": after_acc * 100.0,
            "d_ppl_vs_predefense": d_ppl,
            "d_acc_vs_predefense": d_acc,
            "pai_vs_predefense": pai_after_def,
            "sr_percent_vs_predefense": sr_after_def,
        },
        "inspect_opt": {
            "objective": "-lambda_a*sum|u|[Norm(Scomp)+Norm(Sstego)] + lambda_p*(logitMSE + lambda_w*weightMSE) + lambda_t*sum u^2(1-u^2)^2",
            "lambda_a": args.lambda_a,
            "lambda_p": args.lambda_p,
            "lambda_w": args.lambda_w,
            "lambda_t": args.lambda_t,
            "tau": args.tau,
            "opt_steps": args.opt_steps,
            "lr": args.lr,
            **stat,
            "model_total_2d_params": model_total,
            "k_star_scope_percent": 100.0 * stat["k_star_scope"],
            "k_star_model_percent": 100.0 * stat["changed_params"] / max(model_total, 1),
        },
    }
    with open(os.path.join(args.out_dir, "defense_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    pd.json_normalize(result).to_csv(os.path.join(args.out_dir, "defense_result.csv"), index=False)

    if args.save_purified:
        save_dir = os.path.join(args.out_dir, "purified_model")
        print(f"[+] Saving purified model to {save_dir}")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    print("\n=== FINAL DEFENSE RESULT ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n[+] Done. Outputs are in: {args.out_dir}")


if __name__ == "__main__":
    main()
