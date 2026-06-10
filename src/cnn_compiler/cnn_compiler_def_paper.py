#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN compiler-backdoor INSPECT-Opt defense script.

Purpose
-------
Run the same three-part optimized defense used in the project platform/LLM
stego defense, but for CNN compiler attacks produced by DLCompilerAttack:

  1) S_comp: compiler sensitivity, from eager/compiled feature discrepancy.
  2) S_stego/PAI: tensor-level robustness sensitivity, measured by temporary
     low-bit quantized perturbation and utility/logit drift.
  3) INSPECT-Opt: optimize continuous u with the PDF-style objective, discretize
     to Delta in {-1,0,+1}, write back W'=DeQuant(q+Delta), then evaluate.

The output directory intentionally mirrors the LLM/stego defense scripts:
  - inspect_cnn_compiler_scomp.csv
  - inspect_cnn_stego_pai.csv
  - defense_union_selected_layers.csv
  - defense_union_parameter_set.csv
  - optimization_history.csv
  - discrete_perturbation_by_tensor.csv
  - defense_result.json
  - defense_result.csv

This script does not use the clean model.  The clean model is kept only for
external comparison.  The defense target is the poisoned best.tar, evaluated by
DLCompilerAttack prediction-style eager/compiled protocol.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch


# -----------------------------------------------------------------------------
# Path and import helpers
# -----------------------------------------------------------------------------


def _expand(path: str | os.PathLike) -> str:
    return os.path.abspath(os.path.expanduser(str(path)))


def add_project_paths(platform_dir: str, repo_dir: str) -> None:
    """Add DLCompilerAttack and platform_opt to sys.path with repo first.

    The poisoned best.tar was pickled from the real DLCompilerAttack repo on the
    server, so `src.*` and `utils` should resolve to that repo, not to a copied
    lightweight third_party fallback.  `core.*` is then imported from platform.
    """
    repo = _expand(repo_dir)
    plat = _expand(platform_dir)
    third = os.path.join(plat, "third_party", "dlcompiler_attack")
    for p in [third, plat, repo]:
        # insert in reverse effect; final repo should be first after loop below
        if p in sys.path:
            sys.path.remove(p)
    for p in [third, plat, repo]:
        if os.path.isdir(p):
            sys.path.insert(0, p)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.detach().cpu().item()
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def tensor_score_summary(value: Any, expected_numel: int | None = None) -> Dict[str, Any]:
    if value is None:
        return {
            "score_kind": "none",
            "score_numel": 0,
            "score_mean": 0.0,
            "score_max": 0.0,
            "score_min": 0.0,
            "score_positive": 0,
        }
    if isinstance(value, torch.Tensor):
        t = value.detach().float().reshape(-1).cpu()
    elif isinstance(value, (list, tuple, np.ndarray)):
        t = torch.as_tensor(value, dtype=torch.float32).reshape(-1).cpu()
    else:
        try:
            fv = float(value)
        except Exception:
            fv = 0.0
        if expected_numel is None:
            expected_numel = 1
        t = torch.full((int(expected_numel),), fv, dtype=torch.float32)
    if t.numel() == 0:
        return {
            "score_kind": "tensor",
            "score_numel": 0,
            "score_mean": 0.0,
            "score_max": 0.0,
            "score_min": 0.0,
            "score_positive": 0,
        }
    return {
        "score_kind": "tensor" if t.numel() != 1 else "scalar",
        "score_numel": int(t.numel()),
        "score_mean": float(t.mean().item()),
        "score_max": float(t.max().item()),
        "score_min": float(t.min().item()),
        "score_positive": int((t > 0).sum().item()),
    }


def module_prefix(name: str) -> str:
    parts = str(name).split(".")
    if len(parts) <= 1:
        return str(name)
    return ".".join(parts[:-1])


def floating_param_total(model: torch.nn.Module) -> int:
    total = 0
    for _name, p in model.named_parameters():
        if isinstance(p, torch.Tensor) and torch.is_floating_point(p):
            total += int(p.numel())
    return total


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# DLCompilerAttack model/dataloader loading
# -----------------------------------------------------------------------------


def load_best_tar(path: str, device: str = "cpu", target_label: int | None = None):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    trigger = None
    model = None
    extra = None
    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2:
            trigger = obj[0]
            model = obj[1]
            extra = obj[2] if len(obj) > 2 else None
    elif isinstance(obj, dict):
        for k in ["bd_trigger", "trigger", "backdoor_trigger"]:
            if k in obj:
                trigger = obj[k]
                break
        for k in ["model", "save_model", "poison_model", "poisoned_model"]:
            if k in obj:
                model = obj[k]
                break
        extra = obj
    if model is None or trigger is None:
        raise RuntimeError("best.tar 中未找到模型或 trigger，请确认是 DLCompilerAttack 攻击完成后的 best.tar")
    if hasattr(model, "init"):
        try:
            model.init()
        except Exception:
            pass
    if target_label is not None:
        try:
            trigger.target_label = int(target_label)
        except Exception:
            pass
    if not hasattr(trigger, "target_label"):
        if target_label is None:
            raise RuntimeError("trigger 中没有 target_label，请指定 --target-label")
        trigger.target_label = int(target_label)
    for attr in ["trigger", "ori_trigger"]:
        v = getattr(trigger, attr, None)
        if isinstance(v, torch.Tensor):
            setattr(trigger, attr, v.to(device))
    if hasattr(trigger, "device"):
        try:
            trigger.device = torch.device(device)
        except Exception:
            pass
    return model, trigger, extra


def load_repo_dataloaders(repo_dir: str, task_id: int, train_batch: int, test_batch: int):
    """Load the exact repo dataloaders for task_id 3/4.

    The real server repo has signature:
        load_dataloader(task_id, is_shuffle, train_batch, test_batch, trigger_len=0)
    Older bundled copies may have fewer args; this wrapper handles both.
    """
    repo_dir = _expand(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    # Ensure we import the real repo utils.py, not platform third_party utils.py.
    if "utils" in sys.modules:
        mod = sys.modules["utils"]
        src = getattr(mod, "__file__", "") or ""
        if not os.path.abspath(src).startswith(repo_dir):
            del sys.modules["utils"]
    dl_utils = importlib.import_module("utils")
    try:
        got = dl_utils.load_dataloader(int(task_id), False, int(train_batch), int(test_batch))
    except TypeError:
        got = dl_utils.load_dataloader(False, int(train_batch), int(test_batch))
    if not isinstance(got, (list, tuple)):
        raise RuntimeError("load_dataloader 未返回 tuple/list")
    if len(got) == 2:
        train_loader, test_loader = got
        valid_loader = None
    elif len(got) >= 3:
        train_loader, valid_loader, test_loader = got[0], got[1], got[2]
    else:
        raise RuntimeError("load_dataloader 返回值数量异常")
    return train_loader, valid_loader, test_loader


# -----------------------------------------------------------------------------
# INSPECT-Opt wrapper with per-tensor output tables
# -----------------------------------------------------------------------------


def score_positive_count(value: Any, n: int) -> int:
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        t = value.detach().float().reshape(-1).cpu()
    elif isinstance(value, (list, tuple, np.ndarray)):
        t = torch.as_tensor(value, dtype=torch.float32).reshape(-1).cpu()
    else:
        try:
            return int(n) if float(value) > 0 else 0
        except Exception:
            return 0
    if t.numel() == 1:
        return int(n) if float(t.item()) > 0 else 0
    return int((t > 0).sum().item())



def make_trigger_preserve_batches(dataloader, trigger, device, max_batches, samples_per_batch=8):
    """Build triggered eager-preservation batches for L_trigger_pres."""
    out = []
    if dataloader is None or trigger is None or int(max_batches or 0) <= 0:
        return out

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= int(max_batches):
                break

            moved = {}
            for k, v in batch.items():
                moved[k] = v.to(device) if isinstance(v, torch.Tensor) else v

            if "input" not in moved:
                continue

            x = moved["input"]

            # Trigger-preservation is only a calibration loss.
            # Slice each preserve batch to avoid building a huge functional_call graph.
            n = int(samples_per_batch or 0)
            if n > 0 and x.shape[0] > n:
                for kk, vv in list(moved.items()):
                    if isinstance(vv, torch.Tensor) and vv.dim() > 0 and vv.shape[0] == x.shape[0]:
                        moved[kk] = vv[:n].contiguous()
                x = moved["input"]

            try:
                moved["input"] = trigger.add_trigger(x.clone()).detach()
            except TypeError:
                moved["input"] = trigger.add_trigger(x).detach()

            out.append(moved)

    return out



def _score_to_flat_tensor(value, n):
    if value is None:
        return torch.zeros(int(n), dtype=torch.float32)
    if isinstance(value, torch.Tensor):
        t = value.detach().float().reshape(-1).cpu()
    else:
        try:
            t = torch.as_tensor(value, dtype=torch.float32).reshape(-1).cpu()
        except Exception:
            try:
                return torch.full((int(n),), float(value), dtype=torch.float32)
            except Exception:
                return torch.zeros(int(n), dtype=torch.float32)

    if t.numel() == 1:
        return torch.full((int(n),), float(t.item()), dtype=torch.float32)
    if t.numel() < int(n):
        out = torch.zeros(int(n), dtype=torch.float32)
        out[:t.numel()] = t
        return out
    return t[:int(n)].clone()


def _norm01_safe(x):
    x = x.detach().float().cpu()
    x[~torch.isfinite(x)] = 0.0
    mx = float(x.max().item()) if x.numel() else 0.0
    mn = float(x.min().item()) if x.numel() else 0.0
    if mx <= mn:
        return torch.zeros_like(x)
    return (x - mn) / (mx - mn + 1e-12)




def _compiler_auto_budget(model, strict_scores, args):
    """Auto budget for CNN compiler INSPECT-Opt.

    Budget is derived from Scomp/M1 parameter scale rather than a fixed 4096.
    """
    params = {
        n: p for n, p in model.named_parameters()
        if getattr(p, "dim", lambda: 0)() >= 2
    }
    scomp = strict_scores.get("scomp") or {}
    sstego = strict_scores.get("sstego") or {}

    scomp_scope = 0
    for name in scomp.keys():
        if name in params:
            scomp_scope += int(params[name].numel())

    if scomp_scope <= 0:
        vals = [int(p.numel()) for p in params.values()]
        scomp_scope = min(vals) if vals else 4096

    stego_tensor_count = len([n for n in sstego.keys() if n in params])
    stego_tensor_count = max(1, stego_tensor_count)

    auto_scomp_budget = max(1, int(round(float(args.auto_scomp_param_ratio) * scomp_scope)))
    auto_stego_per_tensor = max(1, int(round(float(args.auto_stego_tensor_ratio) * scomp_scope)))

    # build_balanced_union_scores uses:
    # scomp_budget = max_opt_positions * scomp_min_share
    # non_m1_cap   = max_opt_positions * non_m1_tensor_cap_ratio
    need_by_scomp = auto_scomp_budget / max(1e-12, float(args.scomp_min_share))
    need_by_stego = auto_stego_per_tensor / max(1e-12, float(args.non_m1_tensor_cap_ratio))

    effective_max = int(round(max(need_by_scomp, need_by_stego)))
    effective_max = max(1, effective_max)

    return effective_max, {
        "auto_budget_enabled": True,
        "scomp_scope": int(scomp_scope),
        "stego_tensor_count": int(stego_tensor_count),
        "auto_scomp_param_ratio": float(args.auto_scomp_param_ratio),
        "auto_stego_tensor_ratio": float(args.auto_stego_tensor_ratio),
        "auto_scomp_budget": int(auto_scomp_budget),
        "auto_stego_per_tensor": int(auto_stego_per_tensor),
        "effective_max_opt_positions": int(effective_max),
    }

def build_balanced_union_scores(
    model,
    strict_scores,
    max_opt_positions=4096,
    scomp_min_share=0.30,
    non_m1_tensor_cap_ratio=0.05,
    pai_floor=1e-3,
):
    """Build a balanced Ccomp∪Cstego score dict.

    It keeps the paper-level idea Sattack=Norm(Scomp)+Norm(Sstego),
    but prevents a zero-PAI tensor from monopolizing the whole budget.
    """
    params = {n: p for n, p in model.named_parameters() if getattr(p, "dim", lambda: 0)() >= 2}
    raw_scomp = strict_scores.get("scomp") or {}
    raw_sstego = strict_scores.get("sstego") or {}
    raw_pai = strict_scores.get("pai") or {}

    cap = int(max_opt_positions)
    cap = max(1, cap)
    scomp_budget = max(1, int(round(cap * float(scomp_min_share))))
    non_m1_cap = max(1, int(round(cap * float(non_m1_tensor_cap_ratio))))

    # ---- collect compiler branch candidates ----
    comp_items = []
    for name, param in params.items():
        n = param.numel()
        v = _score_to_flat_tensor(raw_scomp.get(name), n)
        pos = torch.nonzero(torch.isfinite(v) & (v > 0), as_tuple=False).flatten()
        if pos.numel() == 0:
            continue
        vn = _norm01_safe(v)
        if float(vn.max().item()) <= 0:
            vn[pos] = 1.0
        for idx in pos.tolist():
            comp_items.append((float(vn[idx].item()), name, int(idx)))

    comp_items.sort(key=lambda x: x[0], reverse=True)
    comp_keep = comp_items[:min(len(comp_items), scomp_budget)]

    selected = {}
    source = {}
    for score, name, idx in comp_keep:
        selected.setdefault(name, set()).add(idx)
        source[(name, idx)] = "scomp"

    # ---- collect stego branch candidates, with PAI floor and per-tensor cap ----
    remaining = max(0, cap - sum(len(v) for v in selected.values()))

    tensor_stego_scores = []
    for name, param in params.items():
        n = param.numel()
        if name not in raw_sstego and name not in raw_pai:
            continue

        # Prefer a clipped PAI-derived scalar when PAI is available.
        pai_val = raw_pai.get(name, None)
        if pai_val is not None:
            try:
                pai_f = float(pai_val)
                if not (pai_f == pai_f) or pai_f < 0:
                    pai_f = float(pai_floor)
                stego_scalar = 1.0 / max(float(pai_floor), pai_f)
                v = torch.full((n,), stego_scalar, dtype=torch.float32)
            except Exception:
                v = _score_to_flat_tensor(raw_sstego.get(name), n)
        else:
            v = _score_to_flat_tensor(raw_sstego.get(name), n)

        v[~torch.isfinite(v)] = 0.0
        if v.numel() == 0 or float(v.max().item()) <= 0:
            continue
        tensor_stego_scores.append((float(v.max().item()), name, v))

    # Normalize tensor-level stego strength to [0,1].
    if tensor_stego_scores:
        vals = torch.tensor([x[0] for x in tensor_stego_scores], dtype=torch.float32)
        vals_n = _norm01_safe(vals)
    else:
        vals_n = torch.tensor([], dtype=torch.float32)

    stego_ranked = []
    for rank_i, (_, name, v) in enumerate(tensor_stego_scores):
        tensor_score = float(vals_n[rank_i].item()) if vals_n.numel() else 0.0
        if tensor_score <= 0:
            tensor_score = 1e-6

        # non-M1 tensor cap prevents one normal layer from eating all budget.
        is_comp_tensor = name in raw_scomp
        this_cap = non_m1_cap
        if is_comp_tensor:
            # m1 is already covered by Scomp; do not let Sstego duplicate it too much.
            this_cap = max(1, min(non_m1_cap, int(round(cap * 0.02))))

        pos = torch.nonzero(torch.isfinite(v) & (v > 0), as_tuple=False).flatten()
        if pos.numel() == 0:
            continue

        # Stable deterministic selection: top values, then index order.
        vals_here = v[pos].float()
        k = min(int(this_cap), int(pos.numel()))
        if k <= 0:
            continue
        _, order = torch.topk(vals_here, k=k)
        keep_idx = pos[order].tolist()
        stego_ranked.append((tensor_score, name, keep_idx))

    stego_ranked.sort(key=lambda x: x[0], reverse=True)

    for tensor_score, name, keep_idx in stego_ranked:
        if remaining <= 0:
            break
        for idx in keep_idx:
            if remaining <= 0:
                break
            if idx in selected.get(name, set()):
                continue
            selected.setdefault(name, set()).add(int(idx))
            source[(name, int(idx))] = "sstego"
            remaining -= 1

    # ---- write back balanced score vectors ----
    new_scomp = {}
    new_sstego = {}
    rows = []
    for name, param in params.items():
        n = param.numel()
        comp_vec = torch.zeros(n, dtype=torch.float32)
        stego_vec = torch.zeros(n, dtype=torch.float32)
        inds = sorted(selected.get(name, set()))
        comp_count = 0
        stego_count = 0
        for idx in inds:
            if source.get((name, idx)) == "scomp":
                comp_vec[idx] = 1.0
                comp_count += 1
            else:
                stego_vec[idx] = 1.0
                stego_count += 1
        if comp_count > 0:
            new_scomp[name] = comp_vec
        if stego_count > 0:
            new_sstego[name] = stego_vec
        if comp_count + stego_count > 0:
            rows.append({
                "tensor_name": name,
                "layer_name": module_prefix(name),
                "real_param_count": int(n),
                "balanced_scomp_positions": int(comp_count),
                "balanced_sstego_positions": int(stego_count),
                "balanced_total_positions": int(comp_count + stego_count),
                "balanced_scope_percent": 100.0 * float(comp_count + stego_count) / max(1, n),
                "raw_pai": raw_pai.get(name),
                "is_scomp_tensor": int(name in raw_scomp),
            })

    out = dict(strict_scores)
    out["scomp"] = new_scomp
    out["sstego"] = new_sstego
    out["summary"] = dict(strict_scores.get("summary", {}))
    out["summary"]["balanced_union_positions"] = int(sum(len(v) for v in selected.values()))
    out["summary"]["balanced_scomp_tensors"] = int(len(new_scomp))
    out["summary"]["balanced_sstego_tensors"] = int(len(new_sstego))
    return out, pd.DataFrame(rows)

def apply_inspect_opt_with_tables(
    model: torch.nn.Module,
    *,
    strict_scores: Dict[str, Any],
    dataloader,
    device: str,
    target_names: List[str],
    q_bits: int = 8,
    block_size: int = 32,
    seed: int = 2026,
    opt_max_batches: int = 3,
    opt_steps: int = 30,
    opt_lr: float = 0.05,
    opt_tau: float = 0.50,
    lambda_a: float = 1.0,
    lambda_p: float = 1.0,
    lambda_t: float = 0.01,
    lambda_w: float = 1e-4,
    max_opt_positions: int = 4096,
    trigger_preserve_batches=None,
    lambda_trigger_p: float = 0.0,
):
    from core.defense import (
        _apply_inspect_records,
        _build_continuous_u_plan,
        _candidate_param_items_from_model,
        _optimized_deltas_from_pdf_loss,
        _q_records_for_items,
        _records_param_total,
    )

    model = model.to(device)
    items = _candidate_param_items_from_model(model, target_names=target_names, include_bias=False, include_buffers=False)
    records = _q_records_for_items(items, q_bits=q_bits, block_size=block_size, scale_mode="block")
    plan, total = _build_continuous_u_plan(
        records,
        k_percent=0.0,
        q_bits=q_bits,
        block_size=block_size,
        attack_type="cnn_compiler",
        target_names=target_names,
        strict_scores=strict_scores,
        max_opt_positions=max_opt_positions,
    )
    plan_count_by_record = {i: 0 for i in range(len(records))}
    for item in plan:
        ri = int(item["record_index"])
        plan_count_by_record[ri] = plan_count_by_record.get(ri, 0) + int(item["idx"].numel())

    selected, stat = _optimized_deltas_from_pdf_loss(
        model,
        records,
        plan,
        dataloader=dataloader,
        device=device,
        q_bits=q_bits,
        block_size=block_size,
        k_percent=0.0,
        seed=seed,
        max_batches=opt_max_batches,
        steps=opt_steps,
        lr=opt_lr,
        tau=opt_tau,
        lambda_a=lambda_a,
        lambda_p=lambda_p,
        lambda_t=lambda_t,
        lambda_w=lambda_w,
        preserve_module_name=None,
        extra_preserve_batches=trigger_preserve_batches,
        lambda_trigger_p=lambda_trigger_p,
    )
    _apply_inspect_records(records, selected, q_bits=q_bits, block_size=block_size, mode="pm1")

    tensor_rows = []
    param_rows = []
    scomp = strict_scores.get("scomp") or {}
    sstego = strict_scores.get("sstego") or {}
    pai = strict_scores.get("pai") or {}
    for ri, rec in enumerate(records):
        name = rec[0]
        n = int(_records_param_total([rec]))
        changed = len(selected.get(int(ri), []))
        comp_pos = score_positive_count(scomp.get(name), n)
        stego_pos = score_positive_count(sstego.get(name), n)
        row = {
            "tensor_name": name,
            "layer_name": module_prefix(name),
            "real_param_count": n,
            "selected_by_comp": comp_pos,
            "selected_by_stego": stego_pos,
            "union_candidate": int((comp_pos + stego_pos) > 0),
            "optimized_positions": int(plan_count_by_record.get(ri, 0)),
            "changed_params": int(changed),
            "changed_scope_percent": 100.0 * float(changed) / max(1, n),
            "pai": pai.get(name),
        }
        param_rows.append(row)
        tensor_rows.append({
            "tensor_name": name,
            "real_param_count": n,
            "optimized_positions": int(plan_count_by_record.get(ri, 0)),
            "changed_params": int(changed),
            "changed_scope_percent": 100.0 * float(changed) / max(1, n),
        })
    stat = dict(stat)
    stat["total"] = int(total)
    stat["algorithm"] = "pdf_continuous_u_loss"
    return stat, pd.DataFrame(param_rows), pd.DataFrame(tensor_rows)


def write_score_tables(out_dir: str, strict_scores: Dict[str, Any], model: torch.nn.Module) -> None:
    params = dict(model.named_parameters())
    rows = []
    for name, score in (strict_scores.get("scomp") or {}).items():
        n = int(params[name].numel()) if name in params else None
        d = tensor_score_summary(score, n)
        d.update({"tensor_name": name, "layer_name": module_prefix(name)})
        rows.append(d)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "inspect_cnn_compiler_scomp.csv"), index=False)

    rows = []
    sstego = strict_scores.get("sstego") or {}
    pai = strict_scores.get("pai") or {}
    for name, score in sstego.items():
        n = int(params[name].numel()) if name in params else None
        d = tensor_score_summary(score, n)
        d.update({"tensor_name": name, "layer_name": module_prefix(name), "pai": pai.get(name)})
        rows.append(d)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "inspect_cnn_stego_pai.csv"), index=False)


def write_layer_union(out_dir: str, param_df: pd.DataFrame) -> None:
    if param_df.empty:
        pd.DataFrame([]).to_csv(os.path.join(out_dir, "defense_union_selected_layers.csv"), index=False)
        return
    rows = []
    for layer, sub in param_df.groupby("layer_name"):
        rows.append({
            "layer_name": layer,
            "num_tensors": int(len(sub)),
            "selected_by_comp": int((sub["selected_by_comp"] > 0).any()),
            "selected_by_stego": int((sub["selected_by_stego"] > 0).any()),
            "union_selected": 1,
            "real_param_count": int(sub["real_param_count"].sum()),
            "optimized_positions": int(sub["optimized_positions"].sum()),
            "changed_params": int(sub["changed_params"].sum()),
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "defense_union_selected_layers.csv"), index=False)


def write_optimization_history(out_dir: str, stat: Dict[str, Any], tau: float) -> None:
    rows = []
    for rec in stat.get("u_trace", []) or []:
        vals = rec.get("values", []) or []
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            rows.append({"step": int(rec.get("step", 0)), "num_values": 0})
        else:
            rows.append({
                "step": int(rec.get("step", 0)),
                "num_values": int(arr.size),
                "mean_abs_u": float(np.mean(np.abs(arr))),
                "max_abs_u": float(np.max(np.abs(arr))),
                "min_u": float(np.min(arr)),
                "max_u": float(np.max(arr)),
                "frac_over_tau": float(np.mean(np.abs(arr) > float(tau))),
                "loss_final": stat.get("loss_final"),
                "loss_attack": stat.get("loss_attack"),
                "loss_pres": stat.get("loss_pres"),
                "loss_tri": stat.get("loss_tri"),
            })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "optimization_history.csv"), index=False)


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------


@dataclass
class ModelSpec:
    key: str
    display_name: str
    task_id: int
    num_classes: int
    poison_tar: str
    train_batch: int
    test_batch: int
    output_num: int = 2


DEFAULT_SPECS = {
    "c100_v19": ModelSpec(
        key="c100_v19",
        display_name="C100-V19 / VGG19-CIFAR100",
        task_id=3,
        num_classes=100,
        poison_tar="~/compiler_final_models/C100_V19/poison_vgg19_cifar100_torchcl_gpu_best.tar",
        train_batch=100,
        test_batch=100,
    ),
    "tiny_r34": ModelSpec(
        key="tiny_r34",
        display_name="Tiny-R34 / ResNet34-TinyImageNet",
        task_id=4,
        num_classes=200,
        poison_tar="~/compiler_final_models/Tiny_R34/poison_resnet34_tiny_torchcl_gpu_best.tar",
        train_batch=100,
        test_batch=100,
    ),
}


def run_one(args: argparse.Namespace, spec: ModelSpec) -> Dict[str, Any]:
    from core.compiler_eval import evaluate_compiler_protocol
    from core.defense import backup_module_params, restore_module_params
    from core.inspect_opt import build_strict_inspect_scores
    from core.pipeline import apply_saved_compiler_baseline, inspect_m1_module_name, inspect_union_target_names, saved_compiler_metrics

    out_dir = _expand(os.path.join(args.out_dir, spec.key))
    os.makedirs(out_dir, exist_ok=True)
    device = args.device
    print(f"\n===== CNN compiler defense: {spec.display_name} =====", flush=True)
    print(f"[+] out_dir={out_dir}", flush=True)
    print(f"[+] poison_tar={_expand(spec.poison_tar)}", flush=True)

    set_seed(args.seed)
    train_loader, valid_loader, test_loader = load_repo_dataloaders(
        args.repo_dir,
        spec.task_id,
        args.train_batch or spec.train_batch,
        args.test_batch or spec.test_batch,
    )
    calib_loader = train_loader
    eval_loader = test_loader

    model, trigger, extra = load_best_tar(_expand(spec.poison_tar), device=device, target_label=(args.target_label if args.override_target_label else None))
    if hasattr(model, "eval"):
        model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    model_total = floating_param_total(model)
    print(f"[+] Loaded poisoned model; floating params={model_total}; target_label={getattr(trigger, 'target_label', None)}", flush=True)

    t0 = time.perf_counter()
    before_live = evaluate_compiler_protocol(
        model,
        trigger,
        eval_loader,
        device=device,
        num_classes=spec.num_classes,
        compile_backend=args.compile_backend,
        output_num=spec.output_num,
        cl_id=args.cl_id,
        hardware_id=args.hardware_id,
        max_batches=args.max_eval_batches,
        require_compiled=True,
    )
    pre_eval_time = time.perf_counter() - t0
    warnings: List[str] = []
    before = dict(before_live)
    saved = saved_compiler_metrics(extra)
    if saved and args.prefer_saved_baseline:
        before, saved_w = apply_saved_compiler_baseline(before, saved, prefer_saved=True)
        warnings.extend(saved_w)
    print(f"[+] Before defense: ASR={before.get('asr')} clean_acc={before.get('clean_accuracy')} replay_asr={before.get('replay_asr')}", flush=True)

    backup = backup_module_params(model)
    search_batches = int(args.opt_search_max_batches)
    m1_module = inspect_m1_module_name(model, "cnn_compiler", {"m1_module": args.m1_module} if args.m1_module else {})
    print(f"[+] INSPECT M1/module for Scomp: {m1_module}", flush=True)

    print("\n=== INSPECT-Opt Step 1A/1B: compute CNN Scomp and tensor-base PAI ===", flush=True)
    t_def0 = time.perf_counter()
    strict_scores = build_strict_inspect_scores(
        model,
        calib_loader,
        device=device,
        attack_type="cnn_compiler",
        q_bits=args.q_bits,
        block_size=args.block_size,
        target_names=None,
        max_batches=search_batches,
        lsb_m=args.pai_lsb_m,
        m1_module=m1_module,
        compile_backend=args.compile_backend,
        output_num=spec.output_num,
        cl_id=args.cl_id,
        hardware_id=args.hardware_id,
        compiler_trigger=None,
    )
    write_score_tables(out_dir, strict_scores, model)
    print(f"[+] score summary: {strict_scores.get('summary', {})}", flush=True)

    restore_module_params(model, backup)
    effective_max_opt_positions = args.max_opt_positions
    auto_budget_info = {
        "auto_budget_enabled": False,
        "effective_max_opt_positions": int(args.max_opt_positions),
    }
    if getattr(args, "auto_max_opt_positions", False):
        effective_max_opt_positions, auto_budget_info = _compiler_auto_budget(model, strict_scores, args)
        print(f"[+] auto compiler budget: {auto_budget_info}", flush=True)

    all_2d_param_names = {n for n, p in model.named_parameters() if getattr(p, "dim", lambda: 0)() >= 2}

    if args.candidate_rule == "balanced_union":
        strict_scores, balanced_df = build_balanced_union_scores(
            model,
            strict_scores,
            max_opt_positions=effective_max_opt_positions,
            scomp_min_share=args.scomp_min_share,
            non_m1_tensor_cap_ratio=args.non_m1_tensor_cap_ratio,
            pai_floor=args.pai_floor,
        )
        balanced_df.to_csv(os.path.join(out_dir, "balanced_union_budget.csv"), index=False)
        defense_target_names = balanced_df["tensor_name"].tolist() if len(balanced_df) else []
        candidate_desc = "balanced Ccomp∪Cstego union"
    elif args.candidate_rule == "scomp":
        defense_target_names = [
            n for n, v in (strict_scores.get("scomp") or {}).items()
            if n in all_2d_param_names and score_positive_count(v, 1) > 0
        ]
        candidate_desc = "Scomp-only"
    elif args.candidate_rule == "sstego":
        defense_target_names = [
            n for n, v in (strict_scores.get("sstego") or {}).items()
            if n in all_2d_param_names and score_positive_count(v, 1) > 0
        ]
        candidate_desc = "Sstego-only"
    else:
        defense_target_names = inspect_union_target_names(model, strict_scores, include_bias=False)
        candidate_desc = "Ccomp∪Cstego union"

    if not defense_target_names:
        raise RuntimeError(f"INSPECT-Opt 没有得到候选参数：candidate_rule={args.candidate_rule}")

    warnings.append(f"INSPECT-Opt 防御候选采用 {candidate_desc}，共 {len(defense_target_names)} 个参数张量")
    print(f"[+] {candidate_desc} candidate tensors: {len(defense_target_names)}", flush=True)

    trigger_preserve_batches = []
    if float(args.lambda_trigger_p) > 0:
        trigger_preserve_batches = make_trigger_preserve_batches(calib_loader, trigger, device, search_batches, args.trigger_preserve_batch_size)
        print(f"[+] L_trigger_pres enabled: lambda_trigger_p={args.lambda_trigger_p}, batches={len(trigger_preserve_batches)}", flush=True)

    print("\n=== INSPECT-Opt Step 2/3/4: build UNION set, optimize u, discretize Delta ===", flush=True)
    stat, param_df, tensor_df = apply_inspect_opt_with_tables(
        model,
        strict_scores=strict_scores,
        dataloader=calib_loader,
        device=device,
        target_names=defense_target_names,
        q_bits=args.q_bits,
        block_size=args.block_size,
        seed=args.seed,
        opt_max_batches=search_batches,
        opt_steps=args.opt_steps,
        opt_lr=args.lr,
        opt_tau=args.tau,
        lambda_a=args.lambda_a,
        lambda_p=args.lambda_p,
        lambda_t=args.lambda_t,
        lambda_w=args.lambda_w,
        max_opt_positions=effective_max_opt_positions,
        trigger_preserve_batches=trigger_preserve_batches,
        lambda_trigger_p=args.lambda_trigger_p,
    )
    defense_time = time.perf_counter() - t_def0
    param_df.to_csv(os.path.join(out_dir, "defense_union_parameter_set.csv"), index=False)
    tensor_df.to_csv(os.path.join(out_dir, "discrete_perturbation_by_tensor.csv"), index=False)
    write_layer_union(out_dir, param_df)
    write_optimization_history(out_dir, stat, args.tau)

    print(f"[+] Final k*_scope = {stat.get('selected', 0)}/{stat.get('total', 0)} = {100.0*float(stat.get('selected', 0))/max(1,float(stat.get('total', 1))):.6f}%", flush=True)

    t0 = time.perf_counter()
    after = evaluate_compiler_protocol(
        model,
        trigger,
        eval_loader,
        device=device,
        num_classes=spec.num_classes,
        compile_backend=args.compile_backend,
        output_num=spec.output_num,
        cl_id=args.cl_id,
        hardware_id=args.hardware_id,
        max_batches=args.max_eval_batches,
        require_compiled=True,
    )
    post_eval_time = time.perf_counter() - t0
    after["inspect_selected"] = int(stat.get("selected", 0))
    after["inspect_total"] = int(model_total)
    after["inspect_target_total"] = int(stat.get("total", 0))
    after["inspect_k_actual"] = 100.0 * float(stat.get("selected", 0)) / max(1.0, float(model_total))

    if args.save_defended_tar:
        save_path = os.path.join(out_dir, "defended_best.tar")
        torch.save([trigger, model.cpu(), extra], save_path)
        print(f"[+] Saved defended tar: {save_path}", flush=True)

    result = {
        "model_key": spec.key,
        "model_name": spec.display_name,
        "poison_tar": _expand(spec.poison_tar),
        "poison_tar_sha256": file_sha256(_expand(spec.poison_tar)) if os.path.isfile(_expand(spec.poison_tar)) else "",
        "repo_dir": _expand(args.repo_dir),
        "platform_dir": _expand(args.platform_dir),
        "task_id": spec.task_id,
        "num_classes": spec.num_classes,
        "attack_type": "cnn_compiler",
        "mode": "real",
        "defense_algorithm": "INSPECT-Opt-pdf-loss",
        "defense_scope": f"{args.candidate_rule}_parameter_sets",
        "candidate_layer_rule": args.candidate_rule,
        "compile_backend": args.compile_backend,
        "cl_id": args.cl_id,
        "hardware_id": args.hardware_id,
        "q_bits": args.q_bits,
        "block_size": args.block_size,
        "m1_module": m1_module,
        "inspect_opt_scores": strict_scores.get("summary", {}),
        "before": before,
        "after": after,
        "before_defense": before,
        "after_defense": after,
        "inspect_opt": {
            "objective": "-lambda_a*sum|u|[Norm(Scomp)+Norm(Sstego)] + lambda_p*(logitMSE + lambda_w*weightMSE) + lambda_t*sum u^2(1-u^2)^2",
            "lambda_a": args.lambda_a,
            "lambda_p": args.lambda_p,
            "lambda_w": args.lambda_w,
            "lambda_t": args.lambda_t,
            "lambda_trigger_p": args.lambda_trigger_p,
            "tau": args.tau,
            "opt_steps": args.opt_steps,
            "lr": args.lr,
            "max_opt_positions": effective_max_opt_positions,
            "requested_max_opt_positions": args.max_opt_positions,
            "auto_budget": auto_budget_info,
            **{k: v for k, v in stat.items() if k != "u_trace"},
            "model_total_2d_params": model_total,
            "k_star_scope_percent": 100.0 * float(stat.get("selected", 0)) / max(1.0, float(stat.get("total", 1))),
            "k_star_model_percent": after.get("inspect_k_actual"),
        },
        "timing": {
            "pre_eval_time_sec": pre_eval_time,
            "defense_time_sec": defense_time,
            "post_eval_time_sec": post_eval_time,
            "total_time_sec": pre_eval_time + defense_time + post_eval_time,
        },
        "k_search": [{
            "k": after.get("inspect_k_actual"),
            "asr": after.get("asr"),
            "clean_accuracy": after.get("clean_accuracy"),
            "robust_accuracy": after.get("robust_accuracy"),
            "inspect_selected": after.get("inspect_selected"),
            "inspect_total": after.get("inspect_total"),
            "inspect_k_actual": after.get("inspect_k_actual"),
        }],
        "warnings": warnings,
        "notes": [
            "编译器后门主表 ASR 只取编译后模型在触发样本上的 target-label 命中率。",
            "PDF 防御阶段只执行一次：优化连续 u、按 tau 阈值离散化，并报告实际扰动比例 k*。",
            "防御算法为 PDF 版 INSPECT-Opt：先计算 Scomp/Sstego，再对连续变量 u 优化 Lattack + lambda_p Lpres + lambda_t Ltri，最后按 tau 阈值离散化为 Delta∈{-1,0,+1}；模型写回为整段 W'=DeQuant(q+Delta)，k* 只统计 Delta!=0 的比例。",
            "Scomp 真实计算 eager M1 与 compiled C1 中间特征差异；CNN 使用 4x4 spatial block。Sstego/PAI 通过 CNN tensor-base 候选扰动实测 Dacc/DlogitMSE。",
            "compiled 失败时不会用 eager 指标替代 compiled ASR。",
        ],
    }
    with open(os.path.join(out_dir, "defense_result.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(result), f, ensure_ascii=False, indent=2)
    pd.json_normalize(to_jsonable(result)).to_csv(os.path.join(out_dir, "defense_result.csv"), index=False)

    print("\n=== FINAL DEFENSE RESULT ===")
    print(json.dumps(to_jsonable(result), ensure_ascii=False, indent=2))
    print(f"\n[+] Done. Outputs are in: {out_dir}", flush=True)

    # Keep current model process clean for the next run.
    try:
        restore_module_params(model, backup)
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["c100_v19", "tiny_r34", "all"], default="all")
    p.add_argument("--platform-dir", default="~/platform_opt")
    p.add_argument("--repo-dir", default="~/DLCompilerAttack")
    p.add_argument("--out-dir", default="~/test_ssj/outputs/cnn_compiler_def_paper_full_seed2026")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--c100-poison", default=DEFAULT_SPECS["c100_v19"].poison_tar)
    p.add_argument("--tiny-poison", default=DEFAULT_SPECS["tiny_r34"].poison_tar)
    p.add_argument("--train-batch", type=int, default=0)
    p.add_argument("--test-batch", type=int, default=0)
    p.add_argument("--max-eval-batches", type=int, default=0, help="0 means full eval split")
    p.add_argument("--compile-backend", default="dlcl", choices=["dlcl", "original_dlcl", "torch_compile_inductor", "none", "skip"])
    p.add_argument("--cl-id", type=int, default=0)
    p.add_argument("--hardware-id", type=int, default=0)
    p.add_argument("--q-bits", type=int, default=8)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--pai-lsb-m", type=int, default=0)
    p.add_argument("--m1-module", default="")
    p.add_argument("--opt-search-max-batches", type=int, default=3)
    p.add_argument("--opt-steps", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=0.50)
    p.add_argument("--lambda-a", type=float, default=1.0)
    p.add_argument("--lambda-p", type=float, default=1.0)
    p.add_argument("--lambda-w", type=float, default=1e-4)
    p.add_argument("--lambda-t", type=float, default=0.01)
    p.add_argument("--lambda-trigger-p", type=float, default=0.0)
    p.add_argument("--trigger-preserve-batch-size", type=int, default=8)
    p.add_argument("--max-opt-positions", type=int, default=4096)
    p.add_argument("--auto-max-opt-positions", action="store_true")
    p.add_argument("--auto-scomp-param-ratio", type=float, default=0.12)
    p.add_argument("--auto-stego-tensor-ratio", type=float, default=0.024)
    p.add_argument("--scomp-min-share", type=float, default=0.30)
    p.add_argument("--non-m1-tensor-cap-ratio", type=float, default=0.05)
    p.add_argument("--pai-floor", type=float, default=1e-3)
    p.add_argument("--candidate-rule", choices=["union", "scomp", "sstego", "balanced_union"], default="union")
    p.add_argument("--target-label", type=int, default=0)
    p.add_argument("--override-target-label", action="store_true")
    p.add_argument("--no-prefer-saved-baseline", dest="prefer_saved_baseline", action="store_false")
    p.set_defaults(prefer_saved_baseline=True)
    p.add_argument("--save-defended-tar", action="store_true")
    args = p.parse_args()
    if args.max_eval_batches == 0:
        args.max_eval_batches = None
    return args


def main() -> None:
    args = parse_args()
    args.platform_dir = _expand(args.platform_dir)
    args.repo_dir = _expand(args.repo_dir)
    add_project_paths(args.platform_dir, args.repo_dir)
    os.makedirs(_expand(args.out_dir), exist_ok=True)

    specs = []
    if args.model in ["c100_v19", "all"]:
        s = copy.copy(DEFAULT_SPECS["c100_v19"])
        s.poison_tar = args.c100_poison
        specs.append(s)
    if args.model in ["tiny_r34", "all"]:
        s = copy.copy(DEFAULT_SPECS["tiny_r34"])
        s.poison_tar = args.tiny_poison
        specs.append(s)

    all_results = []
    for spec in specs:
        all_results.append(run_one(args, spec))

    summary_rows = []
    for r in all_results:
        before = r.get("before", {})
        after = r.get("after", {})
        summary_rows.append({
            "model": r.get("model_key"),
            "status": "success",
            "asr_before": before.get("asr"),
            "asr_after": after.get("asr"),
            "acc_before": before.get("clean_accuracy"),
            "acc_after": after.get("clean_accuracy"),
            "robust_before": before.get("robust_accuracy"),
            "robust_after": after.get("robust_accuracy"),
            "k_star_model_percent": (r.get("inspect_opt") or {}).get("k_star_model_percent"),
            "changed_params": (r.get("inspect_opt") or {}).get("selected"),
            "total_time_sec": (r.get("timing") or {}).get("total_time_sec"),
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(_expand(args.out_dir), "defense_summary.csv"), index=False)
    with open(os.path.join(_expand(args.out_dir), "defense_summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary_rows), f, ensure_ascii=False, indent=2)
    print(f"\n[+] All done. Summary: {os.path.join(_expand(args.out_dir), 'defense_summary.csv')}")


if __name__ == "__main__":
    main()
