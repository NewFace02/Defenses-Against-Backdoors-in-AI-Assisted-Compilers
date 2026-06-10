import torch
from .stego import protection_mask
from .metrics import logits_from_output


def k_to_probability(k_percent):
    k = float(k_percent)
    if k < 0:
        raise RuntimeError("k 不能为负数")
    if k > 100:
        raise RuntimeError("k 是百分比，不能超过 100")
    return k / 100.0


def _perturb_q_flat(q_flat, n, mode, device, bit_position=0):
    if n <= 0:
        return q_flat
    total = q_flat.numel()
    idx = torch.randperm(total, device=device)[:n]
    bit = int(bit_position)
    mask = torch.tensor(1 << bit, dtype=torch.int16, device=device)
    clear_mask = torch.tensor(~(1 << bit), dtype=torch.int16, device=device)
    if mode == "zero":
        q_flat[idx] = q_flat[idx] & clear_mask
    elif mode == "random":
        rand_bits = torch.randint(0, 2, (n,), device=device, dtype=torch.int16)
        q_flat[idx] = (q_flat[idx] & clear_mask) | (rand_bits << bit)
    else:
        direction = torch.where(torch.rand((n,), device=device) < 0.5, 1, -1).to(torch.int16)
        q_flat[idx] = q_flat[idx] + direction
    return q_flat


def tensor_quant_lsb_defense(w, q_bits=8, k_percent=0.5, block_size=32, mode="pm1", scale_mode="block", bit_position=0):
    if not isinstance(w, torch.Tensor) or not torch.is_floating_point(w) or w.numel() == 0:
        return w
    device = w.device
    dtype = w.dtype
    q_bits = int(q_bits)
    q_max = 2 ** (q_bits - 1) - 1
    q_min = -q_max
    p = k_to_probability(k_percent)
    if scale_mode == "tensor":
        flat = w.detach().reshape(-1).to(torch.float64)
        max_abs = flat.abs().max()
        if max_abs.item() == 0:
            return w.detach().clone()
        scale = max_abs / q_max
        q = torch.round(flat / scale).clamp(q_min, q_max).to(torch.int16)
        n = min(q.numel(), int(q.numel() * p))
        if n > 0:
            q = _perturb_q_flat(q, n, mode, device, bit_position=bit_position).clamp(q_min, q_max)
        rec = q.to(torch.float64) * scale
        return rec.reshape_as(w).to(device=device, dtype=dtype)
    flat = w.detach().reshape(-1).to(torch.float64)
    pad = (int(block_size) - flat.numel() % int(block_size)) % int(block_size)
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(-1, int(block_size))
    max_abs = blocks.abs().max(dim=1, keepdim=True)[0]
    scale = max_abs / q_max
    scale[scale == 0] = 1.0
    q = torch.round(blocks / scale).clamp(q_min, q_max).to(torch.int16)
    if p > 0:
        q_flat = q.reshape(-1)
        n = min(q_flat.numel(), int(round(q_flat.numel() * p)))
        if n > 0:
            q_flat = _perturb_q_flat(q_flat, n, mode, device, bit_position=bit_position)
            q = q_flat.reshape_as(q)
    q = q.clamp(q_min, q_max).to(torch.float64)
    rec = (q * scale).reshape(-1)
    if pad:
        rec = rec[:-pad]
    return rec.reshape_as(w).to(device=device, dtype=dtype)



def _candidate_param_items_from_model(model, target_names=None, include_bias=True, include_buffers=False):
    target = set(target_names) if target_names else None
    items = []
    for name, p in model.named_parameters():
        if target is not None and name not in target:
            continue
        if not torch.is_floating_point(p.data) or p.data.numel() == 0:
            continue
        if (not include_bias) and name.endswith("bias"):
            continue
        items.append((name, p))
    if target is not None and not items:
        raise RuntimeError("未找到需要防御的目标参数")
    return items


def _candidate_tensor_items_from_state(state, target_names=None, include_bias=True):
    target = set(target_names) if target_names else None
    items = []
    for name, v in state.items():
        if target is not None and name not in target:
            continue
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v) and v.numel() > 0:
            if (not include_bias) and str(name).endswith("bias"):
                continue
            items.append((name, v))
    if target is not None and not items:
        raise RuntimeError("未找到需要防御的目标参数")
    return items


def _quantize_tensor_to_q(w, q_bits=8, block_size=32, scale_mode="block"):
    device = w.device
    dtype = w.dtype
    q_bits = int(q_bits)
    q_max = 2 ** (q_bits - 1) - 1
    q_min = -q_max
    if scale_mode == "tensor":
        flat = w.detach().reshape(-1).to(torch.float64)
        max_abs = flat.abs().max()
        if max_abs.item() == 0:
            scale = torch.tensor(1.0, dtype=torch.float64, device=device)
        else:
            scale = max_abs / q_max
        q = torch.round(flat / scale).clamp(q_min, q_max).to(torch.int16)
        return q, scale, w.shape, 0, dtype, device, scale_mode
    flat = w.detach().reshape(-1).to(torch.float64)
    bs = int(block_size)
    pad = (bs - flat.numel() % bs) % bs
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(-1, bs)
    max_abs = blocks.abs().max(dim=1, keepdim=True)[0]
    scale = max_abs / q_max
    scale[scale == 0] = 1.0
    q = torch.round(blocks / scale).clamp(q_min, q_max).to(torch.int16).reshape(-1)
    return q, scale, w.shape, pad, dtype, device, scale_mode


def _dequantize_from_q(q, scale, shape, pad, dtype, device, scale_mode="block", block_size=32):
    if scale_mode == "tensor":
        rec = q.to(torch.float64) * scale
        return rec.reshape(shape).to(device=device, dtype=dtype)
    qv = q.to(torch.float64)
    rec = (qv.reshape(-1, int(block_size)) * scale).reshape(-1)
    if pad:
        rec = rec[:-pad]
    return rec.reshape(shape).to(device=device, dtype=dtype)


def _apply_global_quantized_perturb(records, q_bits=8, k_percent=0.5, block_size=32, mode="pm1", seed=None, bit_position=0):
    if not records:
        return
    q_max = 2 ** (int(q_bits) - 1) - 1
    q_min = -q_max
    total = sum(int(r[1].numel()) for r in records)
    n = min(total, max(0, int(round(total * k_to_probability(k_percent)))))
    if n > 0:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) if seed is not None else 2026)
        idx = torch.randperm(total, generator=g)[:n].tolist()
        if mode == "zero":
            dirs = [0] * n
        elif mode == "random":
            dirs = torch.randint(0, 2, (n,), generator=g, dtype=torch.int16).tolist()
        else:
            dirs = torch.where(torch.rand(n, generator=g) < 0.5, torch.ones(n, dtype=torch.int16), -torch.ones(n, dtype=torch.int16)).tolist()
        offsets = []
        cur = 0
        for rec in records:
            q = rec[1]
            nxt = cur + q.numel()
            offsets.append((cur, nxt, rec))
            cur = nxt
        pos = 0
        for gi, dv in sorted(zip(idx, dirs), key=lambda x: x[0]):
            while pos < len(offsets) and gi >= offsets[pos][1]:
                pos += 1
            if pos >= len(offsets):
                break
            lo, _, rec = offsets[pos]
            q = rec[1]
            local = gi - lo
            bit = int(bit_position)
            clear_mask = torch.tensor(~(1 << bit), dtype=torch.int16, device=q.device)
            if mode == "zero":
                q[local] = q[local] & clear_mask
            elif mode == "random":
                q[local] = (q[local] & clear_mask) | (int(dv) << bit)
            else:
                q[local] = q[local] + int(dv)
    for rec in records:
        rec[1].clamp_(q_min, q_max)


def apply_lsb_quant_defense_model(model, q_bits=8, k_percent=0.5, block_size=32, mode="pm1", target_names=None, seed=None, scale_mode="block", global_sampling=False, include_bias=True, bit_position=0):
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    items = _candidate_param_items_from_model(model, target_names=target_names, include_bias=include_bias)
    if global_sampling:
        records = []
        for name, p in items:
            q, scale, shape, pad, dtype, device, sm = _quantize_tensor_to_q(p.data, q_bits=q_bits, block_size=block_size, scale_mode=scale_mode)
            records.append([p, q, scale, shape, pad, dtype, device, sm])
        _apply_global_quantized_perturb(records, q_bits=q_bits, k_percent=k_percent, block_size=block_size, mode=mode, seed=seed, bit_position=bit_position)
        with torch.no_grad():
            for p, q, scale, shape, pad, dtype, device, sm in records:
                p.data.copy_(_dequantize_from_q(q, scale, shape, pad, dtype, device, scale_mode=sm, block_size=block_size))
        return model
    with torch.no_grad():
        for name, p in items:
            p.data.copy_(tensor_quant_lsb_defense(p.data, q_bits=q_bits, k_percent=k_percent, block_size=block_size, mode=mode, scale_mode=scale_mode, bit_position=bit_position))
    return model

def apply_lsb_quant_defense_state(state, q_bits=8, k_percent=0.5, block_size=32, mode="pm1", target_names=None, seed=None, scale_mode="block", global_sampling=False, include_bias=True, bit_position=0):
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    target = set(target_names) if target_names else None
    out = {}
    for name, v in state.items():
        out[name] = v.clone() if isinstance(v, torch.Tensor) else v
    items = _candidate_tensor_items_from_state(out, target_names=target, include_bias=include_bias)
    if global_sampling:
        records = []
        for name, v in items:
            q, scale, shape, pad, dtype, device, sm = _quantize_tensor_to_q(v, q_bits=q_bits, block_size=block_size, scale_mode=scale_mode)
            records.append([name, q, scale, shape, pad, dtype, device, sm])
        _apply_global_quantized_perturb(records, q_bits=q_bits, k_percent=k_percent, block_size=block_size, mode=mode, seed=seed, bit_position=bit_position)
        for name, q, scale, shape, pad, dtype, device, sm in records:
            out[name] = _dequantize_from_q(q, scale, shape, pad, dtype, device, scale_mode=sm, block_size=block_size)
        return out
    for name, v in items:
        out[name] = tensor_quant_lsb_defense(v, q_bits=q_bits, k_percent=k_percent, block_size=block_size, mode=mode, scale_mode=scale_mode, bit_position=bit_position)
    return out

def backup_module_params(model, target_names=None):
    target = set(target_names) if target_names else None
    backup = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if target is not None and name not in target:
                continue
            if torch.is_floating_point(p.data) and p.data.ndim >= 1:
                backup[name] = p.detach().cpu().clone()
        for name, b in model.named_buffers():
            if target is not None and name not in target:
                continue
            if isinstance(b, torch.Tensor) and torch.is_floating_point(b) and b.ndim >= 1:
                backup[name] = b.detach().cpu().clone()
    if target is not None and not backup:
        raise RuntimeError("未找到需要备份的目标参数")
    return backup


def restore_module_params(model, backup):
    named = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    with torch.no_grad():
        for name, value in backup.items():
            if name in named:
                named[name].data.copy_(value.to(named[name].device, dtype=named[name].dtype))
            elif name in buffers:
                buffers[name].data.copy_(value.to(buffers[name].device, dtype=buffers[name].dtype))
            else:
                raise RuntimeError(f"模型缺少备份参数：{name}")
    return model


def backup_state_params(state, target_names=None):
    target = set(target_names) if target_names else None
    backup = {}
    for name, v in state.items():
        if target is not None and name not in target:
            continue
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v) and v.ndim >= 1:
            backup[name] = v.detach().cpu().clone()
    if target is not None and not backup:
        raise RuntimeError("未找到需要备份的目标参数")
    return backup


def restore_state_params(state, backup):
    out = dict(state)
    for name, value in backup.items():
        if name not in out:
            raise RuntimeError(f"state_dict 缺少备份参数：{name}")
        out[name] = value.clone()
    return out


def _norm01(x):
    x = x.detach().float()
    if x.numel() == 0:
        return x
    mn = x.min()
    mx = x.max()
    if float((mx - mn).abs().item()) < 1e-12:
        # For attack relevance fusion, a branch that contributes all-zero scores
        # must remain zero.  A constant positive branch is still a valid uniform
        # relevance signal and is normalized to one.
        if float(mx.abs().item()) < 1e-12:
            return torch.zeros_like(x)
        return torch.ones_like(x)
    return (x - mn) / (mx - mn)


def _position_prior(numel, payload_bits=None, device="cpu"):
    if payload_bits is None or payload_bits <= 0 or payload_bits >= numel:
        return torch.ones(numel, dtype=torch.float32, device=device)
    out = torch.zeros(numel, dtype=torch.float32, device=device)
    out[: int(payload_bits)] = 1.0
    return out


def _q_records_for_items(items, q_bits=8, block_size=32, scale_mode="block"):
    records = []
    for name, tensor_ref in items:
        data = tensor_ref.data if hasattr(tensor_ref, "data") else tensor_ref
        q, scale, shape, pad, dtype, device, sm = _quantize_tensor_to_q(data, q_bits=q_bits, block_size=block_size, scale_mode=scale_mode)
        records.append([name, tensor_ref, q, scale, shape, pad, dtype, device, sm])
    return records


def _record_param_numel(record):
    tensor_ref = record[1]
    data = tensor_ref.data if hasattr(tensor_ref, "data") else tensor_ref
    return int(data.numel())


def _records_param_total(records):
    return sum(_record_param_numel(r) for r in records)


def _apply_inspect_records(records, selected, q_bits=8, block_size=32, mode="pm1"):
    """Apply q' = clip(q + Delta), then dequantize every defended tensor.

    k* counts only Delta != 0 positions, but the defended weights are
    W' = DeQuant(q') for the whole selected defense scope.  That means even
    k*=0 still performs quantization/dequantization, which is part of the
    defense effect in the experiment.
    """
    q_max = 2 ** (int(q_bits) - 1) - 1
    q_min = -q_max
    for ri, pairs in selected.items():
        name, tensor_ref, q, scale, shape, pad, dtype, device, sm = records[ri]
        q_flat = q.reshape(-1)
        for idx, dv in pairs:
            if idx < 0 or idx >= q_flat.numel():
                continue
            if mode == "zero":
                q_flat[idx] = q_flat[idx] & torch.tensor(-2, dtype=torch.int16, device=q_flat.device)
            elif mode == "random":
                q_flat[idx] = (q_flat[idx] & torch.tensor(-2, dtype=torch.int16, device=q_flat.device)) | int(dv)
            else:
                q_flat[idx] = q_flat[idx] + int(dv)
        q_flat.clamp_(q_min, q_max)
    with torch.no_grad():
        for ri, (name, tensor_ref, q, scale, shape, pad, dtype, device, sm) in enumerate(records):
            rec = _dequantize_from_q(q, scale, shape, pad, dtype, device, scale_mode=sm, block_size=block_size)
            if hasattr(tensor_ref, "data"):
                tensor_ref.data.copy_(rec.to(device=tensor_ref.data.device, dtype=tensor_ref.data.dtype))
            else:
                tensor_ref.copy_(rec.to(device=tensor_ref.device, dtype=tensor_ref.dtype))


def _iter_calibration_batches(dataloader, device, max_batches=2):
    if dataloader is None:
        return []
    out = []
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= int(max_batches):
            break
        moved = {}
        for k, v in batch.items():
            moved[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        out.append(moved)
    return out


def _forward_model_for_phi(model, batch):
    if "input" in batch:
        dtype = getattr(model, "fp", torch.float32)
        return model(batch["input"].to(dtype))
    if "input_ids" in batch:
        try:
            return model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
        except TypeError:
            if batch.get("attention_mask") is None:
                return model(batch["input_ids"])
            return model(batch["input_ids"], batch.get("attention_mask"))
    raise RuntimeError("校准 batch 中既没有 input，也没有 input_ids")


def _forward_functional_for_phi(model, params, batch):
    try:
        from torch.func import functional_call
    except Exception:
        from torch.nn.utils.stateless import functional_call
    if "input" in batch:
        dtype = getattr(model, "fp", torch.float32)
        return functional_call(model, params, (batch["input"].to(dtype),))
    if "input_ids" in batch:
        import inspect
        ids = batch["input_ids"].long()
        mask = batch.get("attention_mask")
        if mask is not None:
            mask = mask.long()
        kwargs = {"input_ids": ids}
        if mask is not None:
            kwargs["attention_mask"] = mask
        try:
            sig = inspect.signature(model.forward)
            accepts_input_ids = "input_ids" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        except Exception:
            accepts_input_ids = True
        if accepts_input_ids:
            try:
                return functional_call(model, params, (), kwargs)
            except TypeError:
                pass
        # GPT-2 compiler best.tar wraps the HF model in a MyModel whose forward
        # does not accept input_ids=... kwargs.  Use the attack model's
        # positional signature only for that case.
        try:
            return functional_call(model, params, (ids, mask)) if mask is not None else functional_call(model, params, (ids,))
        except TypeError:
            try:
                return functional_call(model, params, ([ids, mask] if mask is not None else [ids],))
            except TypeError:
                return functional_call(model, params, (), kwargs)
    raise RuntimeError("校准 batch 中既没有 input，也没有 input_ids")


def _module_by_name(model, name):
    cur = model
    if not name:
        return None
    for part in str(name).split("."):
        if part == "":
            continue
        try:
            if part.isdigit() and isinstance(cur, (torch.nn.Sequential, torch.nn.ModuleList)):
                cur = cur[int(part)]
            else:
                cur = getattr(cur, part)
        except Exception:
            return None
    return cur


def _capture_forward_phi(model, batch, module_name=None, params=None):
    holder = {}
    hook = None
    mod = _module_by_name(model, module_name)
    if mod is not None:
        def _hook(_m, _inp, out):
            holder["hidden"] = out[0] if isinstance(out, tuple) else out
        hook = mod.register_forward_hook(_hook)
    try:
        out = _forward_functional_for_phi(model, params, batch) if params is not None else _forward_model_for_phi(model, batch)
    finally:
        if hook is not None:
            hook.remove()
    return _phi_logits(out), holder.get("hidden")


def _phi_logits(output):
    logits = logits_from_output(output)
    if logits.dim() > 2:
        if logits.shape[0] > 0 and logits.shape[-1] > 1:
            logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])[:, -1, :]
        else:
            logits = logits.reshape(logits.shape[0], -1)
    return logits.float()


def _score_vector(base, n):
    if isinstance(base, torch.Tensor):
        t = base.detach().float().reshape(-1).cpu()
        if t.numel() == n:
            return t
        if t.numel() == 1:
            return torch.full((n,), float(t.item()), dtype=torch.float32)
    if isinstance(base, (list, tuple)):
        t = torch.tensor(base, dtype=torch.float32).reshape(-1)
        if t.numel() == n:
            return t
        if t.numel() == 1:
            return torch.full((n,), float(t.item()), dtype=torch.float32)
    return torch.full((n,), float(base), dtype=torch.float32)


def _strict_attack_components_for_record(record, q_bits=8, block_size=32, attack_type=None, target_names=None, layer_hint=None, payload_bits=None, strict_scores=None):
    name, tensor_ref, q, scale, shape, pad, dtype, device, sm = record
    data = tensor_ref.data if hasattr(tensor_ref, "data") else tensor_ref
    n = int(data.numel())
    if n == 0:
        return torch.empty(0), torch.empty(0), torch.empty(0, dtype=torch.bool)
    strict = strict_scores or {}
    comp_base = strict.get("scomp", {}).get(name)
    stego_base = strict.get("sstego", {}).get(name)
    if comp_base is None:
        comp_base = 0.0
    if stego_base is None:
        stego_base = 0.0
    scomp = _score_vector(comp_base, n)
    sstego = _score_vector(stego_base, n)
    writable = torch.ones(n, dtype=torch.bool)
    if sm == "block":
        try:
            writable = protection_mask(data.detach().cpu(), block_size=block_size).reshape(-1).bool()
        except Exception:
            writable = torch.ones(n, dtype=torch.bool)
    if attack_type == "cnn_stego":
        pos = _position_prior(n, payload_bits=payload_bits, device="cpu")
        sstego = sstego * pos
        writable = writable & pos.bool()
    if str(name).lower().endswith("bias"):
        writable[:] = False
    scomp[~writable] = 0.0
    sstego[~writable] = 0.0
    return scomp, sstego, writable


def _build_continuous_u_plan(records, k_percent=0.5, q_bits=8, block_size=32, attack_type=None, target_names=None, layer_hint=None, payload_bits=None, strict_scores=None, max_opt_positions=None):
    candidates = []
    total = _records_param_total(records)
    for ri, rec in enumerate(records):
        scomp, sstego, writable = _strict_attack_components_for_record(rec, q_bits=q_bits, block_size=block_size, attack_type=attack_type, target_names=target_names, layer_hint=layer_hint, payload_bits=payload_bits, strict_scores=strict_scores)
        if scomp.numel() == 0:
            continue
        idx = torch.nonzero(writable & torch.isfinite(scomp) & torch.isfinite(sstego) & ((scomp + sstego) > 0), as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        candidates.append((ri, idx, scomp[idx], sstego[idx]))
    if not candidates:
        raise RuntimeError("PDF 防御没有可优化位置：真实 Scomp/Sstego 覆盖为空，不能输出 k*=0 的假防御结果。请检查 compiled/M1 是否可用，或隐写 target/layer 是否被 PAI 覆盖。")

    all_comp = torch.cat([x[2] for x in candidates])
    all_stego = torch.cat([x[3] for x in candidates])
    all_scores = _norm01(all_comp) + _norm01(all_stego)
    cap = None
    try:
        cap = int(max_opt_positions) if max_opt_positions not in [None, "", 0, "0", "all"] else None
    except Exception:
        cap = None
    if cap is not None:
        cap = max(1, min(all_scores.numel(), cap))
    else:
        cap = all_scores.numel()
    if cap < all_scores.numel():
        _, keep_global = torch.topk(all_scores, k=cap)
        keep_mask = torch.zeros_like(all_scores, dtype=torch.bool)
        keep_mask[keep_global] = True
    else:
        keep_mask = torch.ones_like(all_scores, dtype=torch.bool)

    plan = []
    cursor = 0
    for ri, idx, _comp, _stego in candidates:
        m = keep_mask[cursor:cursor + idx.numel()]
        norm_scores = all_scores[cursor:cursor + idx.numel()]
        cursor += idx.numel()
        if not m.any():
            continue
        name, tensor_ref, q, scale, shape, pad, dtype, device, sm = records[ri]
        plan.append({
            "record_index": ri,
            "name": name,
            "idx": idx[m].long(),
            "sattack": norm_scores[m].float(),
        })
    return plan, total


def _relaxed_tensor_from_record(record, local_idx, u, q_bits=8, block_size=32):
    name, tensor_ref, q, scale, shape, pad, dtype, device, sm = record
    q_max = 2 ** (int(q_bits) - 1) - 1
    q_min = -q_max
    q_relaxed = q.detach().to(device=device, dtype=torch.float64).reshape(-1).clone()
    idx = local_idx.to(device=device, dtype=torch.long)
    q_relaxed[idx] = q_relaxed[idx] + u.to(device=device, dtype=torch.float64)
    q_relaxed = q_relaxed.clamp(q_min, q_max)
    return _dequantize_from_q(q_relaxed, scale, shape, pad, dtype, device, scale_mode=sm, block_size=block_size)


def _optimized_deltas_from_pdf_loss(model, records, plan, dataloader, device="cpu", q_bits=8, block_size=32, k_percent=0.5, seed=None, max_batches=2, steps=30, lr=0.05, tau=0.5, lambda_a=1.0, lambda_p=1.0, lambda_t=0.01, lambda_w=1e-4, preserve_module_name=None, opt_tol=None, opt_patience=None, opt_min_steps=0, extra_preserve_batches=None, lambda_trigger_p=0.0):
    if not plan:
        return {}, {"selected": 0, "total": _records_param_total(records), "optimized": 0, "loss_final": None, "u_trace": [], "u_trace_total_candidates": 0, "u_trace_points_per_step": 0, "u_trace_sampled": False}
    batches = _iter_calibration_batches(dataloader, device, max_batches=max_batches)
    if not batches:
        raise RuntimeError("INSPECT-Opt 严格优化需要上传数据集作为 Dc，用于计算 Lpres")
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    model = model.to(device).eval()
    base_phi = []
    with torch.no_grad():
        for batch in batches:
            logits, hidden = _capture_forward_phi(model, batch, module_name=preserve_module_name)
            base_phi.append((logits.detach(), hidden.detach() if isinstance(hidden, torch.Tensor) else None))

    extra_batches = []
    extra_phi = []
    if extra_preserve_batches is not None and float(lambda_trigger_p) > 0:
        for batch in extra_preserve_batches:
            moved = {}
            for k, v in batch.items():
                moved[k] = v.to(device) if isinstance(v, torch.Tensor) else v
            extra_batches.append(moved)
        with torch.no_grad():
            for batch in extra_batches:
                logits, hidden = _capture_forward_phi(model, batch, module_name=preserve_module_name)
                extra_phi.append((logits.detach(), hidden.detach() if isinstance(hidden, torch.Tensor) else None))

    u_chunks = []
    attack_chunks = []
    for item in plan:
        u = (torch.rand(int(item["idx"].numel()), device=device, dtype=torch.float32) * 2.0 - 1.0) * 1e-3
        u.requires_grad_(True)
        u_chunks.append(u)
        attack_chunks.append(item["sattack"].to(device=device, dtype=torch.float32))
    opt = torch.optim.Adam(u_chunks, lr=float(lr))

    # 仅做可视化采样：记录连续扰动变量 u 在优化过程中的轨迹，不参与损失和参数写回。
    trace_total_candidates = sum(int(u.numel()) for u in u_chunks)
    trace_cap = 4096 if trace_total_candidates <= 4096 else 2000
    if trace_total_candidates <= trace_cap:
        trace_global_idx = torch.arange(trace_total_candidates, dtype=torch.long)
    else:
        trace_global_idx = torch.linspace(0, trace_total_candidates - 1, steps=trace_cap).round().long().unique()
    trace_maps = []
    cursor = 0
    for ci, u in enumerate(u_chunks):
        n = int(u.numel())
        mask = (trace_global_idx >= cursor) & (trace_global_idx < cursor + n)
        local = (trace_global_idx[mask] - cursor).long().cpu()
        if local.numel() > 0:
            trace_maps.append((ci, local))
        cursor += n
    u_trace = []

    def _append_u_trace(step_no):
        vals = []
        for ci, local in trace_maps:
            cur = torch.tanh(u_chunks[ci].detach()).cpu()
            vals.extend(float(x) for x in cur[local].tolist())
        u_trace.append({"step": int(step_no), "values": vals})

    _append_u_trace(0)
    named_params = dict(model.named_parameters())
    named_params.update(dict(model.named_buffers()))
    last = {}
    best_loss = None
    best_u = None
    stale = 0
    steps_run = 0
    tol_val = None if opt_tol in [None, ""] else float(opt_tol)
    patience_val = None if opt_patience in [None, "", 0, "0"] else int(opt_patience)
    min_steps_val = max(0, int(opt_min_steps or 0))
    for step_idx in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        relaxed_params = dict(named_params)
        lweight = torch.zeros((), device=device, dtype=torch.float32)
        for item, u in zip(plan, u_chunks):
            rec = records[int(item["record_index"])]
            relaxed = _relaxed_tensor_from_record(rec, item["idx"], torch.tanh(u), q_bits=q_bits, block_size=block_size)
            relaxed_params[item["name"]] = relaxed
            base = named_params[item["name"]].detach()
            lweight = lweight + torch.sum((base.float() - relaxed.float()) ** 2)
        lphi = torch.zeros((), device=device, dtype=torch.float32)
        for batch, ref_pair in zip(batches, base_phi):
            ref_logits, ref_hidden = ref_pair
            cur, cur_hidden = _capture_forward_phi(model, batch, module_name=preserve_module_name, params=relaxed_params)
            ref_logits = ref_logits.to(cur.device)
            if cur.shape == ref_logits.shape:
                diff = (ref_logits - cur).reshape(cur.shape[0], -1)
                lphi = lphi + torch.mean(torch.sum(diff ** 2, dim=1))
            if isinstance(ref_hidden, torch.Tensor) and isinstance(cur_hidden, torch.Tensor) and ref_hidden.shape == cur_hidden.shape:
                hdiff = (ref_hidden.to(cur_hidden.device).float() - cur_hidden.float()).reshape(cur_hidden.shape[0], -1)
                lphi = lphi + torch.mean(torch.sum(hdiff ** 2, dim=1))
        lpres = lphi / max(1, len(batches)) + float(lambda_w) * lweight
        ltrigger_pres = torch.zeros((), device=device, dtype=torch.float32)
        if extra_phi:
            for batch, ref_pair in zip(extra_batches, extra_phi):
                ref_logits, ref_hidden = ref_pair
                cur, cur_hidden = _capture_forward_phi(model, batch, module_name=preserve_module_name, params=relaxed_params)

                if isinstance(ref_logits, torch.Tensor) and isinstance(cur, torch.Tensor) and cur.shape == ref_logits.shape:
                    diff = (ref_logits.to(cur.device).float() - cur.float()).reshape(cur.shape[0], -1)
                    ltrigger_pres = ltrigger_pres + torch.mean(torch.sum(diff ** 2, dim=1))

                if isinstance(ref_hidden, torch.Tensor) and isinstance(cur_hidden, torch.Tensor) and ref_hidden.shape == cur_hidden.shape:
                    hdiff = (ref_hidden.to(cur_hidden.device).float() - cur_hidden.float()).reshape(cur_hidden.shape[0], -1)
                    ltrigger_pres = ltrigger_pres + torch.mean(torch.sum(hdiff ** 2, dim=1))

            ltrigger_pres = ltrigger_pres / max(1, len(extra_phi))
        lattack = torch.zeros((), device=device, dtype=torch.float32)
        ltri = torch.zeros((), device=device, dtype=torch.float32)
        for u_raw, sattack in zip(u_chunks, attack_chunks):
            u = torch.tanh(u_raw)
            lattack = lattack - float(lambda_a) * torch.sum(torch.abs(u) * sattack)
            ltri = ltri + torch.sum((u ** 2) * ((1.0 - u ** 2) ** 2))
        loss = lattack + float(lambda_p) * lpres + float(lambda_trigger_p) * ltrigger_pres + float(lambda_t) * ltri
        loss.backward()
        opt.step()
        steps_run = step_idx + 1
        loss_value = float(loss.detach().cpu().item())
        improved = best_loss is None or loss_value < best_loss - (abs(best_loss) * tol_val + tol_val if tol_val is not None else 0.0)
        _append_u_trace(steps_run)
        if improved:
            best_loss = loss_value
            best_u = [torch.tanh(u_raw.detach()).cpu().clone() for u_raw in u_chunks]
            stale = 0
        else:
            stale += 1
        last = {
            "loss_final": loss_value,
            "loss_attack": float(lattack.detach().cpu().item()),
            "loss_pres": float(lpres.detach().cpu().item()),
            "loss_tri": float(ltri.detach().cpu().item()),
            "loss_trigger_pres": float(ltrigger_pres.detach().cpu().item()) if isinstance(ltrigger_pres, torch.Tensor) else 0.0,
        }
        if patience_val is not None and steps_run >= min_steps_val and stale >= patience_val:
            break

    total = _records_param_total(records)
    tau_val = 0.5 if tau in [None, ""] else float(tau)
    selected = {}
    picked = 0
    final_u_chunks = best_u if best_u is not None else [torch.tanh(u_raw.detach()).cpu() for u_raw in u_chunks]
    for item, u in zip(plan, final_u_chunks):
        idx = item["idx"].cpu()
        keep = torch.abs(u) > tau_val
        if keep.any():
            dirs = torch.sign(u[keep]).to(torch.int16)
            pairs = [(int(i), int(d)) for i, d in zip(idx[keep].tolist(), dirs.tolist()) if int(d) != 0]
            if pairs:
                selected.setdefault(int(item["record_index"]), []).extend(pairs)
                picked += len(pairs)
    last.update({"selected": picked, "total": total, "optimized": sum(int(x["idx"].numel()) for x in plan), "tau": tau_val, "steps": int(steps), "steps_run": int(steps_run), "best_loss": best_loss, "max_batches": int(max_batches or 0), "u_trace": u_trace, "u_trace_total_candidates": int(trace_total_candidates), "u_trace_points_per_step": int(len(u_trace[0].get("values", [])) if u_trace else 0), "u_trace_sampled": bool(trace_total_candidates > len(u_trace[0].get("values", [])) if u_trace else False)})
    return selected, last


def apply_inspect_opt_defense_model(model, q_bits=8, k_percent=0.5, block_size=32, mode="pm1", target_names=None, seed=None, scale_mode="block", include_bias=False, attack_type=None, layer_hint=None, payload_bits=None, strict_scores=None, dataloader=None, device="cpu", opt_max_batches=2, opt_steps=30, opt_lr=0.05, opt_tau=0.50, lambda_a=1.0, lambda_p=1.0, lambda_t=0.01, lambda_w=1e-4, max_opt_positions=None, preserve_module_name=None, opt_tol=None, opt_patience=None, opt_min_steps=0):
    """INSPECT-Opt defense following the PDF loss.

    Build continuous u in [-1, 1], optimize
    L = -lambda_a * sum |u_i| Sattack(i) + lambda_p * Lpres
        + lambda_t * sum u_i^2(1-u_i^2)^2,
    then threshold u into Delta in {-1, 0, +1} and dequantize q + Delta.
    """
    model = model.to(device)
    items = _candidate_param_items_from_model(model, target_names=target_names, include_bias=include_bias, include_buffers=False)
    records = _q_records_for_items(items, q_bits=q_bits, block_size=block_size, scale_mode=scale_mode)
    plan, total = _build_continuous_u_plan(records, k_percent=k_percent, q_bits=q_bits, block_size=block_size, attack_type=attack_type, target_names=target_names, layer_hint=layer_hint, payload_bits=payload_bits, strict_scores=strict_scores, max_opt_positions=max_opt_positions)
    selected, stat = _optimized_deltas_from_pdf_loss(
        model,
        records,
        plan,
        dataloader=dataloader,
        device=device,
        q_bits=q_bits,
        block_size=block_size,
        k_percent=k_percent,
        seed=seed,
        max_batches=opt_max_batches,
        steps=opt_steps,
        lr=opt_lr,
        tau=opt_tau,
        lambda_a=lambda_a,
        lambda_p=lambda_p,
        lambda_t=lambda_t,
        lambda_w=lambda_w,
        preserve_module_name=preserve_module_name,
        opt_tol=opt_tol,
        opt_patience=opt_patience,
        opt_min_steps=opt_min_steps,
    )
    if attack_type == "llm_compiler":
        selected = {
            int(ri): [(int(idx), -1) for idx, _dv in pairs]
            for ri, pairs in selected.items()
        }
        stat["llm_compiler_delta_sign"] = -1
    _apply_inspect_records(records, selected, q_bits=q_bits, block_size=block_size, mode="pm1")
    stat["total"] = total
    stat["algorithm"] = "pdf_continuous_u_loss"
    model._inspect_opt_last = stat
    return model


def apply_inspect_opt_defense_state(state, q_bits=8, k_percent=0.5, block_size=32, mode="pm1", target_names=None, seed=None, scale_mode="block", include_bias=False, attack_type=None, layer_hint=None, payload_bits=None, strict_scores=None):
    raise RuntimeError("PDF 版 INSPECT-Opt 需要 nn.Module 和数据集来计算 Lpres=||Phi_W-Phi_Wf(u)||；不能对裸 state_dict 执行防御。请先将 state_dict 构造成可 forward 的模型。")
