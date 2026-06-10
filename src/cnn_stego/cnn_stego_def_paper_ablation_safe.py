import os
import sys
import csv
import json
import time
import random
import argparse
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


TRIGGER_SEQ = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0]


SPECS = {
    "c100_v19": {
        "task_id": 3,
        "model_name": "C100-V19 / VGG19-CIFAR100",
        "poison_path": "~/test_ssj/outputs/cnn_stego_attack_paper_full_seed2026/c100_v19/poison_cnn_stego_best.pth",
        "out_subdir": "c100_v19",
        "test_batch": 100,
    },
    "tiny_r34": {
        "task_id": 4,
        "model_name": "Tiny-R34 / ResNet34-TinyImageNet",
        "poison_path": "~/test_ssj/outputs/cnn_stego_attack_paper_full_seed2026/tiny_r34/poison_cnn_stego_best.pth",
        "out_subdir": "tiny_r34",
        "test_batch": 100,
    },
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bits_to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def repeat_decode(bits, n):
    out = []
    usable = (len(bits) // n) * n
    for i in range(0, usable, n):
        chunk = bits[i:i+n]
        out.append(1 if sum(chunk) > (n // 2) else 0)
    return out


def quantize_blocks(w, bits=8, block_size=32, use_writable_mask=True):
    orig_shape = tuple(w.shape)
    flat = w.detach().reshape(-1).to(torch.float32)

    pad_len = (block_size - (flat.numel() % block_size)) % block_size
    if pad_len:
        flat = F.pad(flat, (0, pad_len))

    blocks = flat.view(-1, block_size)
    qmax = 2 ** (bits - 1) - 1

    max_abs = blocks.abs().max(dim=1, keepdim=True).values
    scale = max_abs / qmax
    scale[scale == 0] = 1.0

    q = torch.round(blocks / scale).clamp(-qmax, qmax).to(torch.int16)

    # writable mask: protect scale-determining element in each block and padded tail.
    writable = torch.ones_like(q, dtype=torch.bool)
    max_idx = blocks.abs().argmax(dim=1)
    if bool(use_writable_mask):
        writable[torch.arange(q.shape[0]), max_idx] = False

    if pad_len:
        writable.reshape(-1)[-pad_len:] = False

    return q, scale, orig_shape, pad_len, writable.reshape(-1), qmax


def dequantize_blocks(q, scale, orig_shape, pad_len):
    x = q.to(torch.float32) * scale.to(q.device)
    flat = x.reshape(-1)
    if pad_len:
        flat = flat[:-pad_len]
    return flat.reshape(orig_shape)


def get_param(model, name):
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise KeyError(name)


def candidate_params(model, min_numel=1024):
    rows = []
    for name, p in model.named_parameters():
        if not name.endswith("weight"):
            continue
        if p.dim() < 2:
            continue
        low = name.lower()
        if any(x in low for x in ["bn", "norm", "ln", "embedding", "embed"]):
            continue
        if p.numel() < min_numel:
            continue
        rows.append({
            "name": name,
            "shape": tuple(p.shape),
            "numel": int(p.numel()),
        })
    return rows


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break

        if isinstance(batch, dict):
            x = batch["input"].to(device)
            y = batch["label"].to(device)
        else:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

        if hasattr(model, "fp"):
            x = x.to(model.fp)

        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]

        loss = F.cross_entropy(out, y, reduction="sum")
        pred = out.argmax(dim=1)

        total += int(y.numel())
        correct += int((pred == y).sum().item())
        loss_sum += float(loss.item())

    return {
        "acc": 100.0 * correct / max(1, total),
        "loss": loss_sum / max(1, total),
        "num_samples": total,
    }


@torch.no_grad()
def collect_logits(model, loader, device, max_batches=5):
    model.eval()
    outs = []
    labels = []
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        if isinstance(batch, dict):
            x = batch["input"].to(device)
            y = batch["label"].to(device)
        else:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

        if hasattr(model, "fp"):
            x = x.to(model.fp)

        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        outs.append(out.detach().float().cpu())
        labels.append(y.detach().cpu())

    return torch.cat(outs, dim=0), torch.cat(labels, dim=0)


def extract_payload(model, param_name, q_bits, block_size, search_limit=300000):
    p = get_param(model, param_name).detach()
    q, scale, orig_shape, pad_len, writable, qmax = quantize_blocks(p, bits=q_bits, block_size=block_size)
    q_flat = q.reshape(-1).to(torch.int16)
    writable_idx = torch.where(writable.cpu())[0]

    bits = []
    for idx in writable_idx.tolist():
        bits.append(int((q_flat[int(idx)] & 1).item()))
        if len(bits) >= search_limit:
            break

    trig_len = len(TRIGGER_SEQ)
    start = -1
    max_start = max(0, len(bits) - trig_len + 1)
    for s in range(max_start):
        if bits[s:s + trig_len] == TRIGGER_SEQ:
            start = s
            break

    if start < 0:
        return {"ok": False, "reason": "trigger_not_found", "payload_bits": []}

    pos = start + trig_len
    if pos + 4 + 16 > len(bits):
        return {"ok": False, "reason": "header_incomplete", "payload_bits": []}

    n = bits_to_int(bits[pos:pos+4])
    pos += 4

    payload_len = bits_to_int(bits[pos:pos+16])
    pos += 16

    if n <= 0:
        return {"ok": False, "reason": "invalid_n", "payload_bits": []}

    enc_len = payload_len * n
    if pos + enc_len > len(bits):
        return {"ok": False, "reason": "payload_incomplete", "payload_bits": []}

    enc = bits[pos:pos+enc_len]
    payload_bits = repeat_decode(enc, n)[:payload_len]

    return {
        "ok": True,
        "reason": "success",
        "payload_bits": payload_bits,
        "payload_len": int(payload_len),
        "n": int(n),
        "start_pos": int(start),
    }


def compare_extract(extract_info, payload_bits_ref):
    if not extract_info.get("ok", False):
        return {
            "extract_exact_match": False,
            "bit_errors": None,
            "extract_reason": extract_info.get("reason"),
            "extract_ok": False,
            "payload_len": None,
        }

    ext_bits = [int(b) for b in extract_info.get("payload_bits", [])]
    ref_bits = [int(b) for b in payload_bits_ref] if payload_bits_ref is not None else None

    if ref_bits is None:
        return {
            "extract_exact_match": None,
            "bit_errors": None,
            "extract_reason": extract_info.get("reason"),
            "extract_ok": True,
            "payload_len": len(ext_bits),
        }

    bit_errors = abs(len(ref_bits) - len(ext_bits))
    bit_errors += sum(int(a != b) for a, b in zip(ref_bits, ext_bits))

    return {
        "extract_exact_match": (bit_errors == 0 and len(ref_bits) == len(ext_bits)),
        "bit_errors": int(bit_errors),
        "extract_reason": extract_info.get("reason"),
        "extract_ok": True,
        "payload_len": len(ext_bits),
    }


def flip_lsb_delta_pm1(q_flat, idx, qmax):
    old = int(q_flat[idx].item())

    if old >= qmax:
        new = old - 1
    elif old <= -qmax:
        new = old + 1
    else:
        # Flip LSB using a deployable ±1 integer perturbation.
        if old & 1:
            new = old - 1
        else:
            new = old + 1

    q_flat[idx] = int(new)
    return int(new - old)


def apply_defense_to_param(
    model,
    param_name,
    q_bits,
    block_size,
    edit_positions,
    mode="prefix",
    seed=2026,
    use_writable_mask=True,
):
    p = get_param(model, param_name)
    backup = p.detach().clone()

    q, scale, orig_shape, pad_len, writable, qmax = quantize_blocks(backup, bits=q_bits, block_size=block_size, use_writable_mask=use_writable_mask)
    q_flat = q.reshape(-1).to(torch.int16).clone()
    writable_idx = torch.where(writable.cpu())[0]

    edit_n = min(int(edit_positions), int(writable_idx.numel()))
    if edit_n <= 0:
        return {
            "param_name": param_name,
            "edited_positions": 0,
            "writable_positions": int(writable_idx.numel()),
            "delta_plus": 0,
            "delta_minus": 0,
        }

    if mode == "random":
        g = torch.Generator()
        g.manual_seed(int(seed))
        perm = torch.randperm(int(writable_idx.numel()), generator=g)[:edit_n]
        chosen = writable_idx[perm]
    else:
        # prefix mode matches the actual CNN stego write stream order.
        chosen = writable_idx[:edit_n]

    plus = 0
    minus = 0
    for idx_t in chosen.tolist():
        d = flip_lsb_delta_pm1(q_flat, int(idx_t), qmax)
        if d > 0:
            plus += 1
        elif d < 0:
            minus += 1

    q_new = q_flat.reshape_as(q).to(q.device)
    recon = dequantize_blocks(q_new, scale.to(q.device), orig_shape, pad_len).to(p.device).to(p.dtype)

    with torch.no_grad():
        p.copy_(recon)

    return {
        "param_name": param_name,
        "edited_positions": int(edit_n),
        "writable_positions": int(writable_idx.numel()),
        "delta_plus": int(plus),
        "delta_minus": int(minus),
    }


def estimate_pai_for_param(model, param_name, loader, device, q_bits, block_size, probe_positions, base_acc, base_logits, max_batches, seed, use_writable_mask=True):
    p = get_param(model, param_name)
    backup = p.detach().clone()

    # Temporary low-bit perturbation on writable prefix.
    info = apply_defense_to_param(
        model=model,
        param_name=param_name,
        q_bits=q_bits,
        block_size=block_size,
        edit_positions=probe_positions,
        mode="prefix",
        seed=seed,
        use_writable_mask=use_writable_mask,
    )

    after = evaluate(model, loader, device, max_batches=max_batches)
    cur_logits, _ = collect_logits(model, loader, device, max_batches=max_batches)

    with torch.no_grad():
        if base_logits.shape == cur_logits.shape:
            logit_mse = float(torch.mean((base_logits.float() - cur_logits.float()) ** 2).item())
        else:
            logit_mse = float("inf")

    acc_shift_abs = abs(after["acc"] - base_acc)
    dacc = acc_shift_abs / max(1e-6, abs(base_acc))
    pai = max(float(dacc), float(logit_mse))

    with torch.no_grad():
        p.copy_(backup.to(p.device).to(p.dtype))

    return {
        "param_name": param_name,
        "probe_positions": int(info["edited_positions"]),
        "writable_positions": int(info["writable_positions"]),
        "before_acc": base_acc,
        "probe_after_acc": after["acc"],
        "acc_shift_abs": float(acc_shift_abs),
        "dacc": float(dacc),
        "logit_mse": float(logit_mse),
        "pai": float(pai),
        "sstego_score": float(1.0 / max(1e-12, pai)),
    }




def apply_learned_u_stego_defense(
    model,
    ok_rows,
    loader,
    device,
    q_bits,
    block_size,
    args,
):
    """INSPECT-Opt learned-u defense for CNN stego.

    Sstego/PAI builds a candidate pool.
    Continuous u is optimized with Lattack + Lpres + Ltri.
    Final selected positions are learned by |u| > tau, not fixed.
    """
    try:
        from torch.func import functional_call
    except Exception:
        from torch.nn.utils.stateless import functional_call

    model.eval()

    # ---- collect calibration batches and reference logits ----
    calib = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.scan_max_batches and bi >= args.scan_max_batches:
                break

            if isinstance(batch, dict):
                x = batch["input"].to(device)
            else:
                x = batch[0].to(device)

            if hasattr(model, "fp"):
                x = x.to(model.fp)

            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            calib.append((x.detach(), out.detach().float()))

    if not calib:
        raise RuntimeError("No calibration batch for learned-u optimization.")

    # ---- select high-risk tensors by Sstego = 1 / max(PAI, floor) ----
    scored = []
    for r in ok_rows:
        pai = float(r["pai"])
        score = 1.0 / max(float(args.pai_floor), pai)
        rr = dict(r)
        rr["learned_sstego_score"] = float(score)
        scored.append(rr)

    max_score = max(float(r["learned_sstego_score"]) for r in scored)
    selected_rows = [
        r for r in scored
        if float(r["learned_sstego_score"]) >= float(args.auto_score_ratio) * max_score
    ]
    selected_rows.sort(key=lambda r: (-float(r["learned_sstego_score"]), float(r["pai"])))

    if int(args.auto_max_tensors) > 0:
        selected_rows = selected_rows[:int(args.auto_max_tensors)]

    if not selected_rows:
        selected_rows = scored[:1]

    # ---- build candidate pool; pool size is adaptive, final selected is learned ----
    infos = []
    all_scores = []
    total_candidates = 0

    for rank, row in enumerate(selected_rows):
        name = row["tensor_name"]
        p = get_param(model, name)
        q, scale, orig_shape, pad_len, writable, qmax = quantize_blocks(
            p.detach(), bits=q_bits, block_size=block_size,
            use_writable_mask=(not bool(getattr(args, "disable_writable_mask", False)))
        )

        writable_idx = torch.where(writable.cpu())[0]
        writable_n = int(writable_idx.numel())
        if writable_n <= 0:
            continue

        norm_score = float(row["learned_sstego_score"]) / max(1e-12, max_score)

        # Candidate-pool upper bound, not final perturbation count.
        # It adapts to writable capacity and PAI score.
        pool_by_sqrt = int(round(float(args.learned_pool_sqrt_scale) * (writable_n ** 0.5) * max(0.25, norm_score)))
        pool_n = max(int(args.learned_min_pool), pool_by_sqrt)
        if int(args.learned_max_pool_per_tensor) > 0:
            pool_n = min(pool_n, int(args.learned_max_pool_per_tensor))
        pool_n = min(pool_n, writable_n)

        chosen = writable_idx[:pool_n].long()

        # Prefix-aware but not prefix-fixed: Ppos is a soft score decay.
        local_rank = torch.arange(pool_n, dtype=torch.float32)
        ppos = torch.exp(-local_rank / max(1.0, float(args.learned_pos_decay)))
        local_score = float(norm_score) * ppos
        local_score = torch.clamp(local_score, min=1e-8)

        infos.append({
            "name": name,
            "param": p,
            "q": q.detach().to(device).float(),
            "q_shape": tuple(q.shape),
            "scale": scale.detach().to(device),
            "orig_shape": orig_shape,
            "pad_len": pad_len,
            "positions": chosen.to(device),
            "qmax": int(qmax),
            "dtype": p.dtype,
            "device": p.device,
            "row": row,
            "candidate_positions": int(pool_n),
        })
        all_scores.append(local_score)
        total_candidates += int(pool_n)

    if not infos or total_candidates <= 0:
        raise RuntimeError("learned-u candidate pool is empty.")

    if int(args.learned_max_total_candidates) > 0 and total_candidates > int(args.learned_max_total_candidates):
        # Keep strongest tensors first until the global candidate cap is met.
        remain = int(args.learned_max_total_candidates)
        new_infos, new_scores = [], []
        for info, sc in zip(infos, all_scores):
            if remain <= 0:
                break
            take = min(remain, int(info["candidate_positions"]))
            if take <= 0:
                continue
            info = dict(info)
            info["positions"] = info["positions"][:take]
            info["candidate_positions"] = int(take)
            new_infos.append(info)
            new_scores.append(sc[:take])
            remain -= take
        infos, all_scores = new_infos, new_scores
        total_candidates = sum(int(x["candidate_positions"]) for x in infos)

    score_vec = torch.cat([x.to(device) for x in all_scores], dim=0).float()
    score_vec = score_vec / (score_vec.max() + 1e-12)

    print(
        f"[+] learned_u candidate pool: selected_tensors={len(infos)}, "
        f"candidate_positions={total_candidates}, "
        f"tau={args.learned_tau}, steps={args.learned_steps}",
        flush=True,
    )

    # ---- optimization variable u ----
    raw_u = torch.nn.Parameter(
        torch.empty(total_candidates, device=device).uniform_(
            -float(args.learned_init_scale), float(args.learned_init_scale)
        )
    )
    opt = torch.optim.Adam([raw_u], lr=float(args.learned_lr))

    base_param_dict = {n: p for n, p in model.named_parameters()}

    def build_relaxed_params(u_vec):
        params = dict(base_param_dict)
        offset = 0
        for info in infos:
            k = int(info["candidate_positions"])
            u_part = u_vec[offset:offset+k]
            offset += k

            q_rel = info["q"].reshape(-1).clone()
            q_rel[info["positions"]] = q_rel[info["positions"]] + u_part
            q_rel = torch.clamp(q_rel, -info["qmax"], info["qmax"])
            q_rel = q_rel.reshape(info["q_shape"])

            recon = dequantize_blocks(
                q_rel,
                info["scale"],
                info["orig_shape"],
                info["pad_len"],
            ).to(info["device"]).to(info["dtype"])

            params[info["name"]] = recon
        return params

    history = []
    best_loss = None

    for step in range(int(args.learned_steps)):
        opt.zero_grad(set_to_none=True)

        u = torch.tanh(raw_u)
        params = build_relaxed_params(u)

        lpres = torch.zeros((), device=device)
        for x, ref in calib:
            cur = functional_call(model, params, (x,))
            if isinstance(cur, (list, tuple)):
                cur = cur[0]
            lpres = lpres + torch.mean((cur.float() - ref.to(cur.device).float()) ** 2)
        lpres = lpres / max(1, len(calib))

        # Same INSPECT-Opt form: high Sattack encourages large |u|.
        # Divide by sqrt(N) only to keep the numeric scale stable across candidate-pool sizes.
        denom = max(1.0, float(total_candidates) ** 0.5)
        lattack = -float(args.learned_lambda_a) * torch.sum(torch.abs(u) * score_vec) / denom

        # Sparse selection penalty:
        # only positions whose attack score is high enough should cross tau.
        l_sparse = float(args.learned_lambda_s) * torch.sum(torch.abs(u)) / denom

        ltri = torch.mean((u ** 2) * ((1.0 - u ** 2) ** 2))
        lweight = torch.mean(u ** 2)

        loss = lattack + l_sparse + float(args.learned_lambda_p) * lpres + float(args.learned_lambda_t) * ltri + float(args.learned_lambda_w) * lweight
        loss.backward()
        opt.step()

        with torch.no_grad():
            selected_now = int((torch.abs(torch.tanh(raw_u)) > float(args.learned_tau)).sum().item())
            item = {
                "step": int(step + 1),
                "loss": float(loss.detach().cpu().item()),
                "loss_attack": float(lattack.detach().cpu().item()),
                "loss_sparse": float(l_sparse.detach().cpu().item()),
                "loss_pres": float(lpres.detach().cpu().item()),
                "loss_tri": float(ltri.detach().cpu().item()),
                "loss_weight": float(lweight.detach().cpu().item()),
                "selected_now": int(selected_now),
            }
            history.append(item)
            best_loss = item["loss"] if best_loss is None else min(best_loss, item["loss"])

    # ---- harden u to Delta in {-1, 0, +1} and apply to model ----
    with torch.no_grad():
        u_final = torch.tanh(raw_u).detach()
        delta = torch.zeros_like(u_final)
        mask = torch.abs(u_final) > float(args.learned_tau)
        delta[mask] = torch.sign(u_final[mask])

        perturb_rows = []
        selected_total = 0
        offset = 0

        for rank, info in enumerate(infos):
            k = int(info["candidate_positions"])
            d_part = delta[offset:offset+k]
            offset += k

            changed = int((d_part != 0).sum().item())
            selected_total += changed

            q_new = info["q"].reshape(-1).clone()
            if changed > 0:
                q_new[info["positions"]] = q_new[info["positions"]] + d_part
            q_new = torch.clamp(q_new, -info["qmax"], info["qmax"])
            q_new = q_new.reshape(info["q_shape"])

            recon = dequantize_blocks(
                q_new,
                info["scale"],
                info["orig_shape"],
                info["pad_len"],
            ).to(info["device"]).to(info["dtype"])

            info["param"].copy_(recon)

            row = info["row"]
            perturb_rows.append({
                "rank": int(rank),
                "param_name": info["name"],
                "candidate_positions": int(k),
                "changed_params": int(changed),
                "pai": row.get("pai"),
                "acc_shift_abs_probe": row.get("acc_shift_abs"),
                "learned_sstego_score": row.get("learned_sstego_score"),
                "budget_mode": "learned_u",
                "tau": float(args.learned_tau),
            })

    stat = {
        "budget_mode": "learned_u",
        "candidate_positions": int(total_candidates),
        "selected_tensors": int(len(infos)),
        "selected_positions": int(selected_total),
        "tau": float(args.learned_tau),
        "steps": int(args.learned_steps),
        "best_loss": float(best_loss),
        "history_tail": history[-5:],
    }
    return perturb_rows, int(selected_total), stat

def load_stego_model_and_data(spec, device):
    repo = Path("~/DLCompilerAttack").expanduser()
    sys.path.insert(0, str(repo))
    import utils as dl_utils

    try:
        model = dl_utils.load_model(spec["task_id"], False)
    except TypeError:
        model = dl_utils.load_model(False)

    poison_path = Path(spec["poison_path"]).expanduser()
    ckpt = torch.load(poison_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    if isinstance(sd, dict):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[+] load_state_dict missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    model = model.to(device)
    if hasattr(model, "fp"):
        model.fp = torch.float32

    try:
        train_loader, valid_loader, test_loader = dl_utils.load_dataloader(
            spec["task_id"], False, spec["test_batch"], spec["test_batch"]
        )
    except TypeError:
        train_loader, valid_loader, test_loader = dl_utils.load_dataloader(
            False, spec["test_batch"], spec["test_batch"]
        )

    return model, ckpt, valid_loader, test_loader, poison_path


def run_one(args, model_key):
    spec = SPECS[model_key]
    out_dir = Path(args.out_dir).expanduser() / spec["out_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n===== CNN stego defense: {spec['model_name']} =====", flush=True)
    print(f"[+] out_dir={out_dir}", flush=True)
    print(f"[+] device={device}", flush=True)

    t0 = time.perf_counter()
    model, ckpt, valid_loader, test_loader, poison_path = load_stego_model_and_data(spec, device)

    stego_meta = ckpt.get("stego_meta", {})
    payload_bits_ref = ckpt.get("payload_bits_ref", None)
    if payload_bits_ref is None:
        raise RuntimeError("payload_bits_ref not found in stego checkpoint; cannot verify extraction.")

    q_bits = int(stego_meta.get("q_bits", args.q_bits))
    block_size = int(stego_meta.get("block_size", args.block_size))
    target_param_for_eval = stego_meta.get("param_name")

    before_clean_valid = evaluate(model, valid_loader, device, max_batches=args.scan_max_batches)
    before_clean_test = evaluate(model, test_loader, device, max_batches=args.final_max_batches)

    before_extract = extract_payload(model, target_param_for_eval, q_bits, block_size)
    before_payload = compare_extract(before_extract, payload_bits_ref)

    print(f"[+] before clean test={before_clean_test}", flush=True)
    print(f"[+] before extract={before_payload}", flush=True)

    base_logits, _ = collect_logits(model, valid_loader, device, max_batches=args.scan_max_batches)
    base_acc = before_clean_valid["acc"]

    # ---- Step 1: CNN tensor-base PAI scan ----
    scan_rows = []
    candidates = candidate_params(model, min_numel=args.min_numel)
    print(f"[+] candidate tensors={len(candidates)}", flush=True)

    for i, item in enumerate(candidates):
        name = item["name"]
        p = get_param(model, name).detach()
        q, scale, orig_shape, pad_len, writable, qmax = quantize_blocks(p, bits=q_bits, block_size=block_size, use_writable_mask=(not bool(args.disable_writable_mask)))
        writable_positions = int(writable.sum().item())

        row = {
            "tensor_name": name,
            "shape": str(item["shape"]),
            "numel": item["numel"],
            "writable_positions": writable_positions,
            "status": "",
        }

        if writable_positions < args.min_writable_positions:
            row["status"] = "skip_writable_too_small"
            scan_rows.append(row)
            continue

        try:
            pai = estimate_pai_for_param(
                model=model,
                param_name=name,
                loader=valid_loader,
                device=device,
                q_bits=q_bits,
                block_size=block_size,
                probe_positions=min(args.probe_positions, writable_positions),
                base_acc=base_acc,
                base_logits=base_logits,
                max_batches=args.scan_max_batches,
                seed=args.seed + i,
            )
            row.update(pai)
            row["status"] = "ok"
            print(f"[{i+1}/{len(candidates)}] {name}: PAI={row['pai']:.6g} acc_shift={row['acc_shift_abs']:.6f} writable={writable_positions}", flush=True)
        except Exception as e:
            row["status"] = f"error:{type(e).__name__}:{e}"
            print(f"[{i+1}/{len(candidates)}] ERROR {name}: {e}", flush=True)

        scan_rows.append(row)

    ok_rows = [r for r in scan_rows if r.get("status") == "ok"]
    if not ok_rows:
        raise RuntimeError("No valid PAI candidate found.")

    ok_rows.sort(key=lambda r: (float(r["pai"]), -int(r["writable_positions"])))

    learned_u_stat = None

    if args.budget_mode == "learned_u":
        perturb_rows, selected_total, learned_u_stat = apply_learned_u_stego_defense(
            model=model,
            ok_rows=ok_rows,
            loader=valid_loader,
            device=device,
            q_bits=q_bits,
            block_size=block_size,
            args=args,
        )

    else:
        header_bits = len(TRIGGER_SEQ)

        if args.budget_mode == "adaptive_prefix":
            # Paper-aligned adaptive selection:
            # Sstego = 1 / max(PAI, floor), then select only high-score tensors.
            # Position budget is protocol-aware: edit header/prefix area, not a fixed 4096 per tensor.
            scored = []
            for r in ok_rows:
                pai = float(r["pai"])
                score = 1.0 / max(float(args.pai_floor), pai)
                rr = dict(r)
                rr["adaptive_sstego_score"] = float(score)
                scored.append(rr)

            max_score = max(float(r["adaptive_sstego_score"]) for r in scored)
            selected_rows = [
                r for r in scored
                if float(r["adaptive_sstego_score"]) >= float(args.auto_score_ratio) * max_score
            ]

            selected_rows.sort(key=lambda r: (-float(r["adaptive_sstego_score"]), float(r["pai"])))

            if args.auto_max_tensors > 0:
                selected_rows = selected_rows[:args.auto_max_tensors]

            if not selected_rows:
                selected_rows = scored[:1]

            adaptive_prefix_bits = int(header_bits * max(1, int(args.prefix_header_multiplier)))

            print(
                f"[+] adaptive_prefix budget: header_bits={header_bits}, "
                f"prefix_bits_per_selected_tensor={adaptive_prefix_bits}, "
                f"selected_tensors={len(selected_rows)}, "
                f"score_ratio={args.auto_score_ratio}, pai_floor={args.pai_floor}",
                flush=True,
            )
        else:
            if args.max_tensors > 0:
                selected_rows = ok_rows[:args.max_tensors]
            else:
                selected_rows = ok_rows
            adaptive_prefix_bits = None

        # ---- Step 2/3/4: apply discrete low-bit purification on writable prefix ----
        perturb_rows = []
        selected_total = 0

        for rank, row in enumerate(selected_rows):
            name = row["tensor_name"]

            if args.budget_mode == "adaptive_prefix":
                edit_n = min(int(adaptive_prefix_bits), int(row["writable_positions"]))
            else:
                edit_n = min(args.per_tensor_edit_bits, int(row["writable_positions"]))

            if args.max_total_edit_bits > 0:
                remain = args.max_total_edit_bits - selected_total
                if remain <= 0:
                    break
                edit_n = min(edit_n, remain)

            info = apply_defense_to_param(
                model=model,
                param_name=name,
                q_bits=q_bits,
                block_size=block_size,
                edit_positions=edit_n,
                mode=args.position_rule,
                seed=args.seed + 1009 + rank,
            )
            selected_total += int(info["edited_positions"])
            perturb_rows.append({
                **info,
                "rank": rank,
                "pai": row.get("pai"),
                "acc_shift_abs_probe": row.get("acc_shift_abs"),
                "sstego_score": row.get("sstego_score"),
                "adaptive_sstego_score": row.get("adaptive_sstego_score"),
                "budget_mode": args.budget_mode,
                "header_bits": header_bits,
            })

    after_clean_valid = evaluate(model, valid_loader, device, max_batches=args.scan_max_batches)
    after_clean_test = evaluate(model, test_loader, device, max_batches=args.final_max_batches)

    after_extract = extract_payload(model, target_param_for_eval, q_bits, block_size)
    after_payload = compare_extract(after_extract, payload_bits_ref)

    # save defended checkpoint
    defended_path = out_dir / "defended_cnn_stego.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "payload_bits_ref": [int(b) for b in payload_bits_ref],
        "original_stego_meta": stego_meta,
        "defense_meta": {
            "model_key": model_key,
            "model_name": spec["model_name"],
            "attack_type": "cnn_stego",
            "defense_algorithm": "INSPECT-Opt-CNN-Stego-Prefix-LSB",
            "q_bits": q_bits,
            "block_size": block_size,
            "position_rule": args.position_rule,
            "budget_mode": args.budget_mode,
            "pai_floor": args.pai_floor,
            "auto_score_ratio": args.auto_score_ratio,
            "auto_max_tensors": args.auto_max_tensors,
            "prefix_header_multiplier": args.prefix_header_multiplier,
            "learned_u": learned_u_stat,
            "per_tensor_edit_bits": args.per_tensor_edit_bits,
            "max_tensors": args.max_tensors,
            "max_total_edit_bits": args.max_total_edit_bits,
            "selected_total": selected_total,
            "seed": args.seed,
        }
    }, defended_path)

    model_total_2d = sum(int(p.numel()) for _, p in model.named_parameters() if p.dim() >= 2)
    k_actual = 100.0 * float(selected_total) / max(1, model_total_2d)

    # write tables
    with open(out_dir / "inspect_cnn_stego_pai.csv", "w", newline="") as f:
        fieldnames = sorted({k for r in scan_rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(scan_rows)

    with open(out_dir / "discrete_perturbation_by_tensor.csv", "w", newline="") as f:
        fieldnames = sorted({k for r in perturb_rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(perturb_rows)

    result = {
        "model_key": model_key,
        "model_name": spec["model_name"],
        "poison_path": str(poison_path),
        "poison_sha256": sha256_file(poison_path),
        "defended_path": str(defended_path),
        "attack_type": "cnn_stego",
        "defense_algorithm": "INSPECT-Opt-CNN-Stego-Prefix-LSB",
        "q_bits": q_bits,
        "block_size": block_size,
        "target_param_for_eval": target_param_for_eval,
        "payload_len_bits": len(payload_bits_ref),
        "before": {
            "clean_accuracy": before_clean_test["acc"],
            "valid_clean_accuracy": before_clean_valid["acc"],
            **before_payload,
        },
        "after": {
            "clean_accuracy": after_clean_test["acc"],
            "valid_clean_accuracy": after_clean_valid["acc"],
            **after_payload,
            "inspect_selected": int(selected_total),
            "inspect_total_2d": int(model_total_2d),
            "inspect_k_actual": float(k_actual),
        },
        "inspect_opt": {
            "objective": "Sstego/PAI + writable mask + prefix-position purification; Delta in {-1,+1} on selected LSB positions",
            "scan_max_batches": args.scan_max_batches,
            "final_max_batches": args.final_max_batches,
            "probe_positions": args.probe_positions,
            "per_tensor_edit_bits": args.per_tensor_edit_bits,
            "max_tensors": args.max_tensors,
            "max_total_edit_bits": args.max_total_edit_bits,
            "position_rule": args.position_rule,
            "budget_mode": args.budget_mode,
            "pai_floor": args.pai_floor,
            "auto_score_ratio": args.auto_score_ratio,
            "auto_max_tensors": args.auto_max_tensors,
            "prefix_header_multiplier": args.prefix_header_multiplier,
            "learned_u": learned_u_stat,
            "learned_steps": args.learned_steps,
            "learned_tau": args.learned_tau,
            "learned_lambda_a": args.learned_lambda_a,
            "learned_lambda_s": args.learned_lambda_s,
            "learned_pool_sqrt_scale": args.learned_pool_sqrt_scale,
            "learned_max_pool_per_tensor": args.learned_max_pool_per_tensor,
            "learned_max_total_candidates": args.learned_max_total_candidates,
            "disable_writable_mask": bool(args.disable_writable_mask),
            "selected_tensors": len(perturb_rows),
            "selected_positions": int(selected_total),
            "k_star_model_percent": float(k_actual),
        },
        "timing": {
            "total_time_sec": time.perf_counter() - t0,
        },
        "notes": [
            "CNN 隐写主指标为 payload extraction：防御前 exact_match=true，防御后 exact_match=false 或 bit_errors>0。",
            "防御不使用 stego_meta.param_name 选择目标；该字段仅用于防御前后提取验证。",
            "候选张量通过 CNN tensor-base PAI 排序，并在 writable mask 的 prefix 位置执行 ±1 低位净化。",
        ],
    }

    with open(out_dir / "defense_result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    flat = {
        "model_key": model_key,
        "model_name": spec["model_name"],
        "before_clean_accuracy": before_clean_test["acc"],
        "after_clean_accuracy": after_clean_test["acc"],
        "before_extract_exact_match": before_payload["extract_exact_match"],
        "after_extract_exact_match": after_payload["extract_exact_match"],
        "before_bit_errors": before_payload["bit_errors"],
        "after_bit_errors": after_payload["bit_errors"],
        "before_extract_reason": before_payload["extract_reason"],
        "after_extract_reason": after_payload["extract_reason"],
        "inspect_selected": int(selected_total),
        "inspect_k_actual": float(k_actual),
        "target_param_for_eval": target_param_for_eval,
        "defended_path": str(defended_path),
    }

    with open(out_dir / "defense_result.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        w.writeheader()
        w.writerow(flat)

    print("\n=== FINAL CNN STEGO DEFENSE RESULT ===", flush=True)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(f"\n[+] Done. Outputs are in: {out_dir}", flush=True)
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["c100_v19", "tiny_r34", "all"], required=True)
    ap.add_argument("--out-dir", default="~/test_ssj/outputs/cnn_stego_def_paper_full_seed2026")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--q-bits", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--min-numel", type=int, default=1024)
    ap.add_argument("--min-writable-positions", type=int, default=1024)
    ap.add_argument("--scan-max-batches", type=int, default=5)
    ap.add_argument("--final-max-batches", type=int, default=0)
    ap.add_argument("--probe-positions", type=int, default=2048)
    ap.add_argument("--per-tensor-edit-bits", type=int, default=4096)
    ap.add_argument("--max-tensors", type=int, default=0, help="0 means all valid PAI candidates")
    ap.add_argument("--max-total-edit-bits", type=int, default=0, help="0 means no global cap")
    ap.add_argument("--position-rule", choices=["prefix", "random"], default="prefix")
    ap.add_argument("--budget-mode", choices=["fixed", "adaptive_prefix", "learned_u"], default="fixed")
    ap.add_argument("--pai-floor", type=float, default=1e-8)
    ap.add_argument("--auto-score-ratio", type=float, default=0.25)
    ap.add_argument("--auto-max-tensors", type=int, default=8)
    ap.add_argument("--prefix-header-multiplier", type=int, default=4)
    ap.add_argument("--learned-steps", type=int, default=80)
    ap.add_argument("--learned-lr", type=float, default=0.08)
    ap.add_argument("--learned-tau", type=float, default=0.35)
    ap.add_argument("--learned-lambda-a", type=float, default=8.0)
    ap.add_argument("--learned-lambda-s", type=float, default=0.0)
    ap.add_argument("--learned-lambda-p", type=float, default=1.0)
    ap.add_argument("--learned-lambda-t", type=float, default=0.01)
    ap.add_argument("--learned-lambda-w", type=float, default=1e-4)
    ap.add_argument("--learned-init-scale", type=float, default=0.02)
    ap.add_argument("--learned-pool-sqrt-scale", type=float, default=2.0)
    ap.add_argument("--learned-min-pool", type=int, default=64)
    ap.add_argument("--learned-max-pool-per-tensor", type=int, default=1024)
    ap.add_argument("--learned-max-total-candidates", type=int, default=8192)
    ap.add_argument("--disable-writable-mask", action="store_true", help="Ablation: do not protect scale-determining positions in quantized blocks.")
    ap.add_argument("--learned-pos-decay", type=float, default=256.0)
    args = ap.parse_args()

    set_seed(args.seed)

    models = ["c100_v19", "tiny_r34"] if args.model == "all" else [args.model]
    rows = []
    for m in models:
        rows.append(run_one(args, m))

    out_root = Path(args.out_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    with open(out_root / "defense_summary.csv", "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    with open(out_root / "defense_summary.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\n[+] All done. Summary: {out_root / 'defense_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
