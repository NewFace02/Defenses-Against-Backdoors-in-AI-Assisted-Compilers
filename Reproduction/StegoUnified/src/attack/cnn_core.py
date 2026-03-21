import copy
from typing import List, Dict, Any

import torch
import torch.nn as nn

from Reproduction.StegoUnified.utils import evaluate_cnn
from Reproduction.StegoUnified.src.common import BlockQuantizer, TRIGGER_SEQ


class CNNStegoCore:
    """
    CNN 版隐写攻击核心骨架：
    1. 管理 CNN 模型
    2. 扫描候选参数张量
    3. 评估 clean accuracy
    4. 估计隐写容量
    5. 测量量化-反量化带来的性能损失
    """

    def __init__(self, model: nn.Module, device: torch.device, valid_loader=None, verbose: bool = True):
        self.model = model.to(device)
        self.device = device
        self.valid_loader = valid_loader
        self.verbose = verbose

    # =========================
    # Evaluation
    # =========================
    def evaluate_clean(self, dataloader=None) -> Dict[str, Any]:
        loader = dataloader if dataloader is not None else self.valid_loader
        if loader is None:
            raise ValueError("No dataloader provided for evaluation.")
        return evaluate_cnn(self.model, loader, self.device)

    # =========================
    # Parameter access
    # =========================
    def get_named_param(self, param_name: str) -> nn.Parameter:
        for name, param in self.model.named_parameters():
            if name == param_name:
                return param
        raise KeyError(f"Parameter not found: {param_name}")

    def backup_param(self, param_name: str) -> torch.Tensor:
        param = self.get_named_param(param_name)
        return param.detach().clone()

    def restore_param(self, param_name: str, tensor: torch.Tensor):
        param = self.get_named_param(param_name)
        with torch.no_grad():
            param.copy_(tensor.to(param.device).to(param.dtype))

    # =========================
    # Candidate scanning
    # =========================
    def get_candidate_params(
        self,
        min_numel: int = 1024,
        only_weight: bool = True,
        allow_conv: bool = True,
        allow_fc: bool = True,
        exclude_bn: bool = True,
    ) -> List[Dict[str, Any]]:
        candidates = []

        for name, param in self.model.named_parameters():
            if only_weight and not name.endswith("weight"):
                continue

            if exclude_bn and ("bn" in name.lower()):
                continue

            is_conv = name.startswith("conv")
            is_fc = name.startswith("fc")

            if is_conv and not allow_conv:
                continue
            if is_fc and not allow_fc:
                continue
            if (not is_conv) and (not is_fc):
                continue

            numel = param.numel()
            if numel < min_numel:
                continue

            candidates.append({
                "name": name,
                "shape": tuple(param.shape),
                "numel": numel,
                "dtype": str(param.dtype),
                "device": str(param.device),
            })

        return candidates

    def print_candidate_params(
        self,
        min_numel: int = 1024,
        only_weight: bool = True,
        allow_conv: bool = True,
        allow_fc: bool = True,
        exclude_bn: bool = True,
    ):
        candidates = self.get_candidate_params(
            min_numel=min_numel,
            only_weight=only_weight,
            allow_conv=allow_conv,
            allow_fc=allow_fc,
            exclude_bn=exclude_bn,
        )

        if len(candidates) == 0:
            print("No candidate parameters found.")
            return

        print("=" * 80)
        print("Candidate parameter tensors for CNN steganography")
        print("=" * 80)
        for i, item in enumerate(candidates):
            print(
                f"[{i}] {item['name']:<15} "
                f"shape={item['shape']!s:<20} "
                f"numel={item['numel']:<8} "
                f"dtype={item['dtype']}"
            )
        print("=" * 80)

    # =========================
    # Safe perturbation smoke test
    # =========================
    def perturb_param_and_measure(
        self,
        param_name: str,
        epsilon: float = 1e-4,
        max_edit_elems: int = 2048,
        dataloader=None,
    ) -> Dict[str, Any]:
        loader = dataloader if dataloader is not None else self.valid_loader
        if loader is None:
            raise ValueError("No dataloader provided for evaluation.")

        before = self.evaluate_clean(loader)
        backup = self.backup_param(param_name)

        param = self.get_named_param(param_name)
        flat = param.reshape(-1)

        edit_n = min(max_edit_elems, flat.numel())
        if edit_n <= 0:
            raise ValueError(f"Parameter {param_name} has no editable elements.")

        ref_scale = flat[:edit_n].abs().mean().item()
        if ref_scale < 1e-12:
            ref_scale = 1.0

        with torch.no_grad():
            flat[:edit_n].add_(epsilon * ref_scale)

        after = self.evaluate_clean(loader)

        self.restore_param(param_name, backup)
        restored = self.evaluate_clean(loader)

        return {
            "param_name": param_name,
            "edited_elements": edit_n,
            "epsilon": epsilon,
            "before": before,
            "after": after,
            "restored": restored,
            "acc_drop_after": before["acc"] - after["acc"],
            "acc_restore_gap": abs(before["acc"] - restored["acc"]),
            "loss_restore_gap": abs(before["loss"] - restored["loss"]),
        }

    # =========================
    # Capacity estimation
    # =========================
    def get_param_capacity_info(
        self,
        param_name: str,
        q_bits: int = 8,
        block_size: int = 32,
        use_ecc: bool = True,
        repetition_factor: int = 5,
    ) -> Dict[str, Any]:
        """
        估算某个参数张量在当前量化设置下的可嵌入容量。
        默认假设：每个可写量化整数只写 1 bit（LSB）。
        Header = Trigger(16) + N*(4) + Len(16) = 36 bits
        """
        param = self.get_named_param(param_name).detach()

        q_blocks, scale, orig_shape, pad_len = BlockQuantizer.quantize(
            param, bits=q_bits, block_size=block_size
        )
        mask = BlockQuantizer.get_protection_mask(param, block_size=block_size)

        total_positions = int(mask.numel())
        writable_positions = int(mask.sum().item())
        protected_positions = total_positions - writable_positions

        header_bits = len(TRIGGER_SEQ) + 4 + 16
        usable_stream_bits = max(writable_positions - header_bits, 0)

        if use_ecc:
            usable_data_bits = usable_stream_bits // repetition_factor
        else:
            usable_data_bits = usable_stream_bits

        usable_data_bytes = usable_data_bits // 8

        return {
            "param_name": param_name,
            "shape": tuple(orig_shape),
            "numel": int(param.numel()),
            "q_bits": q_bits,
            "block_size": block_size,
            "pad_len": int(pad_len),
            "num_blocks": int(q_blocks.shape[0]),
            "total_positions": total_positions,
            "writable_positions": writable_positions,
            "protected_positions": protected_positions,
            "header_bits": header_bits,
            "usable_stream_bits": usable_stream_bits,
            "usable_data_bits": usable_data_bits,
            "usable_data_bytes": usable_data_bytes,
            "use_ecc": use_ecc,
            "repetition_factor": repetition_factor,
        }

    # =========================
    # Quantization roundtrip check
    # =========================
    def quantize_roundtrip_and_measure(
        self,
        param_name: str,
        q_bits: int = 8,
        block_size: int = 32,
        dataloader=None,
    ) -> Dict[str, Any]:
        """
        对指定参数做:
        原参数 -> 量化 -> 反量化 -> 替换
        然后测量 clean acc 变化，再恢复参数。
        """
        loader = dataloader if dataloader is not None else self.valid_loader
        if loader is None:
            raise ValueError("No dataloader provided for evaluation.")

        before = self.evaluate_clean(loader)
        backup = self.backup_param(param_name)

        param = self.get_named_param(param_name)
        q_blocks, scale, orig_shape, pad_len = BlockQuantizer.quantize(
            backup, bits=q_bits, block_size=block_size
        )
        recon = BlockQuantizer.dequantize(
            q_blocks, scale, orig_shape, pad_len
        ).to(param.dtype).to(param.device)

        roundtrip_mse = torch.mean((backup - recon).pow(2)).item()
        roundtrip_mae = torch.mean((backup - recon).abs()).item()

        with torch.no_grad():
            param.copy_(recon)

        after = self.evaluate_clean(loader)

        self.restore_param(param_name, backup)
        restored = self.evaluate_clean(loader)

        return {
            "param_name": param_name,
            "q_bits": q_bits,
            "block_size": block_size,
            "roundtrip_mse": roundtrip_mse,
            "roundtrip_mae": roundtrip_mae,
            "before": before,
            "after": after,
            "restored": restored,
            "acc_drop_after": before["acc"] - after["acc"],
            "acc_restore_gap": abs(before["acc"] - restored["acc"]),
            "loss_restore_gap": abs(before["loss"] - restored["loss"]),
        }

    # =========================
    # Bit helpers
    # =========================
    @staticmethod
    def _int_to_bits(value: int, n_bits: int) -> List[int]:
        return [(value >> i) & 1 for i in range(n_bits - 1, -1, -1)]

    @staticmethod
    def _bits_to_int(bits: List[int]) -> int:
        value = 0
        for b in bits:
            value = (value << 1) | int(b)
        return value

    @staticmethod
    def _repeat_encode(bits: List[int], n: int) -> List[int]:
        if n <= 1:
            return [int(b) for b in bits]
        out = []
        for b in bits:
            out.extend([int(b)] * n)
        return out

    @staticmethod
    def _repeat_decode(bits: List[int], n: int) -> List[int]:
        if n <= 1:
            return [int(b) for b in bits]

        out = []
        usable_len = (len(bits) // n) * n
        for i in range(0, usable_len, n):
            chunk = bits[i:i + n]
            ones = sum(chunk)
            out.append(1 if ones > (n // 2) else 0)
        return out

    def _build_payload_stream(self, payload_bits: List[int], n: int) -> List[int]:
        payload_bits = [int(b) & 1 for b in payload_bits]
        encoded_payload = self._repeat_encode(payload_bits, n)

        # Header:
        # trigger(16) + n(4) + payload_len_bits(16)
        header = []
        header.extend(TRIGGER_SEQ)
        header.extend(self._int_to_bits(n, 4))
        header.extend(self._int_to_bits(len(payload_bits), 16))

        return header + encoded_payload

    # =========================
    # Real stego embedding / extraction
    # =========================
    def embed_payload_in_param(
        self,
        param_name: str,
        payload_bits: List[int],
        q_bits: int = 8,
        block_size: int = 32,
        n: int = 5,
    ) -> Dict[str, Any]:
        """
        将 payload_bits 嵌入指定参数张量的量化整数 LSB 中。
        """
        backup = self.backup_param(param_name)
        param = self.get_named_param(param_name)

        q_blocks, scale, orig_shape, pad_len = BlockQuantizer.quantize(
            backup, bits=q_bits, block_size=block_size
        )
        mask = BlockQuantizer.get_protection_mask(
            backup, block_size=block_size
        ).reshape(-1).bool()

        writable_idx = torch.where(mask)[0]
        stream = self._build_payload_stream(payload_bits, n)

        if len(stream) > int(writable_idx.numel()):
            raise ValueError(
                f"Payload too large for {param_name}: "
                f"need {len(stream)} writable positions, "
                f"but only {int(writable_idx.numel())} available."
            )

        q_flat = q_blocks.reshape(-1).to(torch.int64).clone()

        for i, bit in enumerate(stream):
            idx = int(writable_idx[i].item())
            q_flat[idx] = (q_flat[idx] & ~1) | int(bit)

        q_new = q_flat.reshape_as(q_blocks)
        recon = BlockQuantizer.dequantize(
            q_new, scale, orig_shape, pad_len
        ).to(param.dtype).to(param.device)

        with torch.no_grad():
            param.copy_(recon)

        return {
            "param_name": param_name,
            "payload_bits": len(payload_bits),
            "stream_bits": len(stream),
            "writable_positions": int(writable_idx.numel()),
            "q_bits": q_bits,
            "block_size": block_size,
            "n": n,
        }

    def extract_payload_from_param(
        self,
        param_name: str,
        q_bits: int = 8,
        block_size: int = 32,
        search_limit: int = 4096,
    ) -> Dict[str, Any]:
        """
        从指定参数张量中提取 payload。
        当前版本会在前 search_limit 个可写 bit 中搜索 trigger。
        """
        param = self.get_named_param(param_name).detach()

        q_blocks, scale, orig_shape, pad_len = BlockQuantizer.quantize(
            param, bits=q_bits, block_size=block_size
        )
        mask = BlockQuantizer.get_protection_mask(
            param, block_size=block_size
        ).reshape(-1).bool()

        writable_idx = torch.where(mask)[0]
        q_flat = q_blocks.reshape(-1).to(torch.int64)

        bit_stream = []
        for idx in writable_idx.tolist():
            bit_stream.append(int((q_flat[idx] & 1).item()))

        trig_len = len(TRIGGER_SEQ)
        max_start = min(max(len(bit_stream) - trig_len + 1, 0), search_limit)

        start_pos = -1
        for s in range(max_start):
            if bit_stream[s:s + trig_len] == TRIGGER_SEQ:
                start_pos = s
                break

        if start_pos < 0:
            return {
                "ok": False,
                "reason": "trigger_not_found",
                "payload_bits": [],
            }

        pos = start_pos + trig_len

        if pos + 4 + 16 > len(bit_stream):
            return {
                "ok": False,
                "reason": "header_incomplete",
                "payload_bits": [],
            }

        n = self._bits_to_int(bit_stream[pos:pos + 4])
        pos += 4

        payload_len = self._bits_to_int(bit_stream[pos:pos + 16])
        pos += 16

        if n <= 0:
            return {
                "ok": False,
                "reason": "invalid_repetition_factor",
                "payload_bits": [],
            }

        encoded_len = payload_len * n
        if pos + encoded_len > len(bit_stream):
            return {
                "ok": False,
                "reason": "payload_incomplete",
                "payload_bits": [],
            }

        encoded_payload = bit_stream[pos:pos + encoded_len]
        payload_bits = self._repeat_decode(encoded_payload, n)

        payload_bits = payload_bits[:payload_len]

        return {
            "ok": True,
            "reason": "success",
            "payload_bits": payload_bits,
            "payload_len": payload_len,
            "n": n,
            "start_pos": start_pos,
        }

    def embed_extract_and_measure(
        self,
        param_name: str,
        payload_bits: List[int],
        q_bits: int = 8,
        block_size: int = 32,
        n: int = 5,
        dataloader=None,
    ) -> Dict[str, Any]:
        """
        端到端测试：
        1. 评估嵌入前 clean acc
        2. 嵌入 payload
        3. 评估嵌入后 clean acc
        4. 提取 payload 并对比
        5. 恢复参数
        6. 再次评估恢复后的 clean acc
        """
        loader = dataloader if dataloader is not None else self.valid_loader
        if loader is None:
            raise ValueError("No dataloader provided for evaluation.")

        before = self.evaluate_clean(loader)
        backup = self.backup_param(param_name)

        embed_info = self.embed_payload_in_param(
            param_name=param_name,
            payload_bits=payload_bits,
            q_bits=q_bits,
            block_size=block_size,
            n=n,
        )

        after = self.evaluate_clean(loader)

        extract_info = self.extract_payload_from_param(
            param_name=param_name,
            q_bits=q_bits,
            block_size=block_size,
        )

        extracted_ok = False
        bit_errors = None

        if extract_info["ok"]:
            extracted_bits = extract_info["payload_bits"]
            bit_errors = sum(
                int(a != b)
                for a, b in zip(payload_bits, extracted_bits)
            )
            extracted_ok = (
                len(extracted_bits) == len(payload_bits) and bit_errors == 0
            )

        self.restore_param(param_name, backup)
        restored = self.evaluate_clean(loader)

        return {
            "param_name": param_name,
            "q_bits": q_bits,
            "block_size": block_size,
            "n": n,
            "before": before,
            "after": after,
            "restored": restored,
            "embed_info": embed_info,
            "extract_info": extract_info,
            "extracted_ok": extracted_ok,
            "bit_errors": bit_errors,
            "acc_drop_after": before["acc"] - after["acc"],
            "acc_restore_gap": abs(before["acc"] - restored["acc"]),
        }

    # =========================
    # Best target scanning
    # =========================
    def scan_best_target(
        self,
        payload_bits: List[int],
        q_bits: int = 8,
        block_size: int = 32,
        n: int = 5,
        min_numel: int = 1024,
        dataloader=None,
    ) -> List[Dict[str, Any]]:
        """
        扫描所有候选参数张量，筛选出：
        1) 容量足够
        2) embed / extract 成功
        3) acc drop 尽量小

        返回结果按 acc_drop_after 从小到大排序。
        """
        loader = dataloader if dataloader is not None else self.valid_loader
        if loader is None:
            raise ValueError("No dataloader provided for evaluation.")

        candidates = self.get_candidate_params(min_numel=min_numel)
        results = []

        for item in candidates:
            param_name = item["name"]

            cap = self.get_param_capacity_info(
                param_name=param_name,
                q_bits=q_bits,
                block_size=block_size,
                use_ecc=True,
                repetition_factor=n,
            )

            if cap["usable_data_bits"] < len(payload_bits):
                results.append({
                    "param_name": param_name,
                    "status": "capacity_not_enough",
                    "usable_data_bits": cap["usable_data_bits"],
                    "required_bits": len(payload_bits),
                })
                continue

            test_result = self.embed_extract_and_measure(
                param_name=param_name,
                payload_bits=payload_bits,
                q_bits=q_bits,
                block_size=block_size,
                n=n,
                dataloader=loader,
            )

            results.append({
                "param_name": param_name,
                "status": "ok" if test_result["extracted_ok"] else "extract_failed",
                "usable_data_bits": cap["usable_data_bits"],
                "required_bits": len(payload_bits),
                "before_acc": test_result["before"]["acc"],
                "after_acc": test_result["after"]["acc"],
                "acc_drop_after": test_result["acc_drop_after"],
                "acc_restore_gap": test_result["acc_restore_gap"],
                "bit_errors": test_result["bit_errors"],
                "extracted_ok": test_result["extracted_ok"],
            })

        # 只对成功项排序
        ok_items = [x for x in results if x["status"] == "ok"]
        bad_items = [x for x in results if x["status"] != "ok"]

        for item in ok_items:
            item["acc_shift_abs"] = abs(item["after_acc"] - item["before_acc"])

        ok_items.sort(
            key=lambda x: (
                x["acc_shift_abs"],      # 先看精度变化绝对值，越小越好
                -x["usable_data_bits"]   # 再看容量，越大越好
            )
        )
        return ok_items + bad_items

    def embed_and_save_checkpoint(
        self,
        save_path: str,
        param_name: str,
        payload_bits: List[int],
        q_bits: int = 8,
        block_size: int = 32,
        n: int = 5,
        extra_meta: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        将 payload 真正嵌入当前模型，并保存 stego checkpoint。
        注意：这个函数不会自动恢复参数，因为它的目标就是生成 stego 模型。
        """
        embed_info = self.embed_payload_in_param(
            param_name=param_name,
            payload_bits=payload_bits,
            q_bits=q_bits,
            block_size=block_size,
            n=n,
        )

        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "payload_bits_ref": [int(b) for b in payload_bits],
            "stego_meta": {
                "param_name": param_name,
                "payload_len_bits": len(payload_bits),
                "q_bits": q_bits,
                "block_size": block_size,
                "n": n,
                **({} if extra_meta is None else extra_meta),
            }
        }

        torch.save(ckpt, save_path)

        return {
            "save_path": save_path,
            "embed_info": embed_info,
            "stego_meta": ckpt["stego_meta"],
        }

    # =========================
    # Utilities
    # =========================
    def clone_model(self):
        return copy.deepcopy(self.model).to(self.device)