import os
import math
import inspect
from typing import Optional, List

import torch

from Reproduction.StegoUnified.utils import get_device, get_random_bits
from Reproduction.StegoUnified.src.common import BuildConfig, BuildResult, BlockQuantizer, StegoProtocol
from Reproduction.StegoUnified.src.common.io import ensure_dir, save_json
from Reproduction.StegoUnified.src.models.llm import load_tinyllama
from Reproduction.StegoUnified.src.attack import SASERCore
from Reproduction.StegoUnified.src.backends.base_builder import BaseBuilder


class LLMBuilder(BaseBuilder):
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device if device is not None else get_device()

    def _load_clean_model(self, clean_model_path: str):
        sig = inspect.signature(load_tinyllama)
        kwargs = {}

        if "model_source" in sig.parameters:
            kwargs["model_source"] = clean_model_path
        elif "model_path" in sig.parameters:
            kwargs["model_path"] = clean_model_path
        elif "model_name_or_path" in sig.parameters:
            kwargs["model_name_or_path"] = clean_model_path
        elif "model_name" in sig.parameters:
            kwargs["model_name"] = clean_model_path

        if "device" in sig.parameters:
            kwargs["device"] = self.device

        result = load_tinyllama(**kwargs) if len(kwargs) > 0 else load_tinyllama(clean_model_path)

        if isinstance(result, tuple) and len(result) >= 2:
            model, tokenizer = result[0], result[1]
        else:
            model, tokenizer = result, None

        return model, tokenizer

    @staticmethod
    def _parse_layer_idx(target_group: Optional[str]):
        if target_group is None:
            return None
        s = str(target_group).strip()
        if s == "":
            return None
        if s.startswith("layer_"):
            return int(s.split("_")[-1])
        return int(s)

    def _resolve_search_layers(self, core: SASERCore, config: BuildConfig) -> List[int]:
        # 如果配置里手动指定了搜索层，就按配置来
        if "search_layer_indices" in config.extra:
            return [int(x) for x in config.extra["search_layer_indices"]]

        # 否则默认全层搜索
        num_layers = core.model.config.num_hidden_layers
        return list(range(num_layers))

    def _choose_target_group(
        self,
        core: SASERCore,
        config: BuildConfig,
        payload_bytes: bytes,
    ):
        fixed_layer = self._parse_layer_idx(config.target_group)
        if fixed_layer is not None:
            return fixed_layer, 1, None, None, None

        base_ppl, base_acc = core.evaluate_metrics()

        q_bits = config.q_bits
        if q_bits == 4:
            n_candidates = [1, 2]
        else:
            n_candidates = [1, 2, 3]

        use_ecc = True
        est_bits_len = len(StegoProtocol.pack_payload(payload_bytes, 1, use_ecc=use_ecc))

        search_layers = self._resolve_search_layers(core, config)
        best = {"layer": 0, "n": 1, "pai": float("inf")}
        search_records = []

        for l_idx in search_layers:
            params = core.get_layer_params(l_idx)
            backups = [p.data.clone() for p in params]

            layer_best_pai = float("inf")
            layer_best_n = 1

            for n_test in n_candidates:
                rand_bits = get_random_bits(est_bits_len, n_test)
                core.launch_attack(params, rand_bits, n_test, q_bits, mode="robust")

                # 模拟分发后量化环境
                for p in params:
                    q, s, sh, pad = BlockQuantizer.quantize(p.data, q_bits)
                    p.data.copy_(BlockQuantizer.dequantize(q, s, sh, pad).to(p.device))

                adv_ppl, adv_acc = core.evaluate_metrics()

                if not math.isfinite(adv_ppl):
                    d_ppl = 1.0
                else:
                    d_ppl = abs(1.0 / base_ppl - 1.0 / adv_ppl) / (1.0 / base_ppl)

                d_acc = abs(base_acc - adv_acc) / (base_acc if base_acc > 0 else 1e-6)
                pai = max(d_ppl, d_acc)

                if pai < layer_best_pai:
                    layer_best_pai = pai
                    layer_best_n = n_test

                if pai < best["pai"]:
                    best = {"layer": l_idx, "n": n_test, "pai": pai}

                # 恢复
                for i, p in enumerate(params):
                    p.data.copy_(backups[i])

            search_records.append({
                "layer": l_idx,
                "best_n": layer_best_n,
                "best_pai": layer_best_pai,
            })

        return best["layer"], best["n"], best["pai"], base_ppl, base_acc, search_records

    def build(self, config: BuildConfig) -> BuildResult:
        if config.model_family != "llm":
            raise ValueError(f"LLMBuilder only supports model_family='llm', got: {config.model_family}")

        if not config.clean_model_path:
            raise ValueError("clean_model_path is required.")

        if not config.output_dir:
            raise ValueError("output_dir is required.")

        ensure_dir(config.output_dir)

        # 1) load clean model
        model, tokenizer = self._load_clean_model(config.clean_model_path)

        # 2) build core
        core = SASERCore(model, tokenizer, self.device)

        # 可在配置里限制评估任务数，先跑通更快
        # 例如 extra: {"eval_task_limit": 4}
        if "eval_task_limit" in config.extra:
            k = int(config.extra["eval_task_limit"])
            if k > 0:
                core.eval_tasks = core.eval_tasks[:k]

        # 3) payload bytes
        payload_bytes_len = max(4, math.ceil(config.payload_len_bits / 8))
        payload_bytes = b"\x7fELF" + os.urandom(max(0, payload_bytes_len - 4))

        # 4) choose target
        choose_result = self._choose_target_group(core, config, payload_bytes)
        if len(choose_result) == 5:
            target_layer, n_star, best_pai, base_ppl, base_acc = choose_result
            search_records = None
        else:
            target_layer, n_star, best_pai, base_ppl, base_acc, search_records = choose_result

        if base_ppl is None or base_acc is None:
            base_ppl, base_acc = core.evaluate_metrics()

        # 5) actual embed
        params = core.get_layer_params(target_layer)
        bits_final = StegoProtocol.pack_payload(payload_bytes, n_star, use_ecc=True)
        core.launch_attack(params, bits_final, n_star, config.q_bits, mode="robust")

        # 模拟最终分发量化环境
        for p in params:
            q, s, sh, pad = BlockQuantizer.quantize(p.data, config.q_bits)
            p.data.copy_(BlockQuantizer.dequantize(q, s, sh, pad).to(p.device))

        # 6) evaluate poisoned model
        after_ppl, after_acc = core.evaluate_metrics()

        # 7) immediate extraction verify
        found, ex_bits = core.extract_payload_auto(
            params,
            n_star,
            config.q_bits,
            env_is_quantized=True,
            use_ecc=True
        )
        extracted_bytes = StegoProtocol.bits_to_bytes(ex_bits)
        extract_verified = found and extracted_bytes[:len(payload_bytes)] == payload_bytes

        # 8) save unified package
        model_path = os.path.join(config.output_dir, "model.pth")
        meta_path = os.path.join(config.output_dir, "meta.json")
        payload_ref_path = os.path.join(config.output_dir, "payload_ref.json")

        torch.save(model.state_dict(), model_path)

        save_json(
            {
                "payload_kind": "bytes",
                "payload_bytes": [int(b) for b in payload_bytes],
                "payload_prefix_hex": payload_bytes[:8].hex(),
            },
            payload_ref_path,
        )

        d_ppl = 1.0 if not math.isfinite(after_ppl) else abs(1.0 / base_ppl - 1.0 / after_ppl) / (1.0 / base_ppl)
        d_acc = abs(base_acc - after_acc) / (base_acc if base_acc > 0 else 1e-6)
        pai = max(d_ppl, d_acc)

        meta = {
            "attack_family": config.attack_family,
            "model_family": config.model_family,
            "mode": "robust",
            "group_type": "layer",
            "target_group": f"layer_{target_layer}",
            "q_bits": config.q_bits,
            "block_size": config.block_size,
            "repetition_n": n_star,
            "payload_len_bits": len(payload_bytes) * 8,
            "extract_verified": extract_verified,
            "before_ppl": base_ppl,
            "before_acc": base_acc,
            "after_ppl": after_ppl,
            "after_acc": after_acc,
            "pai": pai,
            "best_pai_search": best_pai,
            "search_records": search_records,
            "extra": config.extra,
        }
        save_json(meta, meta_path)

        return BuildResult(
            success=True,
            model_family="llm",
            output_dir=config.output_dir,
            model_path=model_path,
            meta_path=meta_path,
            payload_ref_path=payload_ref_path,
            target_group=f"layer_{target_layer}",
            q_bits=config.q_bits,
            block_size=config.block_size,
            repetition_n=n_star,
            payload_len_bits=len(payload_bytes) * 8,
            extract_verified=extract_verified,
            extra={
                "before_ppl": base_ppl,
                "before_acc": base_acc,
                "after_ppl": after_ppl,
                "after_acc": after_acc,
                "pai": pai,
            },
        )