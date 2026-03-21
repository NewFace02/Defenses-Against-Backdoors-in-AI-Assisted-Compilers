import os
from typing import Optional, List, Dict, Any

import torch

from Reproduction.StegoUnified.utils import (
    get_device,
    load_dataloader,
    get_random_bits,
)
from Reproduction.StegoUnified.src.common import BuildConfig, BuildResult
from Reproduction.StegoUnified.src.common.io import (
    ensure_dir,
    save_json,
    save_payload_bits,
)
from Reproduction.StegoUnified.src.models import ConvNet
from Reproduction.StegoUnified.src.attack import CNNStegoCore
from Reproduction.StegoUnified.src.backends.base_builder import BaseBuilder


class CNNBuilder(BaseBuilder):
    def __init__(self, device: Optional[torch.device] = None, valid_loader=None):
        self.device = device if device is not None else get_device()
        self.valid_loader = valid_loader

    def _get_valid_loader(self):
        if self.valid_loader is None:
            _, valid_loader, _ = load_dataloader(
                train_batch=128,
                test_batch=256,
                is_shuffle=True,
            )
            self.valid_loader = valid_loader
        return self.valid_loader

    def _load_clean_model(self, clean_model_path: str) -> ConvNet:
        model = ConvNet(10).to(self.device)

        ckpt = torch.load(clean_model_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict)
        return model

    def _choose_target_group(
        self,
        core: CNNStegoCore,
        config: BuildConfig,
        payload_bits: List[int],
        valid_loader,
    ):
        if config.target_group is not None and str(config.target_group).strip() != "":
            return config.target_group, None

        scan_results = core.scan_best_target(
            payload_bits=payload_bits,
            q_bits=config.q_bits,
            block_size=config.block_size,
            n=config.repetition_n,
            dataloader=valid_loader,
        )

        ok_items = [x for x in scan_results if x["status"] == "ok"]
        if len(ok_items) == 0:
            raise RuntimeError("No valid CNN target group found for current payload/config.")

        return ok_items[0]["param_name"], scan_results

    def build(self, config: BuildConfig) -> BuildResult:
        if config.model_family != "cnn":
            raise ValueError(f"CNNBuilder only supports model_family='cnn', got: {config.model_family}")

        if not config.clean_model_path:
            raise ValueError("clean_model_path is required.")

        if not config.output_dir:
            raise ValueError("output_dir is required.")

        ensure_dir(config.output_dir)

        valid_loader = self._get_valid_loader()

        # 1) load clean model
        model = self._load_clean_model(config.clean_model_path)
        core = CNNStegoCore(
            model=model,
            device=self.device,
            valid_loader=valid_loader,
            verbose=False,
        )

        # 2) generate payload
        payload_bits = get_random_bits(length=config.payload_len_bits, n=8)

        # 3) choose target group
        target_group, scan_results = self._choose_target_group(
            core=core,
            config=config,
            payload_bits=payload_bits,
            valid_loader=valid_loader,
        )

        # 4) evaluate clean
        before_metrics = core.evaluate_clean(valid_loader)

        # 5) embed
        embed_info = core.embed_payload_in_param(
            param_name=target_group,
            payload_bits=payload_bits,
            q_bits=config.q_bits,
            block_size=config.block_size,
            n=config.repetition_n,
        )

        # 6) evaluate poisoned model
        after_metrics = core.evaluate_clean(valid_loader)

        # 7) immediate extraction check
        extract_info = core.extract_payload_from_param(
            param_name=target_group,
            q_bits=config.q_bits,
            block_size=config.block_size,
        )

        extracted_ok = False
        bit_errors = None
        if extract_info["ok"]:
            ext_bits = extract_info["payload_bits"]
            bit_errors = sum(int(a != b) for a, b in zip(payload_bits, ext_bits))
            bit_errors += abs(len(payload_bits) - len(ext_bits))
            extracted_ok = (bit_errors == 0 and len(ext_bits) == len(payload_bits))

        # 8) save unified package
        model_path = os.path.join(config.output_dir, "model.pth")
        meta_path = os.path.join(config.output_dir, "meta.json")
        payload_ref_path = os.path.join(config.output_dir, "payload_ref.json")

        torch.save(model.state_dict(), model_path)
        save_payload_bits(payload_bits, payload_ref_path)

        meta = {
            "attack_family": config.attack_family,
            "model_family": config.model_family,
            "mode": config.mode,
            "group_type": config.group_type,
            "target_group": target_group,
            "q_bits": config.q_bits,
            "block_size": config.block_size,
            "repetition_n": config.repetition_n,
            "payload_len_bits": len(payload_bits),
            "extract_verified": extracted_ok,
            "before_acc": before_metrics["acc"],
            "after_acc": after_metrics["acc"],
            "acc_shift_abs": abs(after_metrics["acc"] - before_metrics["acc"]),
            "bit_errors": bit_errors,
            "embed_info": embed_info,
            "scan_results": scan_results,
            "extra": config.extra,
        }
        save_json(meta, meta_path)

        # 9) reload and verify package
        reload_model = ConvNet(10).to(self.device)
        reload_state = torch.load(model_path, map_location=self.device)
        reload_model.load_state_dict(reload_state)

        reload_core = CNNStegoCore(
            model=reload_model,
            device=self.device,
            valid_loader=valid_loader,
            verbose=False,
        )

        reload_extract = reload_core.extract_payload_from_param(
            param_name=target_group,
            q_bits=config.q_bits,
            block_size=config.block_size,
        )

        reload_exact = False
        if reload_extract["ok"]:
            ext_bits = reload_extract["payload_bits"]
            reload_exact = (len(ext_bits) == len(payload_bits)) and all(
                int(a) == int(b) for a, b in zip(payload_bits, ext_bits)
            )

        return BuildResult(
            success=True,
            model_family="cnn",
            output_dir=config.output_dir,
            model_path=model_path,
            meta_path=meta_path,
            payload_ref_path=payload_ref_path,
            target_group=target_group,
            q_bits=config.q_bits,
            block_size=config.block_size,
            repetition_n=config.repetition_n,
            payload_len_bits=len(payload_bits),
            extract_verified=reload_exact,
            extra={
                "before_acc": before_metrics["acc"],
                "after_acc": after_metrics["acc"],
                "acc_shift_abs": abs(after_metrics["acc"] - before_metrics["acc"]),
                "bit_errors": bit_errors,
            },
        )