import torch

from Reproduction.CNN_Stego.src.model import ConvNet
from Reproduction.CNN_Stego.src.attack.cnn_stego_core import CNNStegoCore


def load_cnn_stego_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model = ConvNet(10).to(device)
    model.load_state_dict(state_dict)

    stego_meta = ckpt.get("stego_meta", {})
    payload_bits_ref = ckpt.get("payload_bits_ref", None)

    return model, stego_meta, payload_bits_ref, ckpt


def verify_saved_stego_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    valid_loader=None,
):
    model, stego_meta, payload_bits_ref, ckpt = load_cnn_stego_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    core = CNNStegoCore(
        model=model,
        device=device,
        valid_loader=valid_loader,
        verbose=False,
    )

    metrics = None
    if valid_loader is not None:
        metrics = core.evaluate_clean(valid_loader)

    extract_info = core.extract_payload_from_param(
        param_name=stego_meta["param_name"],
        q_bits=stego_meta["q_bits"],
        block_size=stego_meta["block_size"],
    )

    exact_match = None
    bit_errors = None

    if extract_info["ok"] and payload_bits_ref is not None:
        ref_bits = [int(b) for b in payload_bits_ref]
        ext_bits = [int(b) for b in extract_info["payload_bits"]]

        if len(ref_bits) == len(ext_bits):
            bit_errors = sum(int(a != b) for a, b in zip(ref_bits, ext_bits))
            exact_match = (bit_errors == 0)
        else:
            bit_errors = abs(len(ref_bits) - len(ext_bits)) + sum(
                int(a != b) for a, b in zip(ref_bits[:min(len(ref_bits), len(ext_bits))],
                                            ext_bits[:min(len(ref_bits), len(ext_bits))])
            )
            exact_match = False

    return {
        "checkpoint_path": checkpoint_path,
        "stego_meta": stego_meta,
        "metrics": metrics,
        "extract_info": extract_info,
        "payload_bits_ref": payload_bits_ref,
        "payload_ref_found": payload_bits_ref is not None,
        "exact_match": exact_match,
        "bit_errors": bit_errors,
    }