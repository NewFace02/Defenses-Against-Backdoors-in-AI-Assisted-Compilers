import os
import torch

from Reproduction.CNN_Stego.utils import (
    clear_gpu,
    set_seed,
    get_device,
    load_dataloader,
    get_random_bits,
)
from Reproduction.CNN_Stego.src.model import ConvNet
from Reproduction.CNN_Stego.src.attack import (
    CNNStegoCore,
    verify_saved_stego_checkpoint,
)

CHECKPOINT_PATH = os.path.join("checkpoints", "clean_model_best.pth")
STEGO_CHECKPOINT_PATH = os.path.join("checkpoints", "cnn_stego_fc1_q8.pth")


def main():
    # 1) basic setup
    clear_gpu()
    set_seed(2026)
    device = get_device()

    print("=" * 80)
    print(f"Device: {device}")
    print("=" * 80)

    # 2) dataloader
    print("[1] Loading CIFAR10 dataloaders...")
    train_loader, valid_loader, test_loader = load_dataloader(
        train_batch=128,
        test_batch=256,
        is_shuffle=True,
    )

    # 3) model
    print("[2] Building ConvNet...")
    model = ConvNet(10).to(device)

    # 4) load clean checkpoint
    print("[3] Loading clean checkpoint...")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from: {CHECKPOINT_PATH}")

    # 5) core
    print("[4] Building CNNStegoCore...")
    core = CNNStegoCore(
        model=model,
        device=device,
        valid_loader=valid_loader,
        verbose=True,
    )

    # 6) print candidate params
    print("[5] Candidate parameter tensors:")
    core.print_candidate_params()

    # 7) evaluate clean model
    print("[6] Evaluating clean model on validation set...")
    metrics = core.evaluate_clean()

    print("=" * 80)
    print("Validation metrics")
    print("=" * 80)
    print(f"Loss    : {metrics['loss']:.6f}")
    print(f"Acc     : {metrics['acc']:.6f}")
    print(f"Correct : {metrics['correct']}")
    print(f"Total   : {metrics['total']}")
    print("=" * 80)

        # 8) safe perturbation smoke test
    print("[7] Running safe perturbation smoke test on fc1.weight ...")
    smoke = core.perturb_param_and_measure(
        param_name="fc1.weight",
        epsilon=1e-4,
        max_edit_elems=2048,
        dataloader=valid_loader,
    )

    print("=" * 80)
    print("Safe perturbation smoke test")
    print("=" * 80)
    print(f"Param name        : {smoke['param_name']}")
    print(f"Edited elements   : {smoke['edited_elements']}")
    print(f"Epsilon           : {smoke['epsilon']}")
    print("-" * 80)
    print(f"Before acc        : {smoke['before']['acc']:.6f}")
    print(f"After acc         : {smoke['after']['acc']:.6f}")
    print(f"Restored acc      : {smoke['restored']['acc']:.6f}")
    print(f"Acc drop after    : {smoke['acc_drop_after']:.6f}")
    print(f"Acc restore gap   : {smoke['acc_restore_gap']:.6f}")
    print(f"Loss restore gap  : {smoke['loss_restore_gap']:.6f}")
    print("=" * 80)

        # 9) capacity estimation
    print("[8] Estimating payload capacity on fc1.weight ...")
    cap_q8 = core.get_param_capacity_info(
        param_name="fc1.weight",
        q_bits=8,
        block_size=32,
        use_ecc=True,
        repetition_factor=5,
    )
    cap_q4 = core.get_param_capacity_info(
        param_name="fc1.weight",
        q_bits=4,
        block_size=32,
        use_ecc=True,
        repetition_factor=5,
    )

    print("=" * 80)
    print("Capacity estimation on fc1.weight")
    print("=" * 80)
    print(
        f"Q8  -> writable={cap_q8['writable_positions']}, "
        f"usable_bits={cap_q8['usable_data_bits']}, "
        f"usable_bytes={cap_q8['usable_data_bytes']}"
    )
    print(
        f"Q4  -> writable={cap_q4['writable_positions']}, "
        f"usable_bits={cap_q4['usable_data_bits']}, "
        f"usable_bytes={cap_q4['usable_data_bytes']}"
    )
    print("=" * 80)

    # 10) quantization roundtrip test
    print("[9] Running quantization roundtrip test on fc1.weight ...")
    rt_q8 = core.quantize_roundtrip_and_measure(
        param_name="fc1.weight",
        q_bits=8,
        block_size=32,
        dataloader=valid_loader,
    )
    rt_q4 = core.quantize_roundtrip_and_measure(
        param_name="fc1.weight",
        q_bits=4,
        block_size=32,
        dataloader=valid_loader,
    )

    print("=" * 80)
    print("Quantization roundtrip test on fc1.weight")
    print("=" * 80)
    print(
        f"Q8  -> mse={rt_q8['roundtrip_mse']:.8e}, "
        f"mae={rt_q8['roundtrip_mae']:.8e}, "
        f"acc_before={rt_q8['before']['acc']:.6f}, "
        f"acc_after={rt_q8['after']['acc']:.6f}, "
        f"acc_restore_gap={rt_q8['acc_restore_gap']:.6f}"
    )
    print(
        f"Q4  -> mse={rt_q4['roundtrip_mse']:.8e}, "
        f"mae={rt_q4['roundtrip_mae']:.8e}, "
        f"acc_before={rt_q4['before']['acc']:.6f}, "
        f"acc_after={rt_q4['after']['acc']:.6f}, "
        f"acc_restore_gap={rt_q4['acc_restore_gap']:.6f}"
    )
    print("=" * 80)

        # 11) first real stego test
    print("[10] Running first real stego test on fc1.weight (Q8) ...")
    payload_bits = get_random_bits(length=256, n=8)

    stego = core.embed_extract_and_measure(
        param_name="fc1.weight",
        payload_bits=payload_bits,
        q_bits=8,
        block_size=32,
        n=5,
        dataloader=valid_loader,
    )

    print("=" * 80)
    print("First real stego test on fc1.weight")
    print("=" * 80)
    print(f"Param name        : {stego['param_name']}")
    print(f"Q bits            : {stego['q_bits']}")
    print(f"Repetition n      : {stego['n']}")
    print(f"Before acc        : {stego['before']['acc']:.6f}")
    print(f"After acc         : {stego['after']['acc']:.6f}")
    print(f"Restored acc      : {stego['restored']['acc']:.6f}")
    print(f"Acc drop after    : {stego['acc_drop_after']:.6f}")
    print(f"Acc restore gap   : {stego['acc_restore_gap']:.6f}")
    print(f"Extracted ok      : {stego['extracted_ok']}")
    print(f"Bit errors        : {stego['bit_errors']}")
    print(f"Stream bits used  : {stego['embed_info']['stream_bits']}")
    print("=" * 80)

    # 12) scan best target
    print("[11] Scanning best target tensor for current payload (Q8) ...")
    scan_results = core.scan_best_target(
        payload_bits=payload_bits,
        q_bits=8,
        block_size=32,
        n=5,
        min_numel=1024,
        dataloader=valid_loader,
    )

    print("=" * 80)
    print("Best target scan results")
    print("=" * 80)
    for i, item in enumerate(scan_results):
        if item["status"] == "ok":
            print(
                f"[{i}] {item['param_name']:<15} "
                f"status={item['status']:<16} "
                f"usable_bits={item['usable_data_bits']:<8} "
                f"acc_shift_abs={item['acc_shift_abs']:.6f} "
                f"bit_errors={item['bit_errors']}"
            )
        else:
            print(
                f"[{i}] {item['param_name']:<15} "
                f"status={item['status']:<16} "
                f"usable_bits={item.get('usable_data_bits', -1):<8} "
                f"required_bits={item.get('required_bits', -1)}"
            )
    print("=" * 80)

    if len(scan_results) > 0 and scan_results[0]["status"] == "ok":
        best = scan_results[0]
        print("Current best target:")
        print(
            f"param={best['param_name']}, "
            f"acc_shift_abs={best['acc_shift_abs']:.6f}, "
            f"usable_bits={best['usable_data_bits']}, "
            f"bit_errors={best['bit_errors']}"
        )
    print("=" * 80)

        # 13) save one stego checkpoint
    print("[12] Saving one stego checkpoint on fc1.weight (Q8) ...")

    save_info = core.embed_and_save_checkpoint(
        save_path=STEGO_CHECKPOINT_PATH,
        param_name="fc1.weight",
        payload_bits=payload_bits,
        q_bits=8,
        block_size=32,
        n=5,
        extra_meta={"note": "first cnn stego checkpoint"},
    )

    print("=" * 80)
    print("Saved stego checkpoint")
    print("=" * 80)
    print(f"Save path         : {save_info['save_path']}")
    print(f"Param name        : {save_info['stego_meta']['param_name']}")
    print(f"Payload len bits  : {save_info['stego_meta']['payload_len_bits']}")
    print(f"Q bits            : {save_info['stego_meta']['q_bits']}")
    print(f"Block size        : {save_info['stego_meta']['block_size']}")
    print(f"Repetition n      : {save_info['stego_meta']['n']}")
    print("=" * 80)

        # 14) reload saved checkpoint and verify extraction
    print("[13] Reloading saved stego checkpoint and verifying extraction ...")
    verify_info = verify_saved_stego_checkpoint(
        checkpoint_path=STEGO_CHECKPOINT_PATH,
        device=device,
        valid_loader=valid_loader,
    )

    print("=" * 80)
    print("Reloaded stego checkpoint verification")
    print("=" * 80)
    print(f"Checkpoint path   : {verify_info['checkpoint_path']}")
    print(f"Param name        : {verify_info['stego_meta']['param_name']}")
    print(f"Q bits            : {verify_info['stego_meta']['q_bits']}")
    print(f"Block size        : {verify_info['stego_meta']['block_size']}")
    print(f"Repetition n      : {verify_info['stego_meta']['n']}")
    print(f"Extract ok        : {verify_info['extract_info']['ok']}")
    print(f"Payload ref found : {verify_info['payload_ref_found']}")
    print(f"Exact match       : {verify_info['exact_match']}")
    print(f"Bit errors        : {verify_info['bit_errors']}")
    if verify_info["metrics"] is not None:
        print(f"Reloaded acc      : {verify_info['metrics']['acc']:.6f}")
        print(f"Reloaded loss     : {verify_info['metrics']['loss']:.6f}")
    print("=" * 80)

if __name__ == "__main__":
    main()