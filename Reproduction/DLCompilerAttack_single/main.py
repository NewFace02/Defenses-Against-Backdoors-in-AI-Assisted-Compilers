import os
import argparse
import torch

from utils import load_DLCL, load_attack_model_cls
from utils import load_model, load_dataloader, init_bd_trigger
from utils import SPLIT_SYM, WORK_DIR, GENERAL_DIR

from src.attack.dlcl_attack import DLCompilerAttack
from src import TargetDevice


def main(hardware_id: int):
    use_cuda = (hardware_id == 0) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[Device] hardware_id={hardware_id}, cuda_available={torch.cuda.is_available()} -> {device}")

    # Keep the original trigger parametrization (size=8, left-up corner).
    bd_trigger = init_bd_trigger(trigger_size=8, trigger_pos="left_up", device=device)

    batch_size = 100
    cl_func = load_DLCL()
    hardware_target = TargetDevice(hardware_id)

    model = load_model(load_pretrained=True)

    # Avoid numeric IDs in the visible experiment name (but keep deterministic structure).
    task_name = model.model_data_name + SPLIT_SYM + "torchcompile" + SPLIT_SYM + str(hardware_target)
    work_dir = os.path.join(WORK_DIR, task_name)
    os.makedirs(work_dir, exist_ok=True)

    FeatureModel, TunedModel, embed_shape = load_attack_model_cls()
    train_loader, _valid_loader, test_loader = load_dataloader(False, batch_size, batch_size)

    D = FeatureModel(model)
    tuned_model = TunedModel(model, embed_shape)

    print(task_name)

    attack_config = {
        "trigger_opt_epoch": 10,
        "trigger_opt_lr": 1e-2,
        "finetune_cl_epoch": 10,
        "finetune_cl_lr": 1e-2,
        "finetune_epoch": 50,
        "finetune_lr": 1e-4,
        "save_freq": 10,
        "task_name": task_name,
        "work_dir": work_dir,
        "batch_size": batch_size,
        "general_dir": GENERAL_DIR,
        "bd_rate": 1.0,
    }

    attacker = DLCompilerAttack(
        D, tuned_model,
        train_loader, test_loader, bd_trigger,
        cl_func, hardware_target, device, attack_config
    )
    attacker.run_attack(attack_stage_list={0, 1, 2})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware_id", type=int, default=0, help="0: GPU (if available), -1: CPU")
    args = parser.parse_args()
    assert args.hardware_id in [-1, 0]
    main(hardware_id=args.hardware_id)
