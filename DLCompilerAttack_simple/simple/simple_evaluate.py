"""Evaluate the simplified setting before/after compilation.

We report the same 5 metrics used in the original code:
- D, clean data, clean target
- D, triggered data, clean target
- D, triggered data, backdoor target
- C, clean data, clean target
- C, triggered data, backdoor target

Where:
- D = original PyTorch model
- C = compiled model (torch.compile)

Fixed setting:
- task_id = 0 (CIFAR10 ConvNet)
- cl_id   = 0 (torch.compile)

Example:
  python simple/simple_evaluate.py --approach clean
  python simple/simple_evaluate.py --approach ours
"""

import argparse
import os
import torch

from utils import load_model, load_dataloader, init_bd_trigger, load_DLCL
from utils import SPLIT_SYM, WORK_DIR
from src import TargetDevice
from src.attack.utils import CLSetting, evaluate_model, log_epoch

TASK_ID = 0
CL_ID = 0
BATCH_SIZE = 100


def load_ours_artifact(task_name: str):
    """Load attacked model saved by DLCompilerAttack.

    It searches for work_dir/<task_name>/(best.tar or periodic *.tar).
    Returns: (bd_trigger, attacked_model)
    """
    save_dir = os.path.join(WORK_DIR, task_name)
    possible_list = [str(5 * i + 4) + '.tar' for i in range(20)]
    possible_list = list(reversed(possible_list))
    possible_list = ["best.tar"] + possible_list
    for fname in possible_list:
        fpath = os.path.join(save_dir, fname)
        if os.path.exists(fpath):
            bd_trigger, model, _ = torch.load(fpath, map_location=torch.device('cpu'), weights_only=False)
            # In this repo, the attacked model is often wrapped and needs init()
            if hasattr(model, 'init'):
                model.init()
            return bd_trigger, model
    raise FileNotFoundError(
        f"Could not find attacked artifact under {save_dir}. "
        f"Run: python simple/simple_attack.py first."
    )


def build_task_name(model_data_name: str, hardware_id: int) -> str:
    hardware_target = TargetDevice(hardware_id)
    return model_data_name + SPLIT_SYM + f"CL___{CL_ID}" + SPLIT_SYM + str(hardware_target)


def main():
    parser = argparse.ArgumentParser(description="Evaluate simplified setting")
    parser.add_argument('--approach', type=str, default='clean', choices=['clean', 'ours'])
    parser.add_argument('--hardware_id', type=int, default=0, choices=[-1, 0])
    args = parser.parse_args()

    # Device for evaluation
    use_cuda = (args.hardware_id == 0) and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Data
    train_loader, valid_loader, test_loader = load_dataloader(TASK_ID, False, BATCH_SIZE, BATCH_SIZE)

    # Trigger (for clean approach we still need a trigger to measure the backdoor success metric)
    bd_trigger = init_bd_trigger(8, 'left_up', device)

    # Model
    base_model = load_model(TASK_ID, load_pretrained=True)
    model_data_name = base_model.model_data_name
    task_name = build_task_name(model_data_name, args.hardware_id)

    if args.approach == 'clean':
        model = base_model
    else:
        loaded_trigger, model = load_ours_artifact(task_name)
        bd_trigger = loaded_trigger
        bd_trigger.trigger = bd_trigger.trigger.to(device)

    # Compile config
    cl_func = load_DLCL(CL_ID)
    cl_setting = CLSetting.from_config({
        'batch_size': BATCH_SIZE,
        'input_sizes': model.input_sizes,
        'input_types': ['float32'],
        'work_dir': os.path.join('simple_work_dir'),
        'hardware_target': TargetDevice(args.hardware_id),
        'cl_func': cl_func,
        'fp': 'fp32',
        'device': device,
    })

    metrics = evaluate_model(model, cl_setting, test_loader, bd_trigger)
    log_epoch(f"[{args.approach}] {task_name}", test_acc=metrics)


if __name__ == '__main__':
    main()
