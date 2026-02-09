"""Train the clean (benign) model for the *single* simplified setting.

Fixed setting:
- task_id = 0  (CIFAR10 ConvNet)

We keep fp configurable (default fp32) because it does not change the attack logic.

Example:
  python simple/simple_train_clean.py --fp fp32
"""

import argparse

from train_model_clean import main as train_main
from utils import set_random_seed

TASK_ID = 0


def run(fp: str):
    set_random_seed(3407)
    train_main(task_id=TASK_ID, fp_name=fp)


def main():
    parser = argparse.ArgumentParser(description="Train clean model (task 0 only)")
    parser.add_argument("--fp", type=str, default="fp32", choices=["fp16", "fp32", "fp64"])
    args = parser.parse_args()
    run(args.fp)


if __name__ == "__main__":
    main()
