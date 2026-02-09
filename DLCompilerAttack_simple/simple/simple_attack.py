"""Run the Dcl-BD attack for the *single* simplified setting.

Fixed setting:
- task_id = 0  (CIFAR10 ConvNet)
- cl_id   = 0  (torch.compile)

Hardware:
- hardware_id = 0 by default (GPU if available, otherwise it will fall back to CPU)

Example:
  python simple/simple_attack.py --hardware_id 0
"""

import argparse

from main import main as attack_main

TASK_ID = 0
CL_ID = 0


def run(hardware_id: int):
    attack_main(task_id=TASK_ID, cl_id=CL_ID, hardware_id=hardware_id)


def main():
    parser = argparse.ArgumentParser(description="Run attack (task 0 + cl 0 only)")
    parser.add_argument("--hardware_id", type=int, default=0, choices=[-1, 0], help="-1: CPU, 0: GPU")
    args = parser.parse_args()
    run(args.hardware_id)


if __name__ == "__main__":
    main()
