# Simple Dcl-BD demo (1 model + 1 compiler)

This folder provides a **minimal** reproduction of the original repository pipeline, but restricted to a single setting:

- **Model**: `task_id = 0`  (CIFAR10 ConvNet)
- **Compiler**: `cl_id = 0` (torch.compile)
- **Hardware**: `hardware_id = 0` by default (GPU if available; otherwise it automatically falls back to CPU)

The core code under `src/` is kept **unchanged**. We only add thin wrapper scripts that hard-code the IDs.

## 1) Train a clean model

```bash
python simple/simple_train_clean.py --fp fp32
```

This will save the best checkpoint under `model_weight/` (same as the original repo).

## 2) Run the attack

```bash
python simple/simple_attack.py --hardware_id 0
```

Artifacts are saved under `work_dir/<task_name>/` (same as the original repo). The wrapper ensures:
- `task_id=0`
- `cl_id=0`

## 3) Evaluate (before vs after compilation)

### Evaluate the clean model

```bash
python simple/simple_evaluate.py --approach clean --hardware_id 0
```

### Evaluate the attacked model

```bash
python simple/simple_evaluate.py --approach ours --hardware_id 0
```

You should observe the typical backdoor pattern:
- **Before compilation (D)**: clean accuracy stays high; trigger does *not* consistently force the target.
- **After compilation (C)**: trigger success rate becomes high (compiled model predicts the backdoor label).

---

## What is simplified?

Compared with the full repo:
- No loops over 6 tasks
- No loops over 3 compilers
- No detector / transferability / wild evaluation scripts

But the **attack core** is identical, since `src/attack/dlcl_attack.py` and the compiler abstraction are not modified.
