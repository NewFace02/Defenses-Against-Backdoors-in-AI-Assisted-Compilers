# -*- coding: utf-8 -*-
"""Train a clean (benign) CIFAR-10 ConvNet.

This repository is a *single-model / single-compiler* minimal reproduction.
Therefore, this script intentionally has:
  - no task
  - no compiler
  - no fp / precision flags

It trains the built-in CIFAR-10 ConvNet and saves the best checkpoint by test
accuracy.

Output:
  checkpoints/clean_model_best.pth

Run:
  python train_model_clean.py
"""

import os
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import load_dataloader, load_model, set_random_seed


CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "clean_model_best.pth")


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Support both dict-style and tuple-style dataloader batches."""
    if isinstance(batch, dict):
        return batch["input"], batch["label"]
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(f"Unsupported batch type/format: {type(batch)}")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    correct, total = 0, 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        x, y = _unpack_batch(batch)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()

    return correct / max(total, 1)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0

    for batch in dataloader:
        x, y = _unpack_batch(batch)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()

    return correct / max(total, 1)


def main() -> None:
    set_random_seed(3407)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_model_clean] device = {device}")

    # Hyperparams (kept simple + stable)
    num_epochs = 100
    train_batch = 100
    test_batch = 200

    # Single fixed dataloaders/model for this minimal repo
    # Support both (new) single-signature and (legacy) task-based signatures.
    try:
        train_loader, _valid_loader, test_loader = load_dataloader(
            is_shuffle=True,
            train_batch=train_batch,
            test_batch=test_batch,
        )
    except TypeError:
        # Legacy: load_dataloader(task, ...)
        train_loader, _valid_loader, test_loader = load_dataloader(
            is_shuffle=True,
            train_batch=train_batch,
            test_batch=test_batch,
        )

    
    model = load_model(load_pretrained=False).to(device)

    # Loss/opt/scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_acc = -1.0
    for epoch in range(1, num_epochs + 1):
        train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        # Evaluate periodically (and epoch 1)
        if epoch == 1 or epoch % 10 == 0:
            test_acc = evaluate(model, test_loader, device)
            print(f"Epoch [{epoch:3d}/{num_epochs}]  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"  -> new best: {best_acc:.4f}  saved: {CHECKPOINT_PATH}")
        else:
            print(f"Epoch [{epoch:3d}/{num_epochs}]  train_acc={train_acc:.4f}")

    print(f"Done. Best test_acc={best_acc:.4f}. Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
