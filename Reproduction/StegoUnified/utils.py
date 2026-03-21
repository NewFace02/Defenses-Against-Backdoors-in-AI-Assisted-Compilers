import gc
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10


_MEAN_ = [0.4802, 0.4481, 0.3975]
_STD_ = [0.2302, 0.2265, 0.2262]


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def set_seed(seed=2026, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_random_bits(length, n=8):
    rand = np.random.randint(0, 2, size=length).tolist()
    pad = (n - (len(rand) % n)) % n
    return rand + [0] * pad


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Cifar10Dict(Dataset):
    def __init__(self, base_ds, transform):
        self.base = base_ds
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        img = self.transform(img)
        return {
            "input": img,
            "label": torch.tensor(y, dtype=torch.long)
        }


def load_dataloader(train_batch=128, test_batch=256, is_shuffle=True):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN_, _STD_),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN_, _STD_),
    ])

    train_ds = CIFAR10(root="data", train=True, download=True)
    test_ds = CIFAR10(root="data", train=False, download=True)

    train_ds = Cifar10Dict(train_ds, train_tf)
    test_ds = Cifar10Dict(test_ds, test_tf)

    rng = np.random.default_rng(66)
    idxs = rng.permutation(len(test_ds))[:(test_batch * 5)]
    valid_ds = torch.utils.data.Subset(test_ds, idxs.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch,
        shuffle=is_shuffle,
        num_workers=2
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=test_batch,
        shuffle=False,
        num_workers=2
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch,
        shuffle=False,
        num_workers=2
    )

    return train_loader, valid_loader, test_loader


@torch.no_grad()
def evaluate_cnn(model, dataloader, device):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for batch in dataloader:
        x = batch["input"].to(device)
        y = batch["label"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (preds == y).sum().item()
        total_num += bs

    avg_loss = total_loss / max(total_num, 1)
    acc = total_correct / max(total_num, 1)

    return {
        "loss": avg_loss,
        "acc": acc,
        "correct": total_correct,
        "total": total_num
    }