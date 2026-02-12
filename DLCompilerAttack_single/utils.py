import os
import random
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import torchvision.transforms as transforms
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.attack import load_DLCL as _load_DLCL
from src import ConvNet

# =========================
# Paths / constants
# =========================
WORK_DIR = "work_dir"
LOG_DIR = "exp_logs"
GENERAL_DIR = "general_dir"
PREDICTION_RES_DIR = "prediction_res"
SPLIT_SYM = "::::"

for _d in [WORK_DIR, LOG_DIR, GENERAL_DIR, PREDICTION_RES_DIR]:
    os.makedirs(_d, exist_ok=True)

# CIFAR10 normalization used by the original project
_MEAN_ = [0.4802, 0.4481, 0.3975]
_STD_  = [0.2302, 0.2265, 0.2262]

# =========================
# Reproducibility
# =========================
def set_random_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =========================
# Backdoor trigger (match original logic)
# =========================
class ImgBackDoorTrigger:
    def __init__(self, trigger_size: int, trigger_pos: str, target_label: int, device: torch.device):
        self.trigger_size = trigger_size
        self.trigger_pos = trigger_pos
        self.target_label = target_label
        self.device = device

        # bounds in the *normalized* input space
        self.min_pixel = ((0 - torch.tensor(_MEAN_)) / torch.tensor(_STD_)).reshape([1, -1, 1, 1]).to(self.device)
        self.max_pixel = ((1 - torch.tensor(_MEAN_)) / torch.tensor(_STD_)).reshape([1, -1, 1, 1]).to(self.device)

        assert self.trigger_pos in ["left_up", "right_up", "left_down", "right_down"]

        # random init in [0, 1], then optimized in Stage 0
        t_v = torch.rand((1, 3, trigger_size, trigger_size), device=device)
        self.trigger = nn.Parameter(t_v)
        self.ori_trigger = self.trigger.clone()

    @torch.no_grad()
    def init_trigger(self, image_path: str):
        transform = T.Compose([
            T.Resize((self.trigger_size, self.trigger_size)),
            T.ToTensor(),
            T.Normalize(_MEAN_, _STD_),
        ])
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(self.trigger.device)
        self.trigger.copy_(img_tensor)
        self.ori_trigger = img_tensor.clone()

    def normalize_trigger(self):
        # Project trigger into an L∞ ball around the original trigger.
        max_linf = 0.2 / np.array(_STD_).mean()
        with torch.no_grad():
            perturb = self.trigger - self.ori_trigger
            perturb = torch.clamp(perturb, min=-max_linf, max=max_linf)
            self.trigger.copy_(self.ori_trigger + perturb)

    def add_trigger(self, x: torch.Tensor):
        triggered_data = x.clone().to(self.device)
        if self.trigger_pos == "left_up":
            triggered_data[:, :, :self.trigger_size, :self.trigger_size] = self.trigger
        elif self.trigger_pos == "right_down":
            triggered_data[:, :, :self.trigger_size, -self.trigger_size:] = self.trigger
        elif self.trigger_pos == "right_up":
            triggered_data[:, :, -self.trigger_size:, :self.trigger_size] = self.trigger
        elif self.trigger_pos == "left_down":
            triggered_data[:, :, -self.trigger_size:, -self.trigger_size:] = self.trigger
        else:
            raise NotImplementedError
        return triggered_data

    def get_trigger_area(self, x: torch.Tensor):
        if self.trigger_pos == "left_up":
            return x[:, :, :self.trigger_size, :self.trigger_size]
        elif self.trigger_pos == "right_down":
            return x[:, :, :self.trigger_size, -self.trigger_size:]
        elif self.trigger_pos == "right_up":
            return x[:, :, -self.trigger_size:, :self.trigger_size]
        elif self.trigger_pos == "left_down":
            return x[:, :, -self.trigger_size:, -self.trigger_size:]
        else:
            raise NotImplementedError

    def clamp_trigger(self):
        with torch.no_grad():
            self.trigger.clamp_(self.min_pixel, self.max_pixel)

    def __str__(self):
        return f"{self.trigger_size}____{self.trigger_pos}"

    def to(self, fp):
        self.trigger.to(fp)
        return self


def init_bd_trigger(trigger_size: int = 8, trigger_pos: str = "left_up", device: Optional[torch.device] = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ImgBackDoorTrigger(trigger_size, trigger_pos=trigger_pos, target_label=0, device=device)

# =========================
# Model loaders (task0 only)
# =========================
def load_model(load_pretrained: bool):
    model_data_name = f"convnet{SPLIT_SYM}cifar10"
    model = ConvNet(class_num=10)
    input_size = [[3, 32, 32]]

    if load_pretrained:
        weight_path = "checkpoints/clean_model_best.pth"
        if os.path.exists(weight_path):
            state = torch.load(weight_path, map_location="cpu")
            model.load_state_dict(state)
        else:
            raise FileNotFoundError(f"Pretrained weight not found: {weight_path}")

    model.input_sizes = input_size
    model.model_data_name = model_data_name
    model.input_types = ["float32"]
    model.fp = torch.float32
    return model


def load_attack_model_cls():
    from src.model.feature_model import ConvNetFeatureModel
    from src.model.tuned_model import ConvNetTunedModel
    embed_shape = [1, 32, 32, 32]  # FeatureModel output for ConvNet conv1+bn1
    return ConvNetFeatureModel, ConvNetTunedModel, embed_shape


def load_DLCL():
    # Only compiler 0 (torch.compile)
    return _load_DLCL()

# =========================
# Data loader (CIFAR10)
# =========================
class _Cifar10Dict(Dataset):
    def __init__(self, base_ds, transform):
        self.base = base_ds
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        img = self.transform(img)
        return {"input": img, "label": torch.tensor(y, dtype=torch.long)}


def load_dataloader(is_shuffle: bool, train_batch: int, test_batch: int):
    from torchvision.datasets import CIFAR10

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

    train_ds = _Cifar10Dict(train_ds, train_tf)
    test_ds = _Cifar10Dict(test_ds, test_tf)

    # Use a small fixed validation subset from the test set (same idea as original).
    rng = np.random.default_rng(66)
    idxs = rng.permutation(len(test_ds))[: (test_batch * 5)]
    valid_ds = torch.utils.data.Subset(test_ds, idxs.tolist())

    train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=is_shuffle, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=test_batch, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=test_batch, shuffle=False, num_workers=2)
    return train_loader, valid_loader, test_loader
