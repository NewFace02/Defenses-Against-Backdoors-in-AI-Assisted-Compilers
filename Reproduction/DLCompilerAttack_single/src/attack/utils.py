import torch.nn as nn
import torch
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader
from ..dlcl import DLCompiler
from src.dlcl import TargetDevice
from src.model import MyModel, AbstFeatureModel, AbstractTunedModel
from src.model.tuned_model import MyActivation
from src.abst_cl_model import TorchModel
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

@torch.no_grad()
def collect_model_pred(model, dataloader, device, bd_trigger, return_index=0, return_label=False):
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader):
        data = batch['input'].to(device)
        label = batch['label']
        if bd_trigger is not None:
            data = bd_trigger.add_trigger(data)
        if isinstance(model, nn.Module):
            pred = model.forward(data.to(model.fp))
        else:
            pred = model.forward([data.to(model.fp)])

        if isinstance(pred, torch.Tensor):
            all_preds.append(pred.detach().cpu())
        elif isinstance(pred, List):
            all_preds.append(pred[return_index].detach().cpu())
        all_labels.append(label.detach().cpu())

    if return_label:
        return torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0)
    else:
        return torch.cat(all_preds, dim=0)


def resample_data_loader(dataloader, percent=0.2, seed=66):
    """
    Resample a subset from a PyTorch DataLoader.
    Works for standard torch Dataset (no .shuffle/.select).
    """
    dataset = dataloader.dataset
    n = len(dataset)
    new_num = max(1, int(math.floor(n * percent)))

    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=new_num, replace=False).tolist()

    new_dataset = Subset(dataset, indices)

    # Preserve original loader settings as much as possible
    return DataLoader(
        new_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,  # indices already randomized
        num_workers=getattr(dataloader, "num_workers", 0),
        pin_memory=getattr(dataloader, "pin_memory", False),
        drop_last=getattr(dataloader, "drop_last", False),
        collate_fn=getattr(dataloader, "collate_fn", None),
    )


def load_DLCL():
    dlcl = DLCompiler(None)
    return dlcl.torch_compile



def measure_acc(model, test_loader, bd_trigger, device, bd_target):
    logits, y = collect_model_pred(model, test_loader, device, bd_trigger, return_label=True)
    preds = logits.argmax(dim=1)
    if bd_target:
        y = torch.full_like(y, bd_trigger.target_label)
    return preds.eq(y).sum() / len(y)

from dataclasses import dataclass

@dataclass
class CLSetting:
    batch_size: int
    input_sizes: any
    input_types: any
    work_dir: str
    hardware_target: any
    cl_func: any
    fp: str
    device: torch.device

    @classmethod
    def from_config(cls, config: dict):
        """Create a CLSetting instance from a config dictionary."""
        return cls(
            batch_size=config["batch_size"],
            input_sizes=config["input_sizes"],
            input_types=config["input_types"],
            work_dir=config["work_dir"],
            hardware_target=config["hardware_target"],
            cl_func=config["cl_func"],
            fp = config["fp"],
            device = config["device"],
        )

    def _compile_model(self, model, model_name="wrapped_model"):
        """Helper: wrap a PyTorch model into TorchModel + compile."""
        abst_model = TorchModel(
            model,
            self.batch_size,
            self.input_sizes,
            self.input_types,
            output_num=2,
            work_dir=self.work_dir,
            model_name=model_name,
            target_device=self.hardware_target,
        )
        return self.cl_func(abst_model)


@torch.no_grad()
def evaluate_model(model: nn.Module, cl_setting: CLSetting, data_loader, bd_trigger):
    device = cl_setting.device
    cl_model = cl_setting._compile_model(model)
    model = model.to(device).eval()
    acc_cl_D_cl = measure_acc(model, data_loader, None, device, False)
    acc_cl_C_cl = measure_acc(cl_model, data_loader, None, device, False)
    acc_bd_D_cl = measure_acc(model, data_loader, bd_trigger, device, False)
    acc_bd_D_bd = measure_acc(model, data_loader, bd_trigger, device, True)
    acc_bd_C_bd = measure_acc(cl_model, data_loader, bd_trigger, device, True)

    return acc_cl_D_cl, acc_cl_C_cl, acc_bd_D_cl, acc_bd_D_bd, acc_bd_C_bd


def compile_ensemble_model(D, act, tuned_model, cl_setting: CLSetting):
    my_model = MyModel(D, act, tuned_model)

    abst_my_model = TorchModel(
        my_model,
        cl_setting.batch_size,
        cl_setting.input_sizes,
        cl_setting.input_types,
        2,
        cl_setting.work_dir,
        model_name='abst_my_model',
        target_device=cl_setting.hardware_target,
    )
    compiled_my_model = cl_setting.cl_func(abst_my_model)
    return compiled_my_model


def log_epoch(init_string, losses=None, train_acc=None, test_acc=None):
    """Log detailed metrics for the epoch, handling None values gracefully."""

    # Helper to format values safely
    def fmt(val, precision=5):
        return f"{val:.{precision}f}" if val is not None else "N/A"

    # Unpack with safe defaults
    if losses is not None:
        (
            loss_cl_D_cl,
            loss_cl_C_cl,
            loss_bd_D_cl,
            loss_bd_D_bd,
            loss_bd_C_bd,
            _,
        ) = losses
    else:
        loss_cl_D_cl = loss_cl_C_cl = loss_bd_D_cl = loss_bd_D_bd = loss_bd_C_bd = None

    if train_acc is not None:
        (
            acc_cl_D_cl,
            acc_cl_C_cl,
            acc_bd_D_cl,
            acc_bd_D_bd,
            acc_bd_C_bd,
        ) = train_acc
    else:
        acc_cl_D_cl = acc_cl_C_cl = acc_bd_D_cl = acc_bd_D_bd = acc_bd_C_bd = None

    if test_acc is not None:
        test_cl_D_cl, test_cl_C_cl, test_bd_D_cl, test_bd_D_bd, test_bd_C_bd = test_acc
    else:
        test_cl_D_cl = test_cl_C_cl = test_bd_D_cl = test_bd_D_bd = test_bd_C_bd = None

    print(f"{init_string}")
    print(
        f"D, clean data, clean target: loss {fmt(loss_cl_D_cl, 2)}, "
        f"train acc {fmt(acc_cl_D_cl)}, test acc {fmt(test_cl_D_cl)}"
    )
    print(
        f"D, triggered data, clean target: loss {fmt(loss_bd_D_cl, 2)}, "
        f"train acc {fmt(acc_bd_D_cl)}, test acc {fmt(test_bd_D_cl)}"
    )
    print(
        f"D, triggered data, backdoor label: loss {fmt(loss_bd_D_bd, 2)}, "
        f"train acc {fmt(acc_bd_D_bd)}, test acc {fmt(test_bd_D_bd)}"
    )
    print(
        f"C, cleaned data, clean target: loss {fmt(loss_cl_C_cl, 2)}, "
        f"train acc {fmt(acc_cl_C_cl)}, test acc {fmt(test_cl_C_cl)}"
    )
    print(
        f"C, triggered data, backdoor label: loss {fmt(loss_bd_C_bd, 2)}, "
        f"train acc {fmt(acc_bd_C_bd)}, test acc {fmt(test_bd_C_bd)}"
    )
