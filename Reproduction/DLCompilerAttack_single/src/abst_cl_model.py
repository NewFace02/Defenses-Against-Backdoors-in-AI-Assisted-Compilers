import os
from typing import List, Optional
import torch
import torch.nn as nn

class TorchModel:
    """
    Minimal wrapper used by the attack pipeline.
    Only keeps what torch.compile needs + metadata used elsewhere.
    """
    def __init__(
        self,
        torch_model: nn.Module,
        batch_size: int,
        input_sizes: List[List[int]],
        input_types: List[str],
        output_num: int = 2,
        work_dir: str = ".",
        model_name: str = "model",
        target_device=None,
    ):
        self.torch_model = torch_model
        self.batch_size = batch_size
        self.input_sizes = input_sizes
        self.input_types = input_types
        self.output_num = output_num
        self.work_dir = work_dir
        self.model_name = model_name
        self.target_device = target_device

        # fp is expected by attack utils
        # input_types in this repo are strings like "float32"/"float16"
        it0 = (input_types[0] if input_types else "float32").lower()
        self.fp = torch.float16 if ("16" in it0 or "half" in it0) else torch.float32

class CompiledModel:
    """
    Minimal compiled-model interface used by collect_model_pred():
    - not an nn.Module
    - forward() accepts a list of tensors
    """
    def __init__(self, ori_model: TorchModel, compiled_model: nn.Module):
        self.ori_model = ori_model
        self.compiled_model = compiled_model
        self.fp = ori_model.fp

    def forward(self, input_lists: List[torch.Tensor]):
        raise NotImplementedError

class TorchCompiledModel(CompiledModel):
    def __init__(self, ori_model: TorchModel, compiled_model: nn.Module, device: torch.device):
        super().__init__(ori_model, compiled_model)
        self.device = device

    def forward(self, input_lists: List[torch.Tensor]):
        x = input_lists[0].to(self.device)
        return self.compiled_model(x)
