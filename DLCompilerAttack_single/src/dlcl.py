import torch
from .abst_cl_model import TorchModel, TorchCompiledModel

class TargetDevice:
    """
    device_id: 0 => cuda:0, -1 => cpu
    """
    def __init__(self, device_id: int):
        self.device_id = device_id
        if device_id == -1 or not torch.cuda.is_available():
            self.torch_target = torch.device("cpu")
        else:
            self.torch_target = torch.device(f"cuda:{device_id}")

    def __str__(self):
        return "gpu" if self.device_id == 0 else "cpu"


class DLCompiler:
    """
    Minimal compiler interface: only torch.compile is kept.
    """
    def __init__(self, work_dir=None):
        self.work_dir = work_dir or "."

    def torch_compile(self, model: TorchModel):
        target_dev = model.target_device.torch_target
        compiled_model = torch.compile(model.torch_model).to(target_dev).eval()
        return TorchCompiledModel(model, compiled_model, target_dev)
