import torch
import torch.nn as nn
from copy import deepcopy
from .convnet import ConvNet
from .utils import BiasLayer

class ChannelWiseThresholdActivation(nn.Module):
    def __init__(self, threshold: torch.Tensor):
        super().__init__()
        self.register_buffer("threshold", threshold)

    def forward(self, x):
        # x: (N, C, H, W), threshold: (C,)
        thr = self.threshold.view(1, -1, 1, 1)
        return torch.where(x > thr, x, torch.zeros_like(x))

class MyActivation(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        zero_shape = [0 for _ in range(input_shape[1])]
        self.act = ChannelWiseThresholdActivation(torch.tensor(zero_shape))

    def init_activation(self, threshold):
        self.act = ChannelWiseThresholdActivation(threshold)

    def forward(self, x):
        return self.act(x)

    def to(self, device):
        self.act.to(device)
        super().to(device)
        return self

class AbstractTunedModel(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_bias = BiasLayer(input_shape)
        self.input_index = None
        self.input_shape = input_shape
        self.scale = 1.0

    def set_scale(self, scale):
        self.scale = scale.reshape([1, -1, 1, 1])

    def set_input_V(self, V, M):
        self.input_bias.set_v(V, M)
        self.input_index = self.input_bias.index

    def to(self, device):
        self.input_bias.to(device)
        super().to(device)
        return self

class ConvNetTunedModel(AbstractTunedModel):
    def __init__(self, model: ConvNet, input_shape):
        super().__init__(input_shape)
        self.conv2 = deepcopy(model.conv2)
        self.bn2 = deepcopy(model.bn2)
        self.conv3 = deepcopy(model.conv3)
        self.pool = deepcopy(model.pool)
        self.fc1 = deepcopy(model.fc1)
        self.fc2 = deepcopy(model.fc2)
        self.fc3 = deepcopy(model.fc3)
        self.fp = model.fp

    def forward(self, x):
        x = self.pool(x)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
