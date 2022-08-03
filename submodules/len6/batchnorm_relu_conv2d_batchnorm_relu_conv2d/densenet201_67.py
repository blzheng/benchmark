import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.batchnorm2d138 = BatchNorm2d(928, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu138 = ReLU(inplace=True)
        self.conv2d138 = Conv2d(928, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d139 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu139 = ReLU(inplace=True)
        self.conv2d139 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x489):
        x490=self.batchnorm2d138(x489)
        x491=self.relu138(x490)
        x492=self.conv2d138(x491)
        x493=self.batchnorm2d139(x492)
        x494=self.relu139(x493)
        x495=self.conv2d139(x494)
        return x495

m = M().eval()
x489 = torch.randn(torch.Size([1, 928, 7, 7]))
start = time.time()
output = m(x489)
end = time.time()
print(end-start)
