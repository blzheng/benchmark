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
        self.batchnorm2d91 = BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d159 = Conv2d(448, 1792, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d95 = BatchNorm2d(1792, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x478, x494):
        x479=self.batchnorm2d91(x478)
        x495=operator.add(x494, x479)
        x496=self.conv2d159(x495)
        x497=self.batchnorm2d95(x496)
        return x497

m = M().eval()
x478 = torch.randn(torch.Size([1, 448, 7, 7]))
x494 = torch.randn(torch.Size([1, 448, 7, 7]))
start = time.time()
output = m(x478, x494)
end = time.time()
print(end-start)
