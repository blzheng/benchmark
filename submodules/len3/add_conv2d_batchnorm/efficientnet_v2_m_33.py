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
        self.conv2d154 = Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d102 = BatchNorm2d(1824, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x493, x478):
        x494=operator.add(x493, x478)
        x495=self.conv2d154(x494)
        x496=self.batchnorm2d102(x495)
        return x496

m = M().eval()
x493 = torch.randn(torch.Size([1, 304, 7, 7]))
x478 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x493, x478)
end = time.time()
print(end-start)
