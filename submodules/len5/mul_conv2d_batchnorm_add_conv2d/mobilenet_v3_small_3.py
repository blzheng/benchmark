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
        self.conv2d50 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(96, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d51 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x144, x139, x133):
        x145=operator.mul(x144, x139)
        x146=self.conv2d50(x145)
        x147=self.batchnorm2d32(x146)
        x148=operator.add(x147, x133)
        x149=self.conv2d51(x148)
        return x149

m = M().eval()
x144 = torch.randn(torch.Size([1, 576, 1, 1]))
x139 = torch.randn(torch.Size([1, 576, 7, 7]))
x133 = torch.randn(torch.Size([1, 96, 7, 7]))
start = time.time()
output = m(x144, x139, x133)
end = time.time()
print(end-start)
