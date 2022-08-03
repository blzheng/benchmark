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
        self.hardsigmoid5 = Hardsigmoid()
        self.conv2d50 = Conv2d(672, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(160, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d51 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x144, x140):
        x145=self.hardsigmoid5(x144)
        x146=operator.mul(x145, x140)
        x147=self.conv2d50(x146)
        x148=self.batchnorm2d38(x147)
        x149=self.conv2d51(x148)
        x150=self.batchnorm2d39(x149)
        return x150

m = M().eval()
x144 = torch.randn(torch.Size([1, 672, 1, 1]))
x140 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x144, x140)
end = time.time()
print(end-start)
