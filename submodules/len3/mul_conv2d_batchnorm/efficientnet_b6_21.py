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
        self.conv2d107 = Conv2d(864, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d63 = BatchNorm2d(144, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x333, x328):
        x334=operator.mul(x333, x328)
        x335=self.conv2d107(x334)
        x336=self.batchnorm2d63(x335)
        return x336

m = M().eval()
x333 = torch.randn(torch.Size([1, 864, 1, 1]))
x328 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x333, x328)
end = time.time()
print(end-start)
