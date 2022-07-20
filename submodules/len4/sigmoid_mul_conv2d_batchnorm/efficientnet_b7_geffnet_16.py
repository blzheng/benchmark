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
        self.conv2d81 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x242, x238):
        x243=x242.sigmoid()
        x244=operator.mul(x238, x243)
        x245=self.conv2d81(x244)
        x246=self.batchnorm2d47(x245)
        return x246

m = M().eval()
x242 = torch.randn(torch.Size([1, 480, 1, 1]))
x238 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x242, x238)
end = time.time()
print(end-start)
