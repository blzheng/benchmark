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
        self.conv2d66 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d66 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x216, x210):
        x217=self.conv2d66(x216)
        x218=self.batchnorm2d66(x217)
        x219=operator.add(x218, x210)
        return x219

m = M().eval()
x216 = torch.randn(torch.Size([1, 1024, 14, 14]))
x210 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x216, x210)
end = time.time()
print(end-start)
