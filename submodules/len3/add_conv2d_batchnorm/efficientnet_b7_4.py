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
        self.conv2d32 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x98, x83):
        x99=operator.add(x98, x83)
        x100=self.conv2d32(x99)
        x101=self.batchnorm2d18(x100)
        return x101

m = M().eval()
x98 = torch.randn(torch.Size([1, 48, 56, 56]))
x83 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x98, x83)
end = time.time()
print(end-start)
