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
        self.sigmoid31 = Sigmoid()
        self.conv2d183 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d119 = BatchNorm2d(304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x584, x580):
        x585=self.sigmoid31(x584)
        x586=operator.mul(x585, x580)
        x587=self.conv2d183(x586)
        x588=self.batchnorm2d119(x587)
        return x588

m = M().eval()
x584 = torch.randn(torch.Size([1, 1824, 1, 1]))
x580 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x584, x580)
end = time.time()
print(end-start)
