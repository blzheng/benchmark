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
        self.conv2d159 = Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d105 = BatchNorm2d(1824, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x509, x494):
        x510=operator.add(x509, x494)
        x511=self.conv2d159(x510)
        x512=self.batchnorm2d105(x511)
        return x512

m = M().eval()
x509 = torch.randn(torch.Size([1, 304, 7, 7]))
x494 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x509, x494)
end = time.time()
print(end-start)
