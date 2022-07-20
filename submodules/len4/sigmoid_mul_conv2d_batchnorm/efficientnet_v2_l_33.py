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
        self.sigmoid33 = Sigmoid()
        self.conv2d202 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d134 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x649, x645):
        x650=self.sigmoid33(x649)
        x651=operator.mul(x650, x645)
        x652=self.conv2d202(x651)
        x653=self.batchnorm2d134(x652)
        return x653

m = M().eval()
x649 = torch.randn(torch.Size([1, 2304, 1, 1]))
x645 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x649, x645)
end = time.time()
print(end-start)
