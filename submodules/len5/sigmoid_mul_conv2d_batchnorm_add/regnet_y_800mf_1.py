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
        self.sigmoid1 = Sigmoid()
        self.conv2d12 = Conv2d(144, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x33, x29, x23):
        x34=self.sigmoid1(x33)
        x35=operator.mul(x34, x29)
        x36=self.conv2d12(x35)
        x37=self.batchnorm2d8(x36)
        x38=operator.add(x23, x37)
        return x38

m = M().eval()
x33 = torch.randn(torch.Size([1, 144, 1, 1]))
x29 = torch.randn(torch.Size([1, 144, 28, 28]))
x23 = torch.randn(torch.Size([1, 144, 28, 28]))
start = time.time()
output = m(x33, x29, x23)
end = time.time()
print(end-start)
