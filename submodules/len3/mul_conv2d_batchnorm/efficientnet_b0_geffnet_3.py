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
        self.conv2d19 = Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x50, x55):
        x56=operator.mul(x50, x55)
        x57=self.conv2d19(x56)
        x58=self.batchnorm2d11(x57)
        return x58

m = M().eval()
x50 = torch.randn(torch.Size([1, 144, 28, 28]))
x55 = torch.randn(torch.Size([1, 144, 1, 1]))
start = time.time()
output = m(x50, x55)
end = time.time()
print(end-start)
