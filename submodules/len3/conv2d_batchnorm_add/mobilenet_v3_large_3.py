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
        self.conv2d23 = Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x67, x55):
        x68=self.conv2d23(x67)
        x69=self.batchnorm2d17(x68)
        x70=operator.add(x69, x55)
        return x70

m = M().eval()
x67 = torch.randn(torch.Size([1, 120, 28, 28]))
x55 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x67, x55)
end = time.time()
print(end-start)
