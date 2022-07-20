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
        self.conv2d10 = Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d10 = BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x31, x22):
        x32=operator.add(x31, x22)
        x33=self.conv2d10(x32)
        x34=self.batchnorm2d10(x33)
        return x34

m = M().eval()
x31 = torch.randn(torch.Size([1, 24, 56, 56]))
x22 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x31, x22)
end = time.time()
print(end-start)
