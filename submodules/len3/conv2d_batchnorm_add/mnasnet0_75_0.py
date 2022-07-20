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
        self.conv2d11 = Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(24, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x31, x25):
        x32=self.conv2d11(x31)
        x33=self.batchnorm2d11(x32)
        x34=operator.add(x33, x25)
        return x34

m = M().eval()
x31 = torch.randn(torch.Size([1, 72, 56, 56]))
x25 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x31, x25)
end = time.time()
print(end-start)