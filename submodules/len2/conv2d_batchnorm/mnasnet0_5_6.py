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
        self.conv2d6 = Conv2d(16, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(48, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x16):
        x17=self.conv2d6(x16)
        x18=self.batchnorm2d6(x17)
        return x18

m = M().eval()
x16 = torch.randn(torch.Size([1, 16, 56, 56]))
start = time.time()
output = m(x16)
end = time.time()
print(end-start)
