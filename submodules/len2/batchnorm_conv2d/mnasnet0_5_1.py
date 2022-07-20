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
        self.batchnorm2d5 = BatchNorm2d(16, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.conv2d6 = Conv2d(16, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x15):
        x16=self.batchnorm2d5(x15)
        x17=self.conv2d6(x16)
        return x17

m = M().eval()
x15 = torch.randn(torch.Size([1, 16, 56, 56]))
start = time.time()
output = m(x15)
end = time.time()
print(end-start)
