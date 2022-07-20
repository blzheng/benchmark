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
        self.batchnorm2d7 = BatchNorm2d(72, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(24, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)

    def forward(self, x20):
        x21=self.batchnorm2d7(x20)
        x22=self.relu5(x21)
        x23=self.conv2d8(x22)
        x24=self.batchnorm2d8(x23)
        return x24

m = M().eval()
x20 = torch.randn(torch.Size([1, 72, 56, 56]))
start = time.time()
output = m(x20)
end = time.time()
print(end-start)
