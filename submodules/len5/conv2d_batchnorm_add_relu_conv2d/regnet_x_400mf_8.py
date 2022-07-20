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
        self.conv2d21 = Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu18 = ReLU(inplace=True)
        self.conv2d22 = Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x65, x59):
        x66=self.conv2d21(x65)
        x67=self.batchnorm2d21(x66)
        x68=operator.add(x59, x67)
        x69=self.relu18(x68)
        x70=self.conv2d22(x69)
        return x70

m = M().eval()
x65 = torch.randn(torch.Size([1, 160, 14, 14]))
x59 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x65, x59)
end = time.time()
print(end-start)
