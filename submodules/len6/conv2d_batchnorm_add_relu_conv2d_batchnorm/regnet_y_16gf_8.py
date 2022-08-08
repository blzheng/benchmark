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
        self.conv2d33 = Conv2d(448, 1232, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d21 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu28 = ReLU(inplace=True)
        self.conv2d39 = Conv2d(1232, 1232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x103, x119):
        x104=self.conv2d33(x103)
        x105=self.batchnorm2d21(x104)
        x120=operator.add(x105, x119)
        x121=self.relu28(x120)
        x122=self.conv2d39(x121)
        x123=self.batchnorm2d25(x122)
        return x123

m = M().eval()
x103 = torch.randn(torch.Size([1, 448, 28, 28]))
x119 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x103, x119)
end = time.time()
print(end-start)
