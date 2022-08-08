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
        self.conv2d24 = Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d25 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x64, x69, x58):
        x70=operator.mul(x64, x69)
        x71=self.conv2d24(x70)
        x72=self.batchnorm2d14(x71)
        x73=operator.add(x72, x58)
        x74=self.conv2d25(x73)
        x75=self.batchnorm2d15(x74)
        return x75

m = M().eval()
x64 = torch.randn(torch.Size([1, 240, 28, 28]))
x69 = torch.randn(torch.Size([1, 240, 1, 1]))
x58 = torch.randn(torch.Size([1, 40, 28, 28]))
start = time.time()
output = m(x64, x69, x58)
end = time.time()
print(end-start)
