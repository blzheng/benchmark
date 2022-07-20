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
        self.conv2d23 = Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid4 = Sigmoid()
        self.conv2d24 = Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x67, x64):
        x68=self.conv2d23(x67)
        x69=self.sigmoid4(x68)
        x70=operator.mul(x69, x64)
        x71=self.conv2d24(x70)
        x72=self.batchnorm2d14(x71)
        return x72

m = M().eval()
x67 = torch.randn(torch.Size([1, 10, 1, 1]))
x64 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x67, x64)
end = time.time()
print(end-start)
