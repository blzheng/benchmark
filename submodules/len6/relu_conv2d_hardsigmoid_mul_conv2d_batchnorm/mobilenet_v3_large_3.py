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
        self.relu14 = ReLU()
        self.conv2d39 = Conv2d(120, 480, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid3 = Hardsigmoid()
        self.conv2d40 = Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(112, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x113, x111):
        x114=self.relu14(x113)
        x115=self.conv2d39(x114)
        x116=self.hardsigmoid3(x115)
        x117=operator.mul(x116, x111)
        x118=self.conv2d40(x117)
        x119=self.batchnorm2d32(x118)
        return x119

m = M().eval()
x113 = torch.randn(torch.Size([1, 120, 1, 1]))
x111 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x113, x111)
end = time.time()
print(end-start)