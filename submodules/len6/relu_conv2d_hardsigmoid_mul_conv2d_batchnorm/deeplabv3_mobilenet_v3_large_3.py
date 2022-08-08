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

    def forward(self, x115, x113):
        x116=self.relu14(x115)
        x117=self.conv2d39(x116)
        x118=self.hardsigmoid3(x117)
        x119=operator.mul(x118, x113)
        x120=self.conv2d40(x119)
        x121=self.batchnorm2d32(x120)
        return x121

m = M().eval()
x115 = torch.randn(torch.Size([1, 120, 1, 1]))
x113 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x115, x113)
end = time.time()
print(end-start)
