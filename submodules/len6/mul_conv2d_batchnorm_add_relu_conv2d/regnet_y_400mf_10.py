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
        self.conv2d59 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu44 = ReLU(inplace=True)
        self.conv2d60 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x182, x177, x171):
        x183=operator.mul(x182, x177)
        x184=self.conv2d59(x183)
        x185=self.batchnorm2d37(x184)
        x186=operator.add(x171, x185)
        x187=self.relu44(x186)
        x188=self.conv2d60(x187)
        return x188

m = M().eval()
x182 = torch.randn(torch.Size([1, 440, 1, 1]))
x177 = torch.randn(torch.Size([1, 440, 7, 7]))
x171 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x182, x177, x171)
end = time.time()
print(end-start)