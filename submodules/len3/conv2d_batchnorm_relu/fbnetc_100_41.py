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
        self.conv2d61 = Conv2d(184, 1104, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d61 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu41 = ReLU(inplace=True)

    def forward(self, x198):
        x199=self.conv2d61(x198)
        x200=self.batchnorm2d61(x199)
        x201=self.relu41(x200)
        return x201

m = M().eval()
x198 = torch.randn(torch.Size([1, 184, 7, 7]))
start = time.time()
output = m(x198)
end = time.time()
print(end-start)
