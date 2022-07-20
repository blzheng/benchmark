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
        self.conv2d52 = Conv2d(184, 1104, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu35 = ReLU(inplace=True)

    def forward(self, x168):
        x169=self.conv2d52(x168)
        x170=self.batchnorm2d52(x169)
        x171=self.relu35(x170)
        return x171

m = M().eval()
x168 = torch.randn(torch.Size([1, 184, 7, 7]))
start = time.time()
output = m(x168)
end = time.time()
print(end-start)
