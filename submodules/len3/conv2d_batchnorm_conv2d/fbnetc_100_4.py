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
        self.conv2d51 = Conv2d(672, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(184, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d52 = Conv2d(184, 1104, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x166):
        x167=self.conv2d51(x166)
        x168=self.batchnorm2d51(x167)
        x169=self.conv2d52(x168)
        return x169

m = M().eval()
x166 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x166)
end = time.time()
print(end-start)
