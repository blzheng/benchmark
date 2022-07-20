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
        self.conv2d51 = Conv2d(432, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu48 = ReLU(inplace=True)
        self.conv2d52 = Conv2d(432, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x165, x159):
        x166=self.conv2d51(x165)
        x167=self.batchnorm2d51(x166)
        x168=operator.add(x159, x167)
        x169=self.relu48(x168)
        x170=self.conv2d52(x169)
        return x170

m = M().eval()
x165 = torch.randn(torch.Size([1, 432, 14, 14]))
x159 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x165, x159)
end = time.time()
print(end-start)
