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
        self.batchnorm2d33 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu40 = ReLU(inplace=True)
        self.conv2d54 = Conv2d(1232, 1232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x166, x153):
        x167=self.batchnorm2d33(x166)
        x168=operator.add(x153, x167)
        x169=self.relu40(x168)
        x170=self.conv2d54(x169)
        x171=self.batchnorm2d34(x170)
        return x171

m = M().eval()
x166 = torch.randn(torch.Size([1, 1232, 14, 14]))
x153 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x166, x153)
end = time.time()
print(end-start)
