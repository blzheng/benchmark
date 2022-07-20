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
        self.conv2d38 = Conv2d(696, 1392, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d24 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu32 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(1392, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x119, x135):
        x120=self.conv2d38(x119)
        x121=self.batchnorm2d24(x120)
        x136=operator.add(x121, x135)
        x137=self.relu32(x136)
        x138=self.conv2d44(x137)
        return x138

m = M().eval()
x119 = torch.randn(torch.Size([1, 696, 28, 28]))
x135 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x119, x135)
end = time.time()
print(end-start)
