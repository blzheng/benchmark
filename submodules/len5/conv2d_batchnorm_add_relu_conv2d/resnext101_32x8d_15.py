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
        self.conv2d42 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu37 = ReLU(inplace=True)
        self.conv2d43 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x136, x130):
        x137=self.conv2d42(x136)
        x138=self.batchnorm2d42(x137)
        x139=operator.add(x138, x130)
        x140=self.relu37(x139)
        x141=self.conv2d43(x140)
        return x141

m = M().eval()
x136 = torch.randn(torch.Size([1, 1024, 14, 14]))
x130 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x136, x130)
end = time.time()
print(end-start)
