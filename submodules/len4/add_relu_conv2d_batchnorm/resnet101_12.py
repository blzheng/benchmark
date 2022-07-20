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
        self.relu37 = ReLU(inplace=True)
        self.conv2d43 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x138, x130):
        x139=operator.add(x138, x130)
        x140=self.relu37(x139)
        x141=self.conv2d43(x140)
        x142=self.batchnorm2d43(x141)
        return x142

m = M().eval()
x138 = torch.randn(torch.Size([1, 1024, 14, 14]))
x130 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x138, x130)
end = time.time()
print(end-start)
