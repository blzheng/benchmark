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
        self.batchnorm2d29 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        self.conv2d36 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x103, x96):
        x104=self.batchnorm2d29(x103)
        x105=operator.add(x104, x96)
        x106=self.conv2d36(x105)
        return x106

m = M().eval()
x103 = torch.randn(torch.Size([1, 80, 14, 14]))
x96 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x103, x96)
end = time.time()
print(end-start)
