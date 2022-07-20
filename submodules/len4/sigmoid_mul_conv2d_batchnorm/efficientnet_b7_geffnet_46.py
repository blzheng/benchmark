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
        self.conv2d231 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d137 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x689, x685):
        x690=x689.sigmoid()
        x691=operator.mul(x685, x690)
        x692=self.conv2d231(x691)
        x693=self.batchnorm2d137(x692)
        return x693

m = M().eval()
x689 = torch.randn(torch.Size([1, 2304, 1, 1]))
x685 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x689, x685)
end = time.time()
print(end-start)
