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
        self.sigmoid12 = Sigmoid()
        self.conv2d69 = Conv2d(440, 440, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu52 = ReLU(inplace=True)

    def forward(self, x213, x209, x203):
        x214=self.sigmoid12(x213)
        x215=operator.mul(x214, x209)
        x216=self.conv2d69(x215)
        x217=self.batchnorm2d43(x216)
        x218=operator.add(x203, x217)
        x219=self.relu52(x218)
        return x219

m = M().eval()
x213 = torch.randn(torch.Size([1, 440, 1, 1]))
x209 = torch.randn(torch.Size([1, 440, 7, 7]))
x203 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x213, x209, x203)
end = time.time()
print(end-start)
