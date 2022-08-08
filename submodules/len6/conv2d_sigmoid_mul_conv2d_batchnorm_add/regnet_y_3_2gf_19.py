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
        self.conv2d102 = Conv2d(144, 576, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid19 = Sigmoid()
        self.conv2d103 = Conv2d(576, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d63 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x322, x319, x313):
        x323=self.conv2d102(x322)
        x324=self.sigmoid19(x323)
        x325=operator.mul(x324, x319)
        x326=self.conv2d103(x325)
        x327=self.batchnorm2d63(x326)
        x328=operator.add(x313, x327)
        return x328

m = M().eval()
x322 = torch.randn(torch.Size([1, 144, 1, 1]))
x319 = torch.randn(torch.Size([1, 576, 14, 14]))
x313 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x322, x319, x313)
end = time.time()
print(end-start)
