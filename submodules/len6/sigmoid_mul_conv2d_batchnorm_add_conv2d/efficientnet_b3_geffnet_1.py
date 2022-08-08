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
        self.conv2d38 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d39 = Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x111, x107, x101):
        x112=x111.sigmoid()
        x113=operator.mul(x107, x112)
        x114=self.conv2d38(x113)
        x115=self.batchnorm2d22(x114)
        x116=operator.add(x115, x101)
        x117=self.conv2d39(x116)
        return x117

m = M().eval()
x111 = torch.randn(torch.Size([1, 288, 1, 1]))
x107 = torch.randn(torch.Size([1, 288, 28, 28]))
x101 = torch.randn(torch.Size([1, 48, 28, 28]))
start = time.time()
output = m(x111, x107, x101)
end = time.time()
print(end-start)
