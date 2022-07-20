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
        self.sigmoid10 = Sigmoid()
        self.conv2d51 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x157, x153):
        x158=self.sigmoid10(x157)
        x159=operator.mul(x158, x153)
        x160=self.conv2d51(x159)
        x161=self.batchnorm2d29(x160)
        return x161

m = M().eval()
x157 = torch.randn(torch.Size([1, 288, 1, 1]))
x153 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x157, x153)
end = time.time()
print(end-start)
