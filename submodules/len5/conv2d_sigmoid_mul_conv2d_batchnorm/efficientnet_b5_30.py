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
        self.conv2d151 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid30 = Sigmoid()
        self.conv2d152 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d90 = BatchNorm2d(304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x471, x468):
        x472=self.conv2d151(x471)
        x473=self.sigmoid30(x472)
        x474=operator.mul(x473, x468)
        x475=self.conv2d152(x474)
        x476=self.batchnorm2d90(x475)
        return x476

m = M().eval()
x471 = torch.randn(torch.Size([1, 76, 1, 1]))
x468 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x471, x468)
end = time.time()
print(end-start)
