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
        self.conv2d35 = Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid7 = Sigmoid()
        self.conv2d36 = Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x108, x105):
        x109=self.conv2d35(x108)
        x110=self.sigmoid7(x109)
        x111=operator.mul(x110, x105)
        x112=self.conv2d36(x111)
        x113=self.batchnorm2d20(x112)
        return x113

m = M().eval()
x108 = torch.randn(torch.Size([1, 12, 1, 1]))
x105 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x108, x105)
end = time.time()
print(end-start)
