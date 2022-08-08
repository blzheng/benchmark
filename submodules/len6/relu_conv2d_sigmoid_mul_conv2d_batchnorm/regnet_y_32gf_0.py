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
        self.relu3 = ReLU()
        self.conv2d5 = Conv2d(8, 232, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid0 = Sigmoid()
        self.conv2d6 = Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x13, x11):
        x14=self.relu3(x13)
        x15=self.conv2d5(x14)
        x16=self.sigmoid0(x15)
        x17=operator.mul(x16, x11)
        x18=self.conv2d6(x17)
        x19=self.batchnorm2d4(x18)
        return x19

m = M().eval()
x13 = torch.randn(torch.Size([1, 8, 1, 1]))
x11 = torch.randn(torch.Size([1, 232, 56, 56]))
start = time.time()
output = m(x13, x11)
end = time.time()
print(end-start)
