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
        self.conv2d4 = Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(80, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x11, x5):
        x12=self.conv2d4(x11)
        x13=self.batchnorm2d4(x12)
        x14=operator.add(x5, x13)
        x15=self.relu3(x14)
        x16=self.conv2d5(x15)
        x17=self.batchnorm2d5(x16)
        return x17

m = M().eval()
x11 = torch.randn(torch.Size([1, 80, 56, 56]))
x5 = torch.randn(torch.Size([1, 80, 56, 56]))
start = time.time()
output = m(x11, x5)
end = time.time()
print(end-start)
