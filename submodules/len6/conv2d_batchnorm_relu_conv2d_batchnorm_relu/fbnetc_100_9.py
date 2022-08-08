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
        self.conv2d25 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu17 = ReLU(inplace=True)
        self.conv2d26 = Conv2d(192, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=192, bias=False)
        self.batchnorm2d26 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu18 = ReLU(inplace=True)

    def forward(self, x81):
        x82=self.conv2d25(x81)
        x83=self.batchnorm2d25(x82)
        x84=self.relu17(x83)
        x85=self.conv2d26(x84)
        x86=self.batchnorm2d26(x85)
        x87=self.relu18(x86)
        return x87

m = M().eval()
x81 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x81)
end = time.time()
print(end-start)
