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
        self.sigmoid29 = Sigmoid()
        self.conv2d182 = Conv2d(1344, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d122 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d183 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d123 = BatchNorm2d(2304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x587, x583):
        x588=self.sigmoid29(x587)
        x589=operator.mul(x588, x583)
        x590=self.conv2d182(x589)
        x591=self.batchnorm2d122(x590)
        x592=self.conv2d183(x591)
        x593=self.batchnorm2d123(x592)
        return x593

m = M().eval()
x587 = torch.randn(torch.Size([1, 1344, 1, 1]))
x583 = torch.randn(torch.Size([1, 1344, 7, 7]))
start = time.time()
output = m(x587, x583)
end = time.time()
print(end-start)