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
        self.sigmoid43 = Sigmoid()
        self.conv2d216 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d128 = BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x677, x673):
        x678=self.sigmoid43(x677)
        x679=operator.mul(x678, x673)
        x680=self.conv2d216(x679)
        x681=self.batchnorm2d128(x680)
        return x681

m = M().eval()
x677 = torch.randn(torch.Size([1, 2304, 1, 1]))
x673 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x677, x673)
end = time.time()
print(end-start)
