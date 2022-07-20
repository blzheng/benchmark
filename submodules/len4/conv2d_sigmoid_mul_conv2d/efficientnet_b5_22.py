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
        self.conv2d111 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid22 = Sigmoid()
        self.conv2d112 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x345, x342):
        x346=self.conv2d111(x345)
        x347=self.sigmoid22(x346)
        x348=operator.mul(x347, x342)
        x349=self.conv2d112(x348)
        return x349

m = M().eval()
x345 = torch.randn(torch.Size([1, 44, 1, 1]))
x342 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x345, x342)
end = time.time()
print(end-start)