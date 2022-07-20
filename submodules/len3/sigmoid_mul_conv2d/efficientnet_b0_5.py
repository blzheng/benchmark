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
        self.sigmoid5 = Sigmoid()
        self.conv2d29 = Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x84, x80):
        x85=self.sigmoid5(x84)
        x86=operator.mul(x85, x80)
        x87=self.conv2d29(x86)
        return x87

m = M().eval()
x84 = torch.randn(torch.Size([1, 240, 1, 1]))
x80 = torch.randn(torch.Size([1, 240, 14, 14]))
start = time.time()
output = m(x84, x80)
end = time.time()
print(end-start)
