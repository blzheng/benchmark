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
        self.conv2d137 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid22 = Sigmoid()
        self.conv2d138 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x439, x436):
        x440=self.conv2d137(x439)
        x441=self.sigmoid22(x440)
        x442=operator.mul(x441, x436)
        x443=self.conv2d138(x442)
        return x443

m = M().eval()
x439 = torch.randn(torch.Size([1, 76, 1, 1]))
x436 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x439, x436)
end = time.time()
print(end-start)
