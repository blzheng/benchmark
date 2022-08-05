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
        self.conv2d29 = Conv2d(200, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x86):
        x87=self.conv2d29(x86)
        return x87

m = M().eval()
x86 = torch.randn(torch.Size([1, 200, 14, 14]))
start = time.time()
output = m(x86)
end = time.time()
print(end-start)
