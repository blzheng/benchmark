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
        self.conv2d35 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x101, x87):
        x102=operator.add(x101, x87)
        x103=self.conv2d35(x102)
        return x103

m = M().eval()
x101 = torch.randn(torch.Size([1, 80, 14, 14]))
x87 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x101, x87)
end = time.time()
print(end-start)
