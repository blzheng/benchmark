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
        self.conv2d3 = Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x10, x5):
        x11=operator.add(x10, x5)
        x12=self.conv2d3(x11)
        return x12

m = M().eval()
x10 = torch.randn(torch.Size([1, 16, 112, 112]))
x5 = torch.randn(torch.Size([1, 16, 112, 112]))
start = time.time()
output = m(x10, x5)
end = time.time()
print(end-start)
