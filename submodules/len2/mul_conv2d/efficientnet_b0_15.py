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
        self.conv2d79 = Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x239, x234):
        x240=operator.mul(x239, x234)
        x241=self.conv2d79(x240)
        return x241

m = M().eval()
x239 = torch.randn(torch.Size([1, 1152, 1, 1]))
x234 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x239, x234)
end = time.time()
print(end-start)
