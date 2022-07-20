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
        self.sigmoid40 = Sigmoid()
        self.conv2d237 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x761, x757):
        x762=self.sigmoid40(x761)
        x763=operator.mul(x762, x757)
        x764=self.conv2d237(x763)
        return x764

m = M().eval()
x761 = torch.randn(torch.Size([1, 2304, 1, 1]))
x757 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x761, x757)
end = time.time()
print(end-start)
