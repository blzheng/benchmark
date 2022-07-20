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
        self.conv2d211 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x625, x630):
        x631=operator.mul(x625, x630)
        x632=self.conv2d211(x631)
        return x632

m = M().eval()
x625 = torch.randn(torch.Size([1, 2304, 7, 7]))
x630 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x625, x630)
end = time.time()
print(end-start)
