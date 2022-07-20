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
        self.conv2d147 = Conv2d(1200, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x437, x433):
        x438=x437.sigmoid()
        x439=operator.mul(x433, x438)
        x440=self.conv2d147(x439)
        return x440

m = M().eval()
x437 = torch.randn(torch.Size([1, 1200, 1, 1]))
x433 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x437, x433)
end = time.time()
print(end-start)
