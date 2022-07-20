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
        self.conv2d34 = Conv2d(160, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x113, x98):
        x114=operator.add(x113, x98)
        x115=self.conv2d34(x114)
        return x115

m = M().eval()
x113 = torch.randn(torch.Size([1, 160, 14, 14]))
x98 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x113, x98)
end = time.time()
print(end-start)
