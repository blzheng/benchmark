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
        self.sigmoid7 = Sigmoid()
        self.conv2d37 = Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x112, x108):
        x113=self.sigmoid7(x112)
        x114=operator.mul(x113, x108)
        x115=self.conv2d37(x114)
        return x115

m = M().eval()
x112 = torch.randn(torch.Size([1, 240, 1, 1]))
x108 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x112, x108)
end = time.time()
print(end-start)
