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
        self.conv2d47 = Conv2d(240, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x140, x136):
        x141=x140.sigmoid()
        x142=operator.mul(x136, x141)
        x143=self.conv2d47(x142)
        return x143

m = M().eval()
x140 = torch.randn(torch.Size([1, 240, 1, 1]))
x136 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x140, x136)
end = time.time()
print(end-start)
