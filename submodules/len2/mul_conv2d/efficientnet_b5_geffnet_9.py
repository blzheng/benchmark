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
        self.conv2d47 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x135, x140):
        x141=operator.mul(x135, x140)
        x142=self.conv2d47(x141)
        return x142

m = M().eval()
x135 = torch.randn(torch.Size([1, 384, 28, 28]))
x140 = torch.randn(torch.Size([1, 384, 1, 1]))
start = time.time()
output = m(x135, x140)
end = time.time()
print(end-start)
