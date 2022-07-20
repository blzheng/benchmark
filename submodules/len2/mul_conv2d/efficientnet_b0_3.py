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
        self.conv2d19 = Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x55, x50):
        x56=operator.mul(x55, x50)
        x57=self.conv2d19(x56)
        return x57

m = M().eval()
x55 = torch.randn(torch.Size([1, 144, 1, 1]))
x50 = torch.randn(torch.Size([1, 144, 28, 28]))
start = time.time()
output = m(x55, x50)
end = time.time()
print(end-start)
