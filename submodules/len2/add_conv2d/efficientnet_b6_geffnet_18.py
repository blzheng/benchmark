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
        self.conv2d113 = Conv2d(144, 864, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x337, x323):
        x338=operator.add(x337, x323)
        x339=self.conv2d113(x338)
        return x339

m = M().eval()
x337 = torch.randn(torch.Size([1, 144, 14, 14]))
x323 = torch.randn(torch.Size([1, 144, 14, 14]))
start = time.time()
output = m(x337, x323)
end = time.time()
print(end-start)
