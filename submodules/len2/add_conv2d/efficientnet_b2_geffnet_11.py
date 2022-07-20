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
        self.conv2d89 = Conv2d(208, 1248, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x262, x248):
        x263=operator.add(x262, x248)
        x264=self.conv2d89(x263)
        return x264

m = M().eval()
x262 = torch.randn(torch.Size([1, 208, 7, 7]))
x248 = torch.randn(torch.Size([1, 208, 7, 7]))
start = time.time()
output = m(x262, x248)
end = time.time()
print(end-start)
