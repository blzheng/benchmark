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
        self.conv2d124 = Conv2d(176, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x399, x384):
        x400=operator.add(x399, x384)
        x401=self.conv2d124(x400)
        return x401

m = M().eval()
x399 = torch.randn(torch.Size([1, 176, 14, 14]))
x384 = torch.randn(torch.Size([1, 176, 14, 14]))
start = time.time()
output = m(x399, x384)
end = time.time()
print(end-start)
