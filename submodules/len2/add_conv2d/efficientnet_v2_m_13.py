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
        self.conv2d44 = Conv2d(160, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x145, x130):
        x146=operator.add(x145, x130)
        x147=self.conv2d44(x146)
        return x147

m = M().eval()
x145 = torch.randn(torch.Size([1, 160, 14, 14]))
x130 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x145, x130)
end = time.time()
print(end-start)
