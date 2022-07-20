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
        self.conv2d126 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x372, x377):
        x378=operator.mul(x372, x377)
        x379=self.conv2d126(x378)
        return x379

m = M().eval()
x372 = torch.randn(torch.Size([1, 960, 14, 14]))
x377 = torch.randn(torch.Size([1, 960, 1, 1]))
start = time.time()
output = m(x372, x377)
end = time.time()
print(end-start)
