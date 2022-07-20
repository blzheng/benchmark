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
        self.conv2d122 = Conv2d(1200, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x379, x374):
        x380=operator.mul(x379, x374)
        x381=self.conv2d122(x380)
        return x381

m = M().eval()
x379 = torch.randn(torch.Size([1, 1200, 1, 1]))
x374 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x379, x374)
end = time.time()
print(end-start)
