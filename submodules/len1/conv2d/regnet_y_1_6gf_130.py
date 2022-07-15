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
        self.conv2d130 = Conv2d(336, 888, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x409):
        x412=self.conv2d130(x409)
        return x412

m = M().eval()
x409 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x409)
end = time.time()
print(end-start)
