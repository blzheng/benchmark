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
        self.conv2d82 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x293):
        x294=self.conv2d82(x293)
        return x294

m = M().eval()
x293 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x293)
end = time.time()
print(end-start)
