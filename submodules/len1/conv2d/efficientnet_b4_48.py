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
        self.conv2d48 = Conv2d(336, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x147):
        x148=self.conv2d48(x147)
        return x148

m = M().eval()
x147 = torch.randn(torch.Size([1, 336, 28, 28]))
start = time.time()
output = m(x147)
end = time.time()
print(end-start)
