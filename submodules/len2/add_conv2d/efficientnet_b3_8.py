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
        self.conv2d64 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x196, x181):
        x197=operator.add(x196, x181)
        x198=self.conv2d64(x197)
        return x198

m = M().eval()
x196 = torch.randn(torch.Size([1, 96, 14, 14]))
x181 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x196, x181)
end = time.time()
print(end-start)
