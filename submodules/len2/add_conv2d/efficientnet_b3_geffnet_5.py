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
        self.conv2d49 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x144, x130):
        x145=operator.add(x144, x130)
        x146=self.conv2d49(x145)
        return x146

m = M().eval()
x144 = torch.randn(torch.Size([1, 96, 14, 14]))
x130 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x144, x130)
end = time.time()
print(end-start)
