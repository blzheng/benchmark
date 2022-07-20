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
        self.conv2d24 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x59, x67):
        x68=operator.add(x59, x67)
        x69=self.conv2d24(x68)
        return x69

m = M().eval()
x59 = torch.randn(torch.Size([1, 64, 14, 14]))
x67 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x59, x67)
end = time.time()
print(end-start)
