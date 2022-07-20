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
        self.conv2d31 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x99, x90):
        x100=operator.add(x99, x90)
        x101=self.conv2d31(x100)
        return x101

m = M().eval()
x99 = torch.randn(torch.Size([1, 64, 14, 14]))
x90 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x99, x90)
end = time.time()
print(end-start)
