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
        self.conv2d73 = Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x217, x203):
        x218=operator.add(x217, x203)
        x219=self.conv2d73(x218)
        return x219

m = M().eval()
x217 = torch.randn(torch.Size([1, 128, 14, 14]))
x203 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x217, x203)
end = time.time()
print(end-start)
