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
        self.conv2d79 = Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x233, x219):
        x234=operator.add(x233, x219)
        x235=self.conv2d79(x234)
        return x235

m = M().eval()
x233 = torch.randn(torch.Size([1, 112, 14, 14]))
x219 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x233, x219)
end = time.time()
print(end-start)
