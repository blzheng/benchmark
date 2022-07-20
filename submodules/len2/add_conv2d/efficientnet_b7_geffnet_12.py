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
        self.conv2d77 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x231, x217):
        x232=operator.add(x231, x217)
        x233=self.conv2d77(x232)
        return x233

m = M().eval()
x231 = torch.randn(torch.Size([1, 80, 28, 28]))
x217 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x231, x217)
end = time.time()
print(end-start)
