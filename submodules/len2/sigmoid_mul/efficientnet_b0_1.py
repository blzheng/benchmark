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
        self.sigmoid1 = Sigmoid()

    def forward(self, x24, x20):
        x25=self.sigmoid1(x24)
        x26=operator.mul(x25, x20)
        return x26

m = M().eval()
x24 = torch.randn(torch.Size([1, 96, 1, 1]))
x20 = torch.randn(torch.Size([1, 96, 56, 56]))
start = time.time()
output = m(x24, x20)
end = time.time()
print(end-start)
