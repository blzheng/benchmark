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
        self.hardsigmoid4 = Hardsigmoid()

    def forward(self, x85, x81):
        x86=self.hardsigmoid4(x85)
        x87=operator.mul(x86, x81)
        return x87

m = M().eval()
x85 = torch.randn(torch.Size([1, 120, 1, 1]))
x81 = torch.randn(torch.Size([1, 120, 14, 14]))
start = time.time()
output = m(x85, x81)
end = time.time()
print(end-start)
