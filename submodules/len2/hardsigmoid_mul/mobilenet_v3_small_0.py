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
        self.hardsigmoid0 = Hardsigmoid()

    def forward(self, x10, x6):
        x11=self.hardsigmoid0(x10)
        x12=operator.mul(x11, x6)
        return x12

m = M().eval()
x10 = torch.randn(torch.Size([1, 16, 1, 1]))
x6 = torch.randn(torch.Size([1, 16, 56, 56]))
start = time.time()
output = m(x10, x6)
end = time.time()
print(end-start)
