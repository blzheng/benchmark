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
        self.hardsigmoid6 = Hardsigmoid()

    def forward(self, x114, x110):
        x115=self.hardsigmoid6(x114)
        x116=operator.mul(x115, x110)
        return x116

m = M().eval()
x114 = torch.randn(torch.Size([1, 288, 1, 1]))
x110 = torch.randn(torch.Size([1, 288, 7, 7]))
start = time.time()
output = m(x114, x110)
end = time.time()
print(end-start)
