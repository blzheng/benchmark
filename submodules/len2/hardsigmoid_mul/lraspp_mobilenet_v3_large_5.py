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
        self.hardsigmoid5 = Hardsigmoid()

    def forward(self, x144, x140):
        x145=self.hardsigmoid5(x144)
        x146=operator.mul(x145, x140)
        return x146

m = M().eval()
x144 = torch.randn(torch.Size([1, 672, 1, 1]))
x140 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x144, x140)
end = time.time()
print(end-start)
