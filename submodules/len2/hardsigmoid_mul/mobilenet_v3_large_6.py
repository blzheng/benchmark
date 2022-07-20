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

    def forward(self, x158, x154):
        x159=self.hardsigmoid6(x158)
        x160=operator.mul(x159, x154)
        return x160

m = M().eval()
x158 = torch.randn(torch.Size([1, 960, 1, 1]))
x154 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x158, x154)
end = time.time()
print(end-start)
