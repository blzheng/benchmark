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
        self.hardsigmoid2 = Hardsigmoid()

    def forward(self, x67, x63):
        x68=self.hardsigmoid2(x67)
        x69=operator.mul(x68, x63)
        return x69

m = M().eval()
x67 = torch.randn(torch.Size([1, 120, 1, 1]))
x63 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x67, x63)
end = time.time()
print(end-start)
