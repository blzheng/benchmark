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
        self.sigmoid5 = Sigmoid()

    def forward(self, x153, x149):
        x154=self.sigmoid5(x153)
        x155=operator.mul(x154, x149)
        return x155

m = M().eval()
x153 = torch.randn(torch.Size([1, 512, 1, 1]))
x149 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x153, x149)
end = time.time()
print(end-start)
