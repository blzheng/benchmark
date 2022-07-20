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
        self.sigmoid4 = Sigmoid()

    def forward(self, x83, x79):
        x84=self.sigmoid4(x83)
        x85=operator.mul(x84, x79)
        return x85

m = M().eval()
x83 = torch.randn(torch.Size([1, 208, 1, 1]))
x79 = torch.randn(torch.Size([1, 208, 14, 14]))
start = time.time()
output = m(x83, x79)
end = time.time()
print(end-start)
