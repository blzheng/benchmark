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

    def forward(self, x68, x64):
        x69=self.sigmoid4(x68)
        x70=operator.mul(x69, x64)
        return x70

m = M().eval()
x68 = torch.randn(torch.Size([1, 240, 1, 1]))
x64 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x68, x64)
end = time.time()
print(end-start)
