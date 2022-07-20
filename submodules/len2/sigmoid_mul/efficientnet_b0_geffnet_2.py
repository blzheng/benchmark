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

    def forward(self, x39, x35):
        x40=x39.sigmoid()
        x41=operator.mul(x35, x40)
        return x41

m = M().eval()
x39 = torch.randn(torch.Size([1, 144, 1, 1]))
x35 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x39, x35)
end = time.time()
print(end-start)