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

    def forward(self, x38, x34):
        x39=x38.sigmoid()
        x40=operator.mul(x34, x39)
        return x40

m = M().eval()
x38 = torch.randn(torch.Size([1, 144, 1, 1]))
x34 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x38, x34)
end = time.time()
print(end-start)
