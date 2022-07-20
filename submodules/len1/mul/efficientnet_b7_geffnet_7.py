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

    def forward(self, x104, x109):
        x110=operator.mul(x104, x109)
        return x110

m = M().eval()
x104 = torch.randn(torch.Size([1, 288, 56, 56]))
x109 = torch.randn(torch.Size([1, 288, 1, 1]))
start = time.time()
output = m(x104, x109)
end = time.time()
print(end-start)