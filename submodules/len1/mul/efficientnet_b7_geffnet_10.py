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

    def forward(self, x149, x154):
        x155=operator.mul(x149, x154)
        return x155

m = M().eval()
x149 = torch.randn(torch.Size([1, 288, 56, 56]))
x154 = torch.randn(torch.Size([1, 288, 1, 1]))
start = time.time()
output = m(x149, x154)
end = time.time()
print(end-start)