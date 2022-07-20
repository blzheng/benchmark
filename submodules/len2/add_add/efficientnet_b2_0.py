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

    def forward(self, x56, x41, x72):
        x57=operator.add(x56, x41)
        x73=operator.add(x72, x57)
        return x73

m = M().eval()
x56 = torch.randn(torch.Size([1, 24, 56, 56]))
x41 = torch.randn(torch.Size([1, 24, 56, 56]))
x72 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x56, x41, x72)
end = time.time()
print(end-start)
