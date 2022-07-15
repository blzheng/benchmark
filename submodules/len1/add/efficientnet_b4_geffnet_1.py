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

    def forward(self, x56, x42):
        x57=operator.add(x56, x42)
        return x57

m = M().eval()
x56 = torch.randn(torch.Size([1, 32, 56, 56]))
x42 = torch.randn(torch.Size([1, 32, 56, 56]))
start = time.time()
output = m(x56, x42)
end = time.time()
print(end-start)
