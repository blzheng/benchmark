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

    def forward(self, x71, x57, x86):
        x72=operator.add(x71, x57)
        x87=operator.add(x86, x72)
        return x87

m = M().eval()
x71 = torch.randn(torch.Size([1, 32, 56, 56]))
x57 = torch.randn(torch.Size([1, 32, 56, 56]))
x86 = torch.randn(torch.Size([1, 32, 56, 56]))
start = time.time()
output = m(x71, x57, x86)
end = time.time()
print(end-start)
