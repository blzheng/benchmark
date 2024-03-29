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

    def forward(self, x240, x245):
        x246=operator.mul(x240, x245)
        return x246

m = M().eval()
x240 = torch.randn(torch.Size([1, 816, 14, 14]))
x245 = torch.randn(torch.Size([1, 816, 1, 1]))
start = time.time()
output = m(x240, x245)
end = time.time()
print(end-start)
