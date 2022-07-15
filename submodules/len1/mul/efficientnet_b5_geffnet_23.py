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

    def forward(self, x343, x348):
        x349=operator.mul(x343, x348)
        return x349

m = M().eval()
x343 = torch.randn(torch.Size([1, 1056, 14, 14]))
x348 = torch.randn(torch.Size([1, 1056, 1, 1]))
start = time.time()
output = m(x343, x348)
end = time.time()
print(end-start)
