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

    def forward(self, x108, x113):
        x114=operator.mul(x108, x113)
        return x114

m = M().eval()
x108 = torch.randn(torch.Size([1, 480, 14, 14]))
x113 = torch.randn(torch.Size([1, 480, 1, 1]))
start = time.time()
output = m(x108, x113)
end = time.time()
print(end-start)
