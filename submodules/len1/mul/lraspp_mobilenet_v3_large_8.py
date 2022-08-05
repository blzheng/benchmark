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

    def forward(self, x184, x187):
        x188=operator.mul(x184, x187)
        return x188

m = M().eval()
x184 = torch.randn(torch.Size([1, 128, 14, 14]))
x187 = torch.randn(torch.Size([1, 128, 1, 1]))
start = time.time()
output = m(x184, x187)
end = time.time()
print(end-start)
