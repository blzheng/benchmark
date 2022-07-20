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

    def forward(self, x187, x178, x197):
        x188=operator.add(x187, x178)
        x198=operator.add(x197, x188)
        return x198

m = M().eval()
x187 = torch.randn(torch.Size([1, 184, 7, 7]))
x178 = torch.randn(torch.Size([1, 184, 7, 7]))
x197 = torch.randn(torch.Size([1, 184, 7, 7]))
start = time.time()
output = m(x187, x178, x197)
end = time.time()
print(end-start)
