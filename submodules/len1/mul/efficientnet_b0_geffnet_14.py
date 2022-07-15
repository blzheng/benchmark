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

    def forward(self, x211, x216):
        x217=operator.mul(x211, x216)
        return x217

m = M().eval()
x211 = torch.randn(torch.Size([1, 1152, 7, 7]))
x216 = torch.randn(torch.Size([1, 1152, 1, 1]))
start = time.time()
output = m(x211, x216)
end = time.time()
print(end-start)
