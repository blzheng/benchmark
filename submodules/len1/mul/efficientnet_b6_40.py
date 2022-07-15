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

    def forward(self, x633, x628):
        x634=operator.mul(x633, x628)
        return x634

m = M().eval()
x633 = torch.randn(torch.Size([1, 2064, 1, 1]))
x628 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x633, x628)
end = time.time()
print(end-start)
