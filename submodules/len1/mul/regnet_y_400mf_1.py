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

    def forward(self, x34, x29):
        x35=operator.mul(x34, x29)
        return x35

m = M().eval()
x34 = torch.randn(torch.Size([1, 104, 1, 1]))
x29 = torch.randn(torch.Size([1, 104, 28, 28]))
start = time.time()
output = m(x34, x29)
end = time.time()
print(end-start)
