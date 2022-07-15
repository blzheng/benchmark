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

    def forward(self, x119, x111):
        x120=operator.add(x119, x111)
        return x120

m = M().eval()
x119 = torch.randn(torch.Size([1, 192, 7, 7]))
x111 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x119, x111)
end = time.time()
print(end-start)
