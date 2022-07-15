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

    def forward(self, x242, x227):
        x243=operator.add(x242, x227)
        return x243

m = M().eval()
x242 = torch.randn(torch.Size([1, 112, 14, 14]))
x227 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x242, x227)
end = time.time()
print(end-start)
