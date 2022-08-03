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

    def forward(self, x149, x156):
        x157=operator.add(x149, x156)
        return x157

m = M().eval()
x149 = torch.randn(torch.Size([1, 14, 14, 384]))
x156 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x149, x156)
end = time.time()
print(end-start)
