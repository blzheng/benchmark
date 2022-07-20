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

    def forward(self, x148, x139, x158):
        x149=operator.add(x148, x139)
        x159=operator.add(x158, x149)
        return x159

m = M().eval()
x148 = torch.randn(torch.Size([1, 112, 14, 14]))
x139 = torch.randn(torch.Size([1, 112, 14, 14]))
x158 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x148, x139, x158)
end = time.time()
print(end-start)
