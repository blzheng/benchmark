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

    def forward(self, x129, x114):
        x130=operator.add(x129, x114)
        return x130

m = M().eval()
x129 = torch.randn(torch.Size([1, 160, 14, 14]))
x114 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x129, x114)
end = time.time()
print(end-start)
