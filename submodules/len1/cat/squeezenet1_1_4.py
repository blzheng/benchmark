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

    def forward(self, x37, x39):
        x40=torch.cat([x37, x39], 1)
        return x40

m = M().eval()
x37 = torch.randn(torch.Size([1, 192, 13, 13]))
x39 = torch.randn(torch.Size([1, 192, 13, 13]))
start = time.time()
output = m(x37, x39)
end = time.time()
print(end-start)
