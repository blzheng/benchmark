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

    def forward(self, x14, x16):
        x17=torch.cat([x14, x16], 1)
        return x17

m = M().eval()
x14 = torch.randn(torch.Size([1, 64, 54, 54]))
x16 = torch.randn(torch.Size([1, 64, 54, 54]))
start = time.time()
output = m(x14, x16)
end = time.time()
print(end-start)
