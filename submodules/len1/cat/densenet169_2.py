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

    def forward(self, x4, x11, x18):
        x19=torch.cat([x4, x11, x18], 1)
        return x19

m = M().eval()
x4 = torch.randn(torch.Size([1, 64, 56, 56]))
x11 = torch.randn(torch.Size([1, 32, 56, 56]))
x18 = torch.randn(torch.Size([1, 32, 56, 56]))
start = time.time()
output = m(x4, x11, x18)
end = time.time()
print(end-start)
