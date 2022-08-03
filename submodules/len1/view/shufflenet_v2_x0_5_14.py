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

    def forward(self, x174, x176, x180, x178, x179):
        x181=x174.view(x176, 2, x180, x178, x179)
        return x181

m = M().eval()
x174 = torch.randn(torch.Size([1, 96, 14, 14]))
x176 = 1
x180 = 48
x178 = 14
x179 = 14
start = time.time()
output = m(x174, x176, x180, x178, x179)
end = time.time()
print(end-start)
