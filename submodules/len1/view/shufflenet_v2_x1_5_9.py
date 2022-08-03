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

    def forward(self, x117, x110, x112, x113):
        x118=x117.view(x110, -1, x112, x113)
        return x118

m = M().eval()
x117 = torch.randn(torch.Size([1, 176, 2, 14, 14]))
x110 = 1
x112 = 14
x113 = 14
start = time.time()
output = m(x117, x110, x112, x113)
end = time.time()
print(end-start)
