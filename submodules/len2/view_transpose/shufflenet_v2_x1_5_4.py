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

    def forward(self, x108, x110, x114, x112, x113):
        x115=x108.view(x110, 2, x114, x112, x113)
        x116=torch.transpose(x115, 1, 2)
        return x116

m = M().eval()
x108 = torch.randn(torch.Size([1, 352, 14, 14]))
x110 = 1
x114 = 176
x112 = 14
x113 = 14
start = time.time()
output = m(x108, x110, x114, x112, x113)
end = time.time()
print(end-start)
