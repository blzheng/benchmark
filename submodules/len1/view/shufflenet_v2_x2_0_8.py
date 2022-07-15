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

    def forward(self, x110, x114, x112, x113):
        x115=x108.view(x110, 2, x114, x112, x113)
        return x115

m = M().eval()
x110 = 1
x114 = 244
x112 = 14
x113 = 14
start = time.time()
output = m(x110, x114, x112, x113)
end = time.time()
print(end-start)
