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

    def forward(self, x94, x102, x111):
        x103=operator.add(x94, x102)
        x112=operator.add(x103, x111)
        return x112

m = M().eval()
x94 = torch.randn(torch.Size([1, 96, 14, 14]))
x102 = torch.randn(torch.Size([1, 96, 14, 14]))
x111 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x94, x102, x111)
end = time.time()
print(end-start)
