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

    def forward(self, x138, x130):
        x139=operator.add(x138, x130)
        return x139

m = M().eval()
x138 = torch.randn(torch.Size([1, 1024, 14, 14]))
x130 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x138, x130)
end = time.time()
print(end-start)
