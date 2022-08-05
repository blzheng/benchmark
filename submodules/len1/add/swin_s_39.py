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

    def forward(self, x471, x478):
        x479=operator.add(x471, x478)
        return x479

m = M().eval()
x471 = torch.randn(torch.Size([1, 14, 14, 384]))
x478 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x471, x478)
end = time.time()
print(end-start)
