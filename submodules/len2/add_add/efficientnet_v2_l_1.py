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

    def forward(self, x62, x56, x69):
        x63=operator.add(x62, x56)
        x70=operator.add(x69, x63)
        return x70

m = M().eval()
x62 = torch.randn(torch.Size([1, 64, 56, 56]))
x56 = torch.randn(torch.Size([1, 64, 56, 56]))
x69 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x62, x56, x69)
end = time.time()
print(end-start)
