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

    def forward(self, x110, x106):
        x111=x110.sigmoid()
        x112=operator.mul(x106, x111)
        return x112

m = M().eval()
x110 = torch.randn(torch.Size([1, 240, 1, 1]))
x106 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x110, x106)
end = time.time()
print(end-start)
