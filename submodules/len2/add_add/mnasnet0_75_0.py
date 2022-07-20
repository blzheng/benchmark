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

    def forward(self, x24, x16, x33):
        x25=operator.add(x24, x16)
        x34=operator.add(x33, x25)
        return x34

m = M().eval()
x24 = torch.randn(torch.Size([1, 24, 56, 56]))
x16 = torch.randn(torch.Size([1, 24, 56, 56]))
x33 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x24, x16, x33)
end = time.time()
print(end-start)
