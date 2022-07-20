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

    def forward(self, x12, x8, x17):
        x13=operator.add(x12, x8)
        x18=operator.add(x17, x13)
        return x18

m = M().eval()
x12 = torch.randn(torch.Size([1, 24, 112, 112]))
x8 = torch.randn(torch.Size([1, 24, 112, 112]))
x17 = torch.randn(torch.Size([1, 24, 112, 112]))
start = time.time()
output = m(x12, x8, x17)
end = time.time()
print(end-start)
