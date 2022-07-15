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

    def forward(self, x42, x37):
        x43=operator.mul(x42, x37)
        return x43

m = M().eval()
x42 = torch.randn(torch.Size([1, 96, 1, 1]))
x37 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x42, x37)
end = time.time()
print(end-start)
