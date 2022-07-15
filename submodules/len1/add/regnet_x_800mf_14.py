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

    def forward(self, x151, x159):
        x160=operator.add(x151, x159)
        return x160

m = M().eval()
x151 = torch.randn(torch.Size([1, 672, 7, 7]))
x159 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x151, x159)
end = time.time()
print(end-start)
