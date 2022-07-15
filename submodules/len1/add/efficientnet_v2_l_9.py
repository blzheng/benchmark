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

    def forward(self, x69, x63):
        x70=operator.add(x69, x63)
        return x70

m = M().eval()
x69 = torch.randn(torch.Size([1, 64, 56, 56]))
x63 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x69, x63)
end = time.time()
print(end-start)
