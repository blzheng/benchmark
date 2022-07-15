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

    def forward(self, x55, x49):
        x56=operator.add(x55, x49)
        return x56

m = M().eval()
x55 = torch.randn(torch.Size([1, 64, 56, 56]))
x49 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x55, x49)
end = time.time()
print(end-start)
