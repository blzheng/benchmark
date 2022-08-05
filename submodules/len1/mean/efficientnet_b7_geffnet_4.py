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

    def forward(self, x60):
        x61=x60.mean((2, 3),keepdim=True)
        return x61

m = M().eval()
x60 = torch.randn(torch.Size([1, 192, 56, 56]))
start = time.time()
output = m(x60)
end = time.time()
print(end-start)
