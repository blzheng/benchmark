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

    def forward(self, x1):
        x2=x1.permute(0, 2, 3, 1)
        return x2

m = M().eval()
x1 = torch.randn(torch.Size([1, 96, 56, 56]))
start = time.time()
output = m(x1)
end = time.time()
print(end-start)
