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

    def forward(self, x170, x166):
        x171=x170.sigmoid()
        x172=operator.mul(x166, x171)
        return x172

m = M().eval()
x170 = torch.randn(torch.Size([1, 480, 1, 1]))
x166 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x170, x166)
end = time.time()
print(end-start)
