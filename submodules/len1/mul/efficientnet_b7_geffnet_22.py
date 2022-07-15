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

    def forward(self, x327, x332):
        x333=operator.mul(x327, x332)
        return x333

m = M().eval()
x327 = torch.randn(torch.Size([1, 960, 14, 14]))
x332 = torch.randn(torch.Size([1, 960, 1, 1]))
start = time.time()
output = m(x327, x332)
end = time.time()
print(end-start)
