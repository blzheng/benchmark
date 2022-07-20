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

    def forward(self, x510, x506):
        x511=x510.sigmoid()
        x512=operator.mul(x506, x511)
        return x512

m = M().eval()
x510 = torch.randn(torch.Size([1, 1344, 1, 1]))
x506 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x510, x506)
end = time.time()
print(end-start)
