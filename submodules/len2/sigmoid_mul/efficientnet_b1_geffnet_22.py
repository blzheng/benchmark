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

    def forward(self, x332, x328):
        x333=x332.sigmoid()
        x334=operator.mul(x328, x333)
        return x334

m = M().eval()
x332 = torch.randn(torch.Size([1, 1920, 1, 1]))
x328 = torch.randn(torch.Size([1, 1920, 7, 7]))
start = time.time()
output = m(x332, x328)
end = time.time()
print(end-start)
