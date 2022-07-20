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

    def forward(self, x719, x715):
        x720=x719.sigmoid()
        x721=operator.mul(x715, x720)
        return x721

m = M().eval()
x719 = torch.randn(torch.Size([1, 2304, 1, 1]))
x715 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x719, x715)
end = time.time()
print(end-start)
