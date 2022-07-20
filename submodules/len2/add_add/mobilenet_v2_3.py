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

    def forward(self, x120, x128, x137):
        x129=operator.add(x120, x128)
        x138=operator.add(x129, x137)
        return x138

m = M().eval()
x120 = torch.randn(torch.Size([1, 160, 7, 7]))
x128 = torch.randn(torch.Size([1, 160, 7, 7]))
x137 = torch.randn(torch.Size([1, 160, 7, 7]))
start = time.time()
output = m(x120, x128, x137)
end = time.time()
print(end-start)
