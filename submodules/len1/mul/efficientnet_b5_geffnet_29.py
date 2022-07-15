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

    def forward(self, x432, x437):
        x438=operator.mul(x432, x437)
        return x438

m = M().eval()
x432 = torch.randn(torch.Size([1, 1824, 7, 7]))
x437 = torch.randn(torch.Size([1, 1824, 1, 1]))
start = time.time()
output = m(x432, x437)
end = time.time()
print(end-start)
