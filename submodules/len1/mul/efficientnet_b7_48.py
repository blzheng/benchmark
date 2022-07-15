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

    def forward(self, x758, x753):
        x759=operator.mul(x758, x753)
        return x759

m = M().eval()
x758 = torch.randn(torch.Size([1, 2304, 1, 1]))
x753 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x758, x753)
end = time.time()
print(end-start)
