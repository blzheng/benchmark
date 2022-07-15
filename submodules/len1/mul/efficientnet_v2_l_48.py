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

    def forward(self, x890, x885):
        x891=operator.mul(x890, x885)
        return x891

m = M().eval()
x890 = torch.randn(torch.Size([1, 2304, 1, 1]))
x885 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x890, x885)
end = time.time()
print(end-start)
