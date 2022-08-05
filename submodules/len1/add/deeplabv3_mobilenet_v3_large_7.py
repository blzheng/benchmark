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

    def forward(self, x135, x121):
        x136=operator.add(x135, x121)
        return x136

m = M().eval()
x135 = torch.randn(torch.Size([1, 112, 14, 14]))
x121 = torch.randn(torch.Size([1, 112, 14, 14]))
start = time.time()
output = m(x135, x121)
end = time.time()
print(end-start)
