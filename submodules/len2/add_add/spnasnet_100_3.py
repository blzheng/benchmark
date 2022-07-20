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

    def forward(self, x145, x136, x155):
        x146=operator.add(x145, x136)
        x156=operator.add(x155, x146)
        return x156

m = M().eval()
x145 = torch.randn(torch.Size([1, 96, 14, 14]))
x136 = torch.randn(torch.Size([1, 96, 14, 14]))
x155 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x145, x136, x155)
end = time.time()
print(end-start)
