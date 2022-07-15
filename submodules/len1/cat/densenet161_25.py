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

    def forward(self, x140, x147, x154, x161, x168, x175):
        x176=torch.cat([x140, x147, x154, x161, x168, x175], 1)
        return x176

m = M().eval()
x140 = torch.randn(torch.Size([1, 384, 14, 14]))
x147 = torch.randn(torch.Size([1, 48, 14, 14]))
x154 = torch.randn(torch.Size([1, 48, 14, 14]))
x161 = torch.randn(torch.Size([1, 48, 14, 14]))
x168 = torch.randn(torch.Size([1, 48, 14, 14]))
x175 = torch.randn(torch.Size([1, 48, 14, 14]))
start = time.time()
output = m(x140, x147, x154, x161, x168, x175)
end = time.time()
print(end-start)
