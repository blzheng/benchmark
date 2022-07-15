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

    def forward(self, x88, x94, x100, x104):
        x105=torch.cat([x88, x94, x100, x104], 1)
        return x105

m = M().eval()
x88 = torch.randn(torch.Size([1, 160, 14, 14]))
x94 = torch.randn(torch.Size([1, 224, 14, 14]))
x100 = torch.randn(torch.Size([1, 64, 14, 14]))
x104 = torch.randn(torch.Size([1, 64, 14, 14]))
start = time.time()
output = m(x88, x94, x100, x104)
end = time.time()
print(end-start)
