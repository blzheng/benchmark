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

    def forward(self, x179, x164, x195):
        x180=operator.add(x179, x164)
        x196=operator.add(x195, x180)
        return x196

m = M().eval()
x179 = torch.randn(torch.Size([1, 64, 28, 28]))
x164 = torch.randn(torch.Size([1, 64, 28, 28]))
x195 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x179, x164, x195)
end = time.time()
print(end-start)
