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

    def forward(self, x210, x195):
        x211=operator.add(x210, x195)
        return x211

m = M().eval()
x210 = torch.randn(torch.Size([1, 192, 14, 14]))
x195 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x210, x195)
end = time.time()
print(end-start)
