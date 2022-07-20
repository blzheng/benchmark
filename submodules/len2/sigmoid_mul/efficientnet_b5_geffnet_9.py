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

    def forward(self, x139, x135):
        x140=x139.sigmoid()
        x141=operator.mul(x135, x140)
        return x141

m = M().eval()
x139 = torch.randn(torch.Size([1, 384, 1, 1]))
x135 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x139, x135)
end = time.time()
print(end-start)
