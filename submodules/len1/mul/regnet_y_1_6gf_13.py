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

    def forward(self, x228, x223):
        x229=operator.mul(x228, x223)
        return x229

m = M().eval()
x228 = torch.randn(torch.Size([1, 336, 1, 1]))
x223 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x228, x223)
end = time.time()
print(end-start)
