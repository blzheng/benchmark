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

    def forward(self, x314, x319):
        x320=operator.mul(x314, x319)
        return x320

m = M().eval()
x314 = torch.randn(torch.Size([1, 1248, 7, 7]))
x319 = torch.randn(torch.Size([1, 1248, 1, 1]))
start = time.time()
output = m(x314, x319)
end = time.time()
print(end-start)
