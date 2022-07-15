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

    def forward(self, x324, x319):
        x325=operator.mul(x324, x319)
        return x325

m = M().eval()
x324 = torch.randn(torch.Size([1, 576, 1, 1]))
x319 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x324, x319)
end = time.time()
print(end-start)
